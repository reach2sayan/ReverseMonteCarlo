#include "TextParse.hpp"

#include <seitz/data/element_data.hpp>
#include <RMC/io/LammpsReader.hpp>
#include <algorithm>
#include <array>
#include <boost/leaf/result.hpp>
#include <boost/parser/parser.hpp>
#include <format>
#include <fstream>
#include <functional>
#include <ranges>
#include <string>
#include <string_view>
#include <vector>

namespace RMC::io {

namespace {

namespace bp = boost::parser;
using detail::strip_comment;
using detail::trim;

// Atoms-section layout per atom_style (indexed by LammpsAtomStyle): the
// 0-based column of the atom type and of x (y and z follow).
struct StyleColumns {
  std::size_t type, x;
};
constexpr std::array<StyleColumns, 4> kColumns{{
    {1, 2}, // Atomic    : id type x y z
    {1, 3}, // Charge    : id type q x y z
    {2, 3}, // Molecular : id mol type x y z
    {2, 4}, // Full      : id mol type q x y z
}};

// "<lo> <hi> xlo xhi" etc.: the header datum that bounds each box axis.
constexpr std::array<std::string_view, 3> kAxisKeys{"xlo xhi", "ylo yhi",
                                                    "zlo zhi"};

// The recognised body-section keywords. Hitting any of these ends the header
// and starts (or skips) a data block.
bool is_section_keyword(std::string_view kw) {
  static constexpr std::string_view kws[] = {
      "Atoms",           "Velocities",      "Masses",
      "Bonds",           "Angles",          "Dihedrals",
      "Impropers",       "Pair Coeffs",     "PairIJ Coeffs",
      "Bond Coeffs",     "Angle Coeffs",    "Dihedral Coeffs",
      "Improper Coeffs", "BondBond Coeffs", "Atom Type Labels"};
  return std::ranges::contains(kws, kw);
}

} // namespace

Result<LammpsData>
read_lammps_data(const std::filesystem::path &path,
                 const std::vector<std::string> &type_to_element,
                 LammpsAtomStyle style) {
  std::ifstream file(path);
  if (!file) {
    return boost::leaf::new_error(
        std::string{"Cannot open LAMMPS data file: " + path.string()});
  }

  LammpsData out;
  std::size_t declared_atoms = 0;
  vec3_t lo = vec3_t::Zero(), hi = vec3_t::Zero();
  std::array<bool, 3> have_axis{};
  detail::StructureBuilder atoms;
  const auto [tcol, xcol] = kColumns[static_cast<std::size_t>(style)];

  // Header datum: numbers, then keywords ("0.0 10.0 xlo xhi", "3 atoms").
  const auto word_p = bp::lexeme[+(bp::char_ - bp::char_(" \t\r\n"))];
  const auto header_p = +bp::double_ >> +word_p;
  std::vector<double> cols; // one Atoms line, reused

  // Line 1 is a free-form comment; then the header, then sections. Only the
  // Atoms section is read; the data of every other section is skipped.
  enum class Block { Header, Atoms, Other } block = Block::Header;
  std::string line;
  std::getline(file, line);
  while (std::getline(file, line)) {
    const std::string_view content = trim(strip_comment(line));
    if (content.empty()) {
      continue;
    }
    if (is_section_keyword(content)) {
      if (block == Block::Atoms) {
        break;
      }
      block = content == "Atoms" ? Block::Atoms : Block::Other;
      continue;
    }

    if (block == Block::Header) {
      const auto datum = bp::parse(content, header_p, bp::ws);
      if (!datum) {
        continue; // not a header datum we use
      }
      const auto &[nums, words] = *datum;
      const auto key = words | std::views::join_with(' ') |
                       std::ranges::to<std::string>();
      const auto axis = std::ranges::find(kAxisKeys, key);
      if (key == "atoms") {
        declared_atoms = static_cast<std::size_t>(nums[0]);
      } else if (axis != kAxisKeys.end() && nums.size() == 2) {
        const auto d = axis - kAxisKeys.begin();
        lo[d] = nums[0];
        hi[d] = nums[1];
        have_axis[static_cast<std::size_t>(d)] = true;
      } else if (key == "xy xz yz" && nums.size() == 3) {
        out.box(0, 1) = nums[0];
        out.box(0, 2) = nums[1];
        out.box(1, 2) = nums[2];
      }
    } else if (block == Block::Atoms) {
      // Every column is numeric; index the run of reals by atom_style.
      cols.clear();
      if (!bp::parse(content, +bp::double_, bp::ws, cols)) {
        return boost::leaf::new_error(
            std::format("LAMMPS Atoms line parse error: {}", content));
      }
      if (cols.size() < xcol + 3) {
        return boost::leaf::new_error(
            std::format("LAMMPS Atoms line has too few columns ({}, need {}): {}",
                        cols.size(), xcol + 3, content));
      }
      const int type = static_cast<int>(cols[tcol]);
      const auto idx = static_cast<std::size_t>(type - 1);
      const std::string sym =
          type >= 1 && idx < type_to_element.size() &&
                  !type_to_element[idx].empty()
              ? type_to_element[idx]
              : std::format("X{}", type);
      const int z = seitz::data::atomic_number(sym).value_or(0);
      atoms.add({cols[xcol], cols[xcol + 1], cols[xcol + 2]}, z != 0 ? z : type,
                sym);
    }
  }

  if (!std::ranges::all_of(have_axis, std::identity{})) {
    return boost::leaf::new_error(
        std::string{"LAMMPS data file missing box bounds: " + path.string()});
  }
  if (block != Block::Atoms) {
    return boost::leaf::new_error(
        std::string{"LAMMPS data file has no Atoms section: " + path.string()});
  }
  if (declared_atoms != 0 && atoms.size() != declared_atoms) {
    return boost::leaf::new_error(std::format(
        "LAMMPS data file declared {} atoms but Atoms section has {}",
        declared_atoms, atoms.size()));
  }

  // Box columns are the cell edge vectors; tilts were stored above.
  out.box.diagonal() = hi - lo;
  out.origin = lo;
  out.structure = std::move(atoms).finish();
  return out;
}

Result<void> write_lammps_data(const AtomicStructure &s, const mat3_t &box,
                               const std::filesystem::path &path,
                               const vec3_t &origin) {
  std::ofstream f(path);
  if (!f) {
    return boost::leaf::new_error(
        std::string{"Cannot write LAMMPS data file: " + path.string()});
  }

  // Integer types by first appearance of each element symbol.
  const auto groups = detail::group_by_element(s);
  std::vector<std::size_t> types(s.size());
  for (const auto &[t, g] : groups | std::views::enumerate) {
    for (const std::size_t i : g.atoms) {
      types[i] = static_cast<std::size_t>(t) + 1;
    }
  }

  f << "LAMMPS data file written by RMC\n\n";
  f << std::format("{} atoms\n{} atom types\n\n", s.size(), groups.size());
  for (int d = 0; d < 3; ++d) {
    f << std::format("{:.8f} {:.8f} {}\n", origin[d], origin[d] + box(d, d),
                     kAxisKeys[static_cast<std::size_t>(d)]);
  }
  if (box(0, 1) != 0.0 || box(0, 2) != 0.0 || box(1, 2) != 0.0) {
    f << std::format("{:.8f} {:.8f} {:.8f} xy xz yz\n", box(0, 1), box(0, 2),
                     box(1, 2));
  }

  f << "\nAtoms # atomic\n\n";
  for (std::size_t i = 0; i < s.size(); ++i) {
    const auto r = static_cast<Eigen::Index>(i);
    f << std::format("{} {} {:.8f} {:.8f} {:.8f}\n", i + 1, types[i],
                     s.coordinates(r, 0), s.coordinates(r, 1),
                     s.coordinates(r, 2));
  }

  // Legend so the file round-trips through read_lammps_data(type_to_element).
  f << "\n# type -> element:";
  for (const auto &[t, g] : groups | std::views::enumerate) {
    f << std::format(" {}={}", t + 1, g.symbol);
  }
  f << "\n";
  return {};
}

} // namespace RMC::io
