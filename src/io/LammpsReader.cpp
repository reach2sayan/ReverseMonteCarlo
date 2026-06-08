#include <RMC/io/AtomicNumbers.hpp>
#include <RMC/io/LammpsReader.hpp>
#include <boost/algorithm/string/classification.hpp> // boost::is_any_of
#include <boost/leaf/result.hpp>
#include <boost/parser/parser.hpp>
#include <format>
#include <fstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace RMC::io {

namespace {

namespace bp = boost::parser;

// Trim ASCII whitespace from both ends, returning a sub-view (no allocation).
// Uses a boost classification predicate rather than two find_*_not_of scans.
std::string_view trim(std::string_view sv) {
  static const auto is_ws = boost::is_any_of(" \t\r\n");
  while (!sv.empty() && is_ws(sv.front())) {
    sv.remove_prefix(1);
  }
  while (!sv.empty() && is_ws(sv.back())) {
    sv.remove_suffix(1);
  }
  return sv;
}

// Strip an inline `# ...` comment (LAMMPS allows them on most lines).
std::string_view strip_comment(std::string_view sv) {
  const auto h = sv.find('#');
  return (h == std::string_view::npos) ? sv : sv.substr(0, h);
}

// Number of columns before x,y,z for each atom_style.
int coord_offset(LammpsAtomStyle style) {
  switch (style) {
  case LammpsAtomStyle::Atomic:
    return 2; // id type
  case LammpsAtomStyle::Charge:
    return 3; // id type q
  case LammpsAtomStyle::Molecular:
    return 3; // id mol type
  case LammpsAtomStyle::Full:
    return 4; // id mol type q
  }
  return 2;
}

// Column index (0-based) holding the atom type for each style.
int type_column(LammpsAtomStyle style) {
  return (style == LammpsAtomStyle::Molecular || style == LammpsAtomStyle::Full)
             ? 2
             : 1;
}

// The recognised body-section keywords. Hitting any of these ends the header
// and starts (or skips) a data block.
bool is_section_keyword(std::string_view kw) {
  static constexpr std::string_view kws[] = {
      "Atoms",           "Velocities",      "Masses",
      "Bonds",           "Angles",          "Dihedrals",
      "Impropers",       "Pair Coeffs",     "PairIJ Coeffs",
      "Bond Coeffs",     "Angle Coeffs",    "Dihedral Coeffs",
      "Improper Coeffs", "BondBond Coeffs", "Atom Type Labels"};
  for (auto k : kws) {
    if (kw == k) {
      return true;
    }
  }
  return false;
}

} // namespace

Result<LammpsData>
read_lammps_data(const std::filesystem::path &path,
                 const std::vector<std::string> &type_to_element,
                 LammpsAtomStyle style) {
  std::ifstream file(path);
  if (!file)
    return boost::leaf::new_error(
        std::string{"Cannot open LAMMPS data file: " + path.string()});

  LammpsData out;
  std::size_t declared_atoms = 0;
  bool have_box[3] = {false, false, false};
  double lo[3] = {0, 0, 0}, hi[3] = {0, 0, 0};
  double tilt[3] = {0, 0, 0}; // xy, xz, yz
  bool have_tilt = false;

  std::string line;
  bool first = true;

  // Header-datum grammars. Each consumes the WHOLE line under the bp::ws
  // skipper, so the trailing keyword disambiguates — "atoms" can't match the
  // leading word of "atom types", which the hand-rolled version had to rule out
  // with tok[tok.size() - 2]. Lines matching none (atom types, bonds, masses
  // count, ...) fall through and are ignored, as before.
  const auto atoms_p = bp::ulong_ >> "atoms";                     // -> count
  const auto xlo_p = bp::double_ >> bp::double_ >> "xlo" >> "xhi"; // -> (lo,hi)
  const auto ylo_p = bp::double_ >> bp::double_ >> "ylo" >> "yhi";
  const auto zlo_p = bp::double_ >> bp::double_ >> "zlo" >> "zhi";
  const auto tilt_p = bp::double_ >> bp::double_ >> bp::double_ >> "xy" >>
                      "xz" >> "yz"; // -> (xy,xz,yz)

  // --- Header: keyword lines until the first section keyword ---
  std::string section;
  while (std::getline(file, line)) {
    if (first) { // line 1 is always a free-form comment
      first = false;
      continue;
    }
    const std::string_view content = trim(strip_comment(line));
    if (content.empty())
      continue;

    // A line whose trailing words are a section keyword starts the body.
    if (is_section_keyword(content)) {
      section = std::string(content);
      break;
    }

    // Parse the trimmed view directly — no per-line std::string or token-vector
    // copies. The first grammar that consumes the whole line wins.
    if (const auto n = bp::parse(content, atoms_p, bp::ws)) {
      declared_atoms = static_cast<std::size_t>(*n);
    } else if (const auto bx = bp::parse(content, xlo_p, bp::ws)) {
      const auto &[l, h] = *bx;
      lo[0] = l;
      hi[0] = h;
      have_box[0] = true;
    } else if (const auto by = bp::parse(content, ylo_p, bp::ws)) {
      const auto &[l, h] = *by;
      lo[1] = l;
      hi[1] = h;
      have_box[1] = true;
    } else if (const auto bz = bp::parse(content, zlo_p, bp::ws)) {
      const auto &[l, h] = *bz;
      lo[2] = l;
      hi[2] = h;
      have_box[2] = true;
    } else if (const auto t = bp::parse(content, tilt_p, bp::ws)) {
      const auto &[xy, xz, yz] = *t;
      tilt[0] = xy;
      tilt[1] = xz;
      tilt[2] = yz;
      have_tilt = true;
    }
  }

  if (!have_box[0] || !have_box[1] || !have_box[2])
    return boost::leaf::new_error(
        std::string{"LAMMPS data file missing box bounds: " + path.string()});

  // Build the box matrix (columns = cell edge vectors).
  const double lx = hi[0] - lo[0];
  const double ly = hi[1] - lo[1];
  const double lz = hi[2] - lo[2];
  out.box.setZero();
  out.box(0, 0) = lx;
  out.box(1, 1) = ly;
  out.box(2, 2) = lz;
  if (have_tilt) {
    out.box(0, 1) = tilt[0]; // xy
    out.box(0, 2) = tilt[1]; // xz
    out.box(1, 2) = tilt[2]; // yz
  }
  out.origin = vec3_t{lo[0], lo[1], lo[2]};

  // --- Locate the Atoms section, skipping the data of any section before it
  while (!section.empty() && section.rfind("Atoms", 0) != 0) {
    // Consume this section's data block until the next keyword (or EOF).
    section.clear();
    while (std::getline(file, line)) {
      const std::string_view t = trim(strip_comment(line));
      if (t.empty()) {
        continue;
      }
      if (is_section_keyword(t)) {
        section = std::string(t); // next header found
        break;
      }
      // otherwise a data line of the section being skipped: ignore it
    }
  }

  if (section.rfind("Atoms", 0) != 0) {
    return boost::leaf::new_error(
        std::string{"LAMMPS data file has no Atoms section: " + path.string()});
  }

  // --- Parse the Atoms block ---
  AtomicStructure &s = out.structure;
  std::vector<std::array<double, 3>> xyz;
  std::vector<int> atom_numbers;
  const int off = coord_offset(style);
  const int tcol = type_column(style);
  std::unordered_map<int, std::string> type_symbol; // type -> element symbol

  auto element_for_type = [&](int type) -> const std::string & {
    auto it = type_symbol.find(type);
    if (it != type_symbol.end()) {
      return it->second;
    }

    const std::size_t idx = static_cast<std::size_t>(type) - 1;
    std::string sym = (type >= 1 && idx < type_to_element.size() &&
           !type_to_element[idx].empty())
              ? type_to_element[idx]
              : std::format("X{}", type);

    return type_symbol.emplace(type, std::move(sym)).first->second;
  };

  // Every column we read (id, mol, type, charge, x, y, z, and any trailing
  // image flags) is numeric, so parse the whole line as a run of reals straight
  // from the view — no token vector, no per-field from_chars — then index by
  // atom_style.
  while (std::getline(file, line)) {
    const std::string_view content = trim(strip_comment(line));
    if (content.empty()) {
      continue;
    }
    if (is_section_keyword(content)) {
      break; // reached the next section
    }

    const auto cols = bp::parse(content, +bp::double_, bp::ws);
    if (!cols) {
      return boost::leaf::new_error(
          std::format("LAMMPS Atoms line parse error: {}", content));
    }
    if (static_cast<int>(cols->size()) < off + 3) {
      return boost::leaf::new_error(
          std::format("LAMMPS Atoms line has too few columns ({}, need {}): {}",
                      cols->size(), off + 3, content));
    }

    const int type = static_cast<int>((*cols)[static_cast<std::size_t>(tcol)]);
    const double x = (*cols)[static_cast<std::size_t>(off)];
    const double y = (*cols)[static_cast<std::size_t>(off) + 1];
    const double z = (*cols)[static_cast<std::size_t>(off) + 2];
    const std::string &sym = element_for_type(type);

    xyz.push_back({x, y, z});
    s.names.push_back(sym);
    s.elements.push_back(sym);
    s.residues.push_back(sym);
    s.molecule_ids.push_back(1);

    const int anum = atomic_number(sym);
    atom_numbers.push_back(anum != 0 ? anum : type);
  }

  if (declared_atoms != 0 && xyz.size() != declared_atoms)
    return boost::leaf::new_error(std::format(
        "LAMMPS data file declared {} atoms but Atoms section has {}",
        declared_atoms, xyz.size()));

  const std::size_t N = xyz.size();
  // xyz is a contiguous std::vector<std::array<double,3>>; map it directly into
  // the row-major coordinate matrix instead of copying component by component.
  s.coordinates = Eigen::Map<const coords_t>(
      reinterpret_cast<const double *>(xyz.data()),
      static_cast<Eigen::Index>(N), 3);
  s.atomic_numbers = Eigen::Map<const ivec_t>(
      atom_numbers.data(), static_cast<Eigen::Index>(atom_numbers.size()));

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

  // Assign integer types by first appearance of each element symbol.
  std::unordered_map<std::string, int> sym_to_type;
  std::vector<std::string> legend; // type i -> legend[i-1]
  std::vector<int> types;
  types.reserve(s.size());
  for (std::size_t i = 0; i < s.size(); ++i) {
    const std::string sym =
        (i < s.elements.size() && !s.elements[i].empty()) ? s.elements[i] : "X";
    auto [it, inserted] =
        sym_to_type.try_emplace(sym, static_cast<int>(legend.size()) + 1);
    if (inserted) {
      legend.push_back(sym);
    }
    types.push_back(it->second);
  }

  f << "LAMMPS data file written by RMC\n\n";
  f << std::format("{} atoms\n", s.size());
  f << std::format("{} atom types\n\n", legend.size());
  f << std::format("{:.8f} {:.8f} xlo xhi\n", origin.x(),
                   origin.x() + box(0, 0));
  f << std::format("{:.8f} {:.8f} ylo yhi\n", origin.y(),
                   origin.y() + box(1, 1));
  f << std::format("{:.8f} {:.8f} zlo zhi\n", origin.z(),
                   origin.z() + box(2, 2));
  if (box(0, 1) != 0.0 || box(0, 2) != 0.0 || box(1, 2) != 0.0)
    f << std::format("{:.8f} {:.8f} {:.8f} xy xz yz\n", box(0, 1), box(0, 2),
                     box(1, 2));

  f << "\nAtoms # atomic\n\n";
  for (std::size_t i = 0; i < s.size(); ++i) {
    f << std::format("{} {} {:.8f} {:.8f} {:.8f}\n", i + 1, types[i],
                     s.coordinates(static_cast<Eigen::Index>(i), 0),
                     s.coordinates(static_cast<Eigen::Index>(i), 1),
                     s.coordinates(static_cast<Eigen::Index>(i), 2));
  }

  // Legend so the file round-trips through read_lammps_data(type_to_element).
  f << "\n# type -> element:";
  for (std::size_t t = 0; t < legend.size(); ++t) {
    f << std::format(" {}={}", t + 1, legend[t]);
  }
  f << "\n";

  return {};
}

} // namespace RMC::io
