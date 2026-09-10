#include "TextParse.hpp"

#include <seitz/data/element_data.hpp>
#include <RMC/io/PdbReader.hpp>
#include <algorithm>
#include <boost/algorithm/string/case_conv.hpp>
#include <boost/leaf/result.hpp>
#include <cctype>
#include <charconv>
#include <format>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

namespace RMC::io {

namespace {

// ATOM/HETATM field offsets (PDB fixed-format columns, 0-based).
// See:
// https://www.wwpdb.org/documentation/file-format-content/format33/sect9.html
constexpr std::size_t NAME_START = 12;
constexpr std::size_t NAME_LEN = 4;
constexpr std::size_t RESNAME_START = 17;
constexpr std::size_t RESNAME_LEN = 3;
constexpr std::size_t CHAINID = 21;
constexpr std::size_t RESSEQ_START = 22;
constexpr std::size_t RESSEQ_LEN = 4;
constexpr std::size_t X_START = 30;
constexpr std::size_t Y_START = 38;
constexpr std::size_t Z_START = 46;
constexpr std::size_t COORD_LEN = 8;
constexpr std::size_t ELEMENT_START = 76;
constexpr std::size_t ELEMENT_LEN = 2;

// A blank-padded numeric field. Throws on a missing or malformed number, which
// read_pdb maps to a leaf error.
template <class T> T parse_field(std::string_view sv) {
  const auto s = detail::trim(sv);
  T v{};
  if (std::from_chars(s.data(), s.data() + s.size(), v).ec != std::errc{}) {
    throw std::runtime_error("invalid number in PDB numeric field");
  }
  return v;
}

} // namespace

Result<AtomicStructure> read_pdb(const std::filesystem::path &path) {
  std::ifstream file(path);
  if (!file) {
    return boost::leaf::new_error(
        std::string{"Cannot open PDB file: " + path.string()});
  }
  detail::StructureBuilder atoms;
  std::size_t mol_id = 0;
  std::optional<std::pair<char, int>> prev_residue; // (chain, resSeq)

  for (std::string line; std::getline(file, line);) {
    if (line.size() < 54 ||
        !(line.starts_with("ATOM") || line.starts_with("HETATM"))) {
      continue;
    }
    const std::string_view sv(line);
    try {
      const std::array<double, 3> xyz{
          parse_field<double>(sv.substr(X_START, COORD_LEN)),
          parse_field<double>(sv.substr(Y_START, COORD_LEN)),
          parse_field<double>(sv.substr(Z_START, COORD_LEN))};
      const std::string_view atom_name =
          detail::trim(sv.substr(NAME_START, NAME_LEN));

      // Element: columns 77-78, else the name's first letter ("1HB" → "H");
      // capitalised as "Xx".
      std::string element{sv.size() > ELEMENT_START
                              ? detail::trim(sv.substr(ELEMENT_START, ELEMENT_LEN))
                              : std::string_view{}};
      if (element.empty()) {
        const auto it = std::ranges::find_if(
            atom_name, [](unsigned char c) { return std::isalpha(c); });
        if (it != atom_name.end()) {
          element = *it;
        }
      }
      boost::algorithm::to_lower(element);
      if (!element.empty()) {
        element[0] = static_cast<char>(
            std::toupper(static_cast<unsigned char>(element[0])));
      }

      // A new (chain, resSeq) pair starts a new molecule.
      const std::pair residue{
          line[CHAINID], parse_field<int>(sv.substr(RESSEQ_START, RESSEQ_LEN))};
      if (residue != prev_residue) {
        ++mol_id;
        prev_residue = residue;
      }

      const int z = seitz::data::atomic_number(element).value_or(0);
      atoms.add(xyz, z, element, std::string(atom_name),
                std::string(detail::trim(sv.substr(RESNAME_START, RESNAME_LEN))),
                mol_id);
    } catch (const std::exception &e) {
      return boost::leaf::new_error(std::string{"PDB parse error: "} +
                                    e.what());
    }
  }
  return std::move(atoms).finish();
}

Result<void> write_pdb(const AtomicStructure &s,
                       const std::filesystem::path &path) {
  std::ofstream f(path);
  if (!f) {
    return boost::leaf::new_error(
        std::string{"Cannot write PDB: " + path.string()});
  }
  for (std::size_t i = 0; i < s.size(); ++i) {
    const auto r = static_cast<Eigen::Index>(i);
    f << std::format(
        "ATOM  {:5d} {:<4.4} {:<3.3} A{:4d}    {:8.3f}{:8.3f}{:8.3f}"
        "  1.00  0.00          {:>2.2}\n",
        i + 1, detail::at_or(s.names, i, "X"),
        detail::at_or(s.residues, i, "UNK"),
        detail::at_or(s.molecule_ids, i, 1), s.coordinates(r, 0),
        s.coordinates(r, 1), s.coordinates(r, 2),
        detail::at_or(s.elements, i, ""));
  }
  f << "END\n";
  return {};
}

} // namespace RMC::io
