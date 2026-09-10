#include <RMC/io/StructFormat.hpp>

#include <RMC/io/LammpsReader.hpp>
#include <RMC/io/PdbReader.hpp>
#include <RMC/io/VaspReader.hpp>

#include <boost/leaf.hpp>
#include <boost/leaf/result.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <span>
#include <string_view>

namespace RMC::io {
namespace {

// One rule per format: the extensions and the bare filenames (stems) that map
// to it. Matching is case-insensitive, so the table lists lowercase forms only.
// Extensions are written with their leading dot to match path::extension().
struct FormatRule {
  StructFormat fmt;
  std::span<const std::string_view> extensions;
  std::span<const std::string_view> stems;
};

constexpr std::string_view kPdbExt[] = {".pdb"};
constexpr std::string_view kVaspExt[] = {".vasp", ".poscar"};
constexpr std::string_view kVaspStem[] = {"poscar", "contcar"};
constexpr std::string_view kLammpsExt[] = {".lammps", ".lmp", ".data"};

constexpr std::array<FormatRule, 3> kRules{{
    {StructFormat::Pdb, kPdbExt, {}},
    {StructFormat::Vasp, kVaspExt, kVaspStem},
    {StructFormat::Lammps, kLammpsExt, {}},
}};

std::string to_lower(std::string_view s) {
  std::string out(s);
  std::ranges::transform(out, out.begin(),
                         [](unsigned char c) { return std::tolower(c); });
  return out;
}

} // namespace

std::optional<StructFormat>
classify_structure_format(const std::filesystem::path &p) {
  const std::string ext = to_lower(p.extension().string());
  const std::string stem = to_lower(p.filename().string());
  for (const auto &rule : kRules) {
    if (std::ranges::contains(rule.extensions, ext) ||
        std::ranges::contains(rule.stems, stem)) {
      return rule.fmt;
    }
  }
  return std::nullopt;
}

Result<LoadedStructure>
read_structure_by_ext(const std::filesystem::path &path,
                      const BoundaryConditions &default_bc,
                      const std::vector<std::string> &type_to_element) {
  return read_structure(
      path, classify_structure_format(path).value_or(StructFormat::Lammps),
      default_bc, type_to_element);
}

Result<LoadedStructure>
read_structure(const std::filesystem::path &path, StructFormat fmt,
               const BoundaryConditions &default_bc,
               const std::vector<std::string> &type_to_element) {
  switch (fmt) {
  case StructFormat::Pdb: {
    BOOST_LEAF_AUTO(s, read_pdb(path));
    return LoadedStructure{std::move(s), default_bc};
  }
  case StructFormat::Vasp: {
    BOOST_LEAF_AUTO(data, read_vasp(path));
    auto bc = data.periodic_bc();
    return LoadedStructure{std::move(data.structure), bc};
  }
  case StructFormat::Lammps:
    break;
  }
  BOOST_LEAF_AUTO(data, read_lammps_data(path, type_to_element));
  auto bc = data.periodic_bc();
  return LoadedStructure{std::move(data.structure), bc};
}

} // namespace RMC::io
