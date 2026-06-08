#include <RMC/io/AtomicNumbers.hpp>
#include <RMC/io/LammpsReader.hpp>
#include <boost/leaf/result.hpp>
#include <charconv>
#include <format>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <vector>

namespace RMC::io {

namespace {

std::string_view trim(std::string_view sv) {
  const auto b = sv.find_first_not_of(" \t\r\n");
  if (b == std::string_view::npos)
    return {};
  const auto e = sv.find_last_not_of(" \t\r\n");
  return sv.substr(b, e - b + 1);
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
  return (style == LammpsAtomStyle::Molecular ||
          style == LammpsAtomStyle::Full)
             ? 2
             : 1;
}

// The recognised body-section keywords. Hitting any of these ends the header
// and starts (or skips) a data block.
bool is_section_keyword(std::string_view kw) {
  static constexpr std::string_view kws[] = {
      "Atoms",       "Velocities",  "Masses",     "Bonds",
      "Angles",      "Dihedrals",   "Impropers",  "Pair Coeffs",
      "PairIJ Coeffs", "Bond Coeffs", "Angle Coeffs", "Dihedral Coeffs",
      "Improper Coeffs", "BondBond Coeffs", "Atom Type Labels"};
  for (auto k : kws)
    if (kw == k)
      return true;
  return false;
}

} // namespace

Result<LammpsData> read_lammps_data(const std::filesystem::path &path,
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

    // Otherwise it is a header datum: "<numbers...> <keyword words>".
    std::istringstream ss{std::string(content)};
    std::vector<std::string> tok;
    for (std::string w; ss >> w;)
      tok.push_back(std::move(w));
    if (tok.empty())
      continue;

    auto to_d = [](const std::string &s, double &v) {
      const auto [p, ec] =
          std::from_chars(s.data(), s.data() + s.size(), v);
      return ec == std::errc{} && p == s.data() + s.size();
    };

    if (tok.size() >= 2 && tok.back() == "atoms") {
      double v;
      if (to_d(tok[0], v))
        declared_atoms = static_cast<std::size_t>(v);
    } else if (tok.size() >= 4 && tok[tok.size() - 2] == "xlo" &&
               tok.back() == "xhi") {
      to_d(tok[0], lo[0]);
      to_d(tok[1], hi[0]);
      have_box[0] = true;
    } else if (tok.size() >= 4 && tok[tok.size() - 2] == "ylo" &&
               tok.back() == "yhi") {
      to_d(tok[0], lo[1]);
      to_d(tok[1], hi[1]);
      have_box[1] = true;
    } else if (tok.size() >= 4 && tok[tok.size() - 2] == "zlo" &&
               tok.back() == "zhi") {
      to_d(tok[0], lo[2]);
      to_d(tok[1], hi[2]);
      have_box[2] = true;
    } else if (tok.size() >= 6 && tok[tok.size() - 3] == "xy" &&
               tok[tok.size() - 2] == "xz" && tok.back() == "yz") {
      to_d(tok[0], tilt[0]);
      to_d(tok[1], tilt[1]);
      to_d(tok[2], tilt[2]);
      have_tilt = true;
    }
    // Anything else (atom types, bonds, masses count, ...) is ignored.
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

  // --- Locate the Atoms section, skipping the data of any section before it ---
  while (!section.empty() && section.rfind("Atoms", 0) != 0) {
    // Consume this section's data block until the next keyword (or EOF).
    section.clear();
    while (std::getline(file, line)) {
      const std::string_view t = trim(strip_comment(line));
      if (t.empty())
        continue;
      if (is_section_keyword(t)) {
        section = std::string(t); // next header found
        break;
      }
      // otherwise a data line of the section being skipped: ignore it
    }
  }

  if (section.rfind("Atoms", 0) != 0)
    return boost::leaf::new_error(
        std::string{"LAMMPS data file has no Atoms section: " + path.string()});

  // --- Parse the Atoms block ---
  AtomicStructure &s = out.structure;
  std::vector<std::array<double, 3>> xyz;
  std::vector<int> atom_numbers;
  const int off = coord_offset(style);
  const int tcol = type_column(style);
  std::unordered_map<int, std::string> type_symbol; // type -> element symbol

  auto element_for_type = [&](int type) -> const std::string & {
    auto it = type_symbol.find(type);
    if (it != type_symbol.end())
      return it->second;
    std::string sym;
    const std::size_t idx = static_cast<std::size_t>(type) - 1;
    if (type >= 1 && idx < type_to_element.size() &&
        !type_to_element[idx].empty())
      sym = type_to_element[idx];
    else
      sym = std::format("X{}", type);
    return type_symbol.emplace(type, std::move(sym)).first->second;
  };

  while (std::getline(file, line)) {
    const std::string_view content = trim(strip_comment(line));
    if (content.empty())
      continue;
    if (is_section_keyword(content))
      break; // reached the next section

    std::istringstream ss{std::string(content)};
    std::vector<std::string> tok;
    for (std::string w; ss >> w;)
      tok.push_back(std::move(w));

    if (static_cast<int>(tok.size()) < off + 3)
      return boost::leaf::new_error(std::format(
          "LAMMPS Atoms line has too few columns ({}, need {}): {}",
          tok.size(), off + 3, std::string(content)));

    auto to_d = [](const std::string &str, double &v) {
      const auto [p, ec] =
          std::from_chars(str.data(), str.data() + str.size(), v);
      return ec == std::errc{} && p == str.data() + str.size();
    };

    double tv = 0, x = 0, y = 0, z = 0;
    const bool ok = to_d(tok[static_cast<std::size_t>(tcol)], tv) &&
                    to_d(tok[static_cast<std::size_t>(off)], x) &&
                    to_d(tok[static_cast<std::size_t>(off) + 1], y) &&
                    to_d(tok[static_cast<std::size_t>(off) + 2], z);
    if (!ok)
      return boost::leaf::new_error(
          std::string{"LAMMPS Atoms line parse error: "} +
          std::string(content));

    const int type = static_cast<int>(tv);
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
  s.coordinates.resize(static_cast<Eigen::Index>(N), 3);
  for (std::size_t i = 0; i < N; ++i) {
    s.coordinates(static_cast<Eigen::Index>(i), 0) = xyz[i][0];
    s.coordinates(static_cast<Eigen::Index>(i), 1) = xyz[i][1];
    s.coordinates(static_cast<Eigen::Index>(i), 2) = xyz[i][2];
  }
  s.atomic_numbers = Eigen::Map<const ivec_t>(
      atom_numbers.data(), static_cast<Eigen::Index>(atom_numbers.size()));

  return out;
}

Result<void> write_lammps_data(const AtomicStructure &s, const mat3_t &box,
                               const std::filesystem::path &path,
                               const vec3_t &origin) {
  std::ofstream f(path);
  if (!f)
    return boost::leaf::new_error(
        std::string{"Cannot write LAMMPS data file: " + path.string()});

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
    if (inserted)
      legend.push_back(sym);
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
  for (std::size_t t = 0; t < legend.size(); ++t)
    f << std::format(" {}={}", t + 1, legend[t]);
  f << "\n";

  return {};
}

} // namespace RMC::io
