#include <algorithm>
#include <charconv>
#include <fstream>
#include <fullrmc/io/PdbReader.hpp>
#include <sstream>
#include <string>
#include <unordered_map>

namespace fullrmc::io {

namespace {

// ATOM/HETATM field offsets (PDB fixed-format columns, 0-based).
// See:
// https://www.wwpdb.org/documentation/file-format-content/format33/sect9.html
constexpr int SERIAL_START = 6;
constexpr int SERIAL_LEN = 5;
constexpr int NAME_START = 12;
constexpr int NAME_LEN = 4;
constexpr int RESNAME_START = 17;
constexpr int RESNAME_LEN = 3;
constexpr int CHAINID = 21;
constexpr int RESSEQ_START = 22;
constexpr int RESSEQ_LEN = 4;
constexpr int X_START = 30;
constexpr int Y_START = 38;
constexpr int Z_START = 46;
constexpr int COORD_LEN = 8;
constexpr int ELEMENT_START = 76;
constexpr int ELEMENT_LEN = 2;

// Map common element symbols to atomic numbers.
const std::unordered_map<std::string, int> ATOMIC_NUMBERS = {
    {"H", 1},   {"C", 6},   {"N", 7},   {"O", 8},   {"F", 9},   {"P", 15},
    {"S", 16},  {"Cl", 17}, {"Ar", 18}, {"K", 19},  {"Ca", 20}, {"Fe", 26},
    {"Ni", 28}, {"Cu", 29}, {"Zn", 30}, {"Br", 35}, {"I", 53},  {"Pb", 82}};

std::string trim(std::string_view sv) {
  auto b = sv.find_first_not_of(' ');
  if (b == std::string_view::npos)
    return "";
  auto e = sv.find_last_not_of(' ');
  return std::string(sv.substr(b, e - b + 1));
}

double parse_real(std::string_view sv) {
  // trim, then stod for portability
  std::string s(trim(sv));
  return std::stod(s);
}

} // namespace

Result<AtomicStructure> read_pdb(const std::filesystem::path &path) {
  std::ifstream file(path);
  if (!file)
    return std::unexpected("Cannot open PDB file: " + path.string());

  AtomicStructure s;
  std::vector<std::array<double, 3>> xyz;

  std::string line;
  std::size_t mol_id = 0;
  std::size_t prev_resseq = -1;
  std::string prev_chain;

  while (std::getline(file, line)) {
    if (line.size() < 54)
      continue;
    bool is_atom = (line.substr(0, 4) == "ATOM");
    bool is_het = (line.substr(0, 6) == "HETATM");
    if (!is_atom && !is_het)
      continue;

    try {
      double x = parse_real(line.substr(X_START, COORD_LEN));
      double y = parse_real(line.substr(Y_START, COORD_LEN));
      double z = parse_real(line.substr(Z_START, COORD_LEN));
      xyz.push_back({x, y, z});

      std::string atom_name = trim(line.substr(NAME_START, NAME_LEN));
      std::string resname = trim(line.substr(RESNAME_START, RESNAME_LEN));

      // Derive element: prefer column 77-78, fall back to first char of name.
      std::string element;
      if (line.size() >= static_cast<std::size_t>(ELEMENT_START + ELEMENT_LEN))
        element = trim(line.substr(ELEMENT_START, ELEMENT_LEN));
      if (element.empty() && !atom_name.empty()) {
        // Strip leading digits (e.g. "1HB" → "H").
        for (char c : atom_name)
          if (std::isalpha(c)) {
            element = std::string(1, c);
            break;
          }
      }
      // Capitalize first letter, lowercase rest.
      if (!element.empty()) {
        element[0] = static_cast<char>(std::toupper(element[0]));
        for (std::size_t k = 1; k < element.size(); ++k)
          element[k] = static_cast<char>(std::tolower(element[k]));
      }

      std::string chain(1, line[CHAINID]);
      std::size_t resseq = 0;
      if (line.size() >= static_cast<std::size_t>(RESSEQ_START + RESSEQ_LEN))
        resseq = std::stoi(line.substr(RESSEQ_START, RESSEQ_LEN));

      if (chain != prev_chain || resseq != prev_resseq) {
        ++mol_id;
        prev_chain = chain;
        prev_resseq = resseq;
      }

      s.names.push_back(atom_name);
      s.residues.push_back(resname);
      s.elements.push_back(element);
      s.molecule_ids.push_back(mol_id);

      auto it = ATOMIC_NUMBERS.find(element);
      s.atomic_numbers.conservativeResize(s.atomic_numbers.size() + 1);
      s.atomic_numbers(s.atomic_numbers.size() - 1) =
          (it != ATOMIC_NUMBERS.end()) ? it->second : 0;

    } catch (const std::exception &e) {
      return std::unexpected(std::string("PDB parse error: ") + e.what());
    }
  }

  const std::size_t N = xyz.size();
  s.coordinates.resize(static_cast<Eigen::Index>(N), 3);
  for (std::size_t i = 0; i < N; ++i) {
    s.coordinates(static_cast<Eigen::Index>(i), 0) = xyz[i][0];
    s.coordinates(static_cast<Eigen::Index>(i), 1) = xyz[i][1];
    s.coordinates(static_cast<Eigen::Index>(i), 2) = xyz[i][2];
  }

  return s;
}

Result<void> write_pdb(const AtomicStructure &s,
                       const std::filesystem::path &path) {
  std::ofstream f(path);
  if (!f)
    return std::unexpected("Cannot write PDB: " + path.string());

  for (Eigen::Index i = 0; i < s.coordinates.rows(); ++i) {
    const std::string &name = (i < static_cast<Eigen::Index>(s.names.size()))
                                  ? s.names[static_cast<std::size_t>(i)]
                                  : "X";
    const std::string &res = (i < static_cast<Eigen::Index>(s.residues.size()))
                                 ? s.residues[static_cast<std::size_t>(i)]
                                 : "UNK";
    const std::string &elem = (i < static_cast<Eigen::Index>(s.elements.size()))
                                  ? s.elements[static_cast<std::size_t>(i)]
                                  : "";
    std::size_t mol = (i < static_cast<Eigen::Index>(s.molecule_ids.size()))
                          ? s.molecule_ids[static_cast<std::size_t>(i)]
                          : 1;

    char buf[81];
    std::snprintf(buf, sizeof(buf),
                  "ATOM  %5lld %-4s %-3s A%4d    %8.3f%8.3f%8.3f  1.00  0.00   "
                  "       %2s\n",
                  static_cast<long long>(i + 1), name.c_str(), res.c_str(),
                  static_cast<int>(mol), s.coordinates(i, 0),
                  s.coordinates(i, 1), s.coordinates(i, 2), elem.c_str());
    f << buf;
  }
  f << "END\n";
  return {};
}

} // namespace fullrmc::io
