#include <RMC/io/AtomicNumbers.hpp>
#include <RMC/io/VaspReader.hpp>

#include <boost/leaf/result.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <format>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
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

bool is_integer_token(const std::string &t) {
  return !t.empty() && std::ranges::all_of(t, [](unsigned char ch) {
           return std::isdigit(ch) != 0;
         });
}

} // namespace

Result<VaspData> read_vasp(const std::filesystem::path &path) {
  std::ifstream file(path);
  if (!file)
    return boost::leaf::new_error(
        std::string{"Cannot open VASP POSCAR file: " + path.string()});

  VaspData out;
  std::string line;
  auto next = [&](std::string &l) -> bool {
    return static_cast<bool>(std::getline(file, l));
  };

  // Line 1: comment.
  if (!next(line))
    return boost::leaf::new_error(
        std::string{"Empty POSCAR: " + path.string()});

  // Line 2: universal scaling factor.
  if (!next(line))
    return boost::leaf::new_error(std::string{"POSCAR missing scaling factor"});
  double scale = 0.0;
  {
    std::istringstream ss{line};
    if (!(ss >> scale))
      return boost::leaf::new_error(
          std::string{"POSCAR scaling factor parse error"});
  }

  // Lines 3-5: lattice vectors a1, a2, a3 — stored as box columns.
  mat3_t lat = mat3_t::Zero();
  for (int i = 0; i < 3; ++i) {
    if (!next(line))
      return boost::leaf::new_error(std::string{"POSCAR missing lattice rows"});
    std::istringstream ss{line};
    double x = 0, y = 0, z = 0;
    if (!(ss >> x >> y >> z))
      return boost::leaf::new_error(std::string{"POSCAR lattice parse error"});
    lat.col(i) = vec3_t{x, y, z};
  }

  // Resolve the scaling factor: a negative value is a target cell volume.
  double s = scale;
  if (scale < 0.0) {
    const double vol =
        std::abs(lat.col(0).dot(lat.col(1).cross(lat.col(2))));
    s = (vol > 0.0) ? std::cbrt(-scale / vol) : 1.0;
  }
  out.box = lat * s;

  // Line 6: element symbols (VASP5) or — for VASP4 — the per-element counts.
  if (!next(line))
    return boost::leaf::new_error(std::string{"POSCAR missing element line"});
  std::vector<std::string> tokens;
  {
    std::istringstream ss{line};
    for (std::string w; ss >> w;)
      tokens.push_back(std::move(w));
  }
  if (tokens.empty())
    return boost::leaf::new_error(std::string{"POSCAR empty element line"});

  const bool vasp5 = !std::ranges::all_of(tokens, is_integer_token);
  if (!vasp5)
    return boost::leaf::new_error(std::string{
        "VASP4 POSCAR has no element symbols; add a VASP5 element line"});

  const std::vector<std::string> symbols = tokens;
  std::vector<int> counts;
  if (!next(line))
    return boost::leaf::new_error(std::string{"POSCAR missing counts line"});
  {
    std::istringstream ss{line};
    for (int c = 0; ss >> c;)
      counts.push_back(c);
  }
  if (counts.size() != symbols.size() || symbols.empty())
    return boost::leaf::new_error(
        std::string{"POSCAR element/count line mismatch"});

  // Optional "Selective dynamics", then the coordinate-mode line.
  if (!next(line))
    return boost::leaf::new_error(std::string{"POSCAR missing coordinate mode"});
  {
    const std::string_view t = trim(line);
    if (!t.empty() && (t.front() == 'S' || t.front() == 's')) {
      if (!next(line))
        return boost::leaf::new_error(
            std::string{"POSCAR missing coordinate mode after Selective"});
    }
  }
  const std::string_view mode_sv = trim(line);
  const char mode = mode_sv.empty() ? 'D' : mode_sv.front();
  const bool cartesian =
      (mode == 'C' || mode == 'c' || mode == 'K' || mode == 'k');

  // Coordinate lines, grouped per element.
  AtomicStructure &st = out.structure;
  std::vector<std::array<double, 3>> cart;
  std::vector<int> anum;
  for (std::size_t e = 0; e < symbols.size(); ++e) {
    const std::string &sym = symbols[e];
    const int z = atomic_number(sym);
    for (int n = 0; n < counts[e]; ++n) {
      if (!next(line))
        return boost::leaf::new_error(
            std::string{"POSCAR has fewer coordinate lines than declared"});
      std::istringstream ss{line};
      double x = 0, y = 0, zc = 0;
      if (!(ss >> x >> y >> zc))
        return boost::leaf::new_error(
            std::string{"POSCAR coordinate line parse error: "} + line);
      const vec3_t c = cartesian ? vec3_t(s * vec3_t{x, y, zc})
                                 : vec3_t(out.box * vec3_t{x, y, zc});
      cart.push_back({c.x(), c.y(), c.z()});
      st.names.push_back(sym);
      st.elements.push_back(sym);
      st.residues.push_back(sym);
      st.molecule_ids.push_back(1);
      anum.push_back(z);
    }
  }

  const std::size_t M = cart.size();
  st.coordinates.resize(static_cast<Eigen::Index>(M), 3);
  for (std::size_t i = 0; i < M; ++i) {
    st.coordinates(static_cast<Eigen::Index>(i), 0) = cart[i][0];
    st.coordinates(static_cast<Eigen::Index>(i), 1) = cart[i][1];
    st.coordinates(static_cast<Eigen::Index>(i), 2) = cart[i][2];
  }
  st.atomic_numbers = Eigen::Map<const ivec_t>(
      anum.data(), static_cast<Eigen::Index>(anum.size()));

  return out;
}

Result<void> write_vasp(const AtomicStructure &s, const mat3_t &box,
                        const std::filesystem::path &path) {
  std::ofstream f(path);
  if (!f)
    return boost::leaf::new_error(
        std::string{"Cannot write VASP POSCAR file: " + path.string()});

  // Group atoms by element in first-appearance order (POSCAR requires grouping).
  std::vector<std::string> order;
  std::vector<std::vector<std::size_t>> groups;
  std::unordered_map<std::string, std::size_t> pos;
  for (std::size_t i = 0; i < s.size(); ++i) {
    const std::string sym =
        (i < s.elements.size() && !s.elements[i].empty()) ? s.elements[i] : "X";
    auto [it, inserted] = pos.try_emplace(sym, order.size());
    if (inserted) {
      order.push_back(sym);
      groups.emplace_back();
    }
    groups[it->second].push_back(i);
  }

  f << "POSCAR written by RMC\n";
  f << "1.0\n";
  // Lattice rows = box columns (the three lattice vectors a1, a2, a3).
  for (int c = 0; c < 3; ++c)
    f << std::format("{:.10f} {:.10f} {:.10f}\n", box(0, c), box(1, c),
                     box(2, c));

  for (std::size_t e = 0; e < order.size(); ++e)
    f << (e ? " " : "") << order[e];
  f << "\n";
  for (std::size_t e = 0; e < order.size(); ++e)
    f << (e ? " " : "") << groups[e].size();
  f << "\n";

  f << "Direct\n";
  const PeriodicBC pbc(box);
  for (std::size_t e = 0; e < order.size(); ++e)
    for (const std::size_t idx : groups[e]) {
      const vec3_t cart{s.coordinates(static_cast<Eigen::Index>(idx), 0),
                        s.coordinates(static_cast<Eigen::Index>(idx), 1),
                        s.coordinates(static_cast<Eigen::Index>(idx), 2)};
      const vec3_t frac = pbc.inv_box() * cart;
      f << std::format("{:.10f} {:.10f} {:.10f}\n", frac.x(), frac.y(),
                       frac.z());
    }

  return {};
}

} // namespace RMC::io
