#include <RMC/io/AtomicNumbers.hpp>
#include <RMC/io/VaspReader.hpp>

#include <boost/algorithm/string/classification.hpp> // boost::is_any_of
#include <boost/leaf/error.hpp>                      // BOOST_LEAF_AUTO
#include <boost/leaf/result.hpp>
#include <boost/parser/parser.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <format>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace RMC::io {

namespace {

namespace bp = boost::parser;

// Three space-separated reals (a lattice row or a fractional/Cartesian coord).
constexpr auto vec3_p = bp::double_ >> bp::double_ >> bp::double_;

// Shape a parsed (x, y, z) tuple into an Eigen column vector.
const auto to_vec3 = [](const auto &t) {
  const auto &[x, y, z] = t;
  return vec3_t{x, y, z};
};

template <class T>
Result<T> or_error(std::optional<T> o, std::string_view msg) {
  return o ? std::move(*o)
           : Result<T>{boost::leaf::new_error(std::string{msg})};
}

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

FORCE_INLINE bool is_integer_token(const std::string &t) {
  return !t.empty() && std::ranges::all_of(t, [](unsigned char ch) {
    return std::isdigit(ch) != 0;
  });
}

} // namespace

Result<VaspData> read_vasp(const std::filesystem::path &path) {
  std::ifstream file(path);
  if (!file) {
    return boost::leaf::new_error(
        std::string{"Cannot open VASP POSCAR file: " + path.string()});
  }

  VaspData out;
  std::string line;
  auto next = [&](std::string &l) -> bool {
    return static_cast<bool>(std::getline(file, l));
  };

  // Line 1: comment.
  if (!next(line)) {
    return boost::leaf::new_error(
        std::string{"Empty POSCAR: " + path.string()});
  }

  // Line 2: universal scaling factor.
  if (!next(line)) {
    return boost::leaf::new_error(std::string{"POSCAR missing scaling factor"});
  }
  auto scale_it = line.begin();
  BOOST_LEAF_AUTO(scale, or_error(bp::prefix_parse(scale_it, line.end(),
                                                   bp::double_, bp::ws),
                                  "POSCAR scaling factor parse error"));

  // Lines 3-5: lattice vectors a1, a2, a3 — stored as box columns.
  mat3_t lat = mat3_t::Zero();
  for (int i = 0; i < 3; ++i) {
    if (!next(line)) {
      return boost::leaf::new_error(std::string{"POSCAR missing lattice rows"});
    }
    auto it = line.begin();
    BOOST_LEAF_AUTO(
        row,
        or_error(
            bp::prefix_parse(it, line.end(), vec3_p, bp::ws).transform(to_vec3),
            "POSCAR lattice parse error"));
    lat.col(i) = row;
  }

  // Resolve the scaling factor: a negative value is a target cell volume.
  double s = scale;
  if (scale < 0.0) {
    const double vol = std::abs(lat.col(0).dot(lat.col(1).cross(lat.col(2))));
    s = (vol > 0.0) ? std::cbrt(-scale / vol) : 1.0;
  }
  out.box = lat * s;

  // Line 6: element symbols (VASP5) or — for VASP4 — the per-element counts.
  if (!next(line)) {
    return boost::leaf::new_error(std::string{"POSCAR missing element line"});
  }
  // Element symbols are whitespace-separated words; +word_p needs at least one,
  // so a blank line fails the parse and or_error turns it into the empty-line
  // error — no separate emptiness check, and no istringstream copy.
  const auto word_p = bp::lexeme[+(bp::char_ - bp::char_(" \t\r\n"))];
  BOOST_LEAF_AUTO(symbols, or_error(bp::parse(line, +word_p, bp::ws),
                                    "POSCAR empty element line"));

  const bool vasp5 = !std::ranges::all_of(symbols, is_integer_token);
  if (!vasp5) {
    return boost::leaf::new_error(std::string{
        "VASP4 POSCAR has no element symbols; add a VASP5 element line"});
  }

  if (!next(line)) {
    return boost::leaf::new_error(std::string{"POSCAR missing counts line"});
  }

  // Parse the per-element counts, then require exactly one per symbol in the
  // same chain: .and_then drops to nullopt (→ the mismatch error) if the
  // lengths disagree or the line held no integers.
  auto counts_it = line.begin();
  BOOST_LEAF_AUTO(
      counts,
      or_error(bp::prefix_parse(counts_it, line.end(), +bp::int_, bp::ws)
                   .and_then([&symbols](std::vector<int> c)
                                 -> std::optional<std::vector<int>> {
                     if (c.size() != symbols.size())
                       return std::nullopt;
                     return c;
                   }),
               "POSCAR element/count line mismatch"));

  // Optional "Selective dynamics", then the coordinate-mode line.
  if (!next(line)) {
    return boost::leaf::new_error(
        std::string{"POSCAR missing coordinate mode"});
  }

  {
    const std::string_view t = trim(line);
    if (!t.empty() && (t.front() == 'S' || t.front() == 's')) {
      if (!next(line)) {
        return boost::leaf::new_error(
            std::string{"POSCAR missing coordinate mode after Selective"});
      }
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
      if (!next(line)) {
        return boost::leaf::new_error(
            std::string{"POSCAR has fewer coordinate lines than declared"});
      }

      auto it = line.begin();

      const auto frac =
          bp::prefix_parse(it, line.end(), vec3_p, bp::ws).transform(to_vec3);
      if (!frac) {
        return boost::leaf::new_error(
            std::string{"POSCAR coordinate line parse error: "} + line);
      }

      const vec3_t c = cartesian ? vec3_t(s * *frac) : vec3_t(out.box * *frac);
      cart.push_back({c.x(), c.y(), c.z()});
      st.names.push_back(sym);
      st.elements.push_back(sym);
      st.residues.push_back(sym);
      st.molecule_ids.push_back(1);
      anum.push_back(z);
    }
  }

  const std::size_t M = cart.size();
  // cart is a contiguous std::vector<std::array<double,3>>; map it directly
  // into the row-major coordinate matrix instead of copying component by
  // component.
  st.coordinates =
      Eigen::Map<const coords_t>(reinterpret_cast<const double *>(cart.data()),
                                 static_cast<Eigen::Index>(M), 3);
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

  // Group atoms by element in first-appearance order (POSCAR requires
  // grouping).
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
      const vec3_t cart =
          s.coordinates.row(static_cast<Eigen::Index>(idx)).transpose();
      const vec3_t frac = pbc.inv_box() * cart;
      f << std::format("{:.10f} {:.10f} {:.10f}\n", frac.x(), frac.y(),
                       frac.z());
    }

  return {};
}

} // namespace RMC::io
