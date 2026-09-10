#include "TextParse.hpp"

#include <seitz/data/element_data.hpp>
#include <RMC/io/VaspReader.hpp>

#include <boost/leaf/error.hpp> // BOOST_LEAF_AUTO
#include <boost/leaf/result.hpp>
#include <boost/parser/parser.hpp>

#include <cmath>
#include <format>
#include <fstream>
#include <optional>
#include <ranges>
#include <string>
#include <string_view>
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

} // namespace

Result<VaspData> read_vasp(const std::filesystem::path &path) {
  std::ifstream file(path);
  if (!file) {
    return boost::leaf::new_error(
        std::string{"Cannot open VASP POSCAR file: " + path.string()});
  }

  VaspData out;
  std::string line;
  // Read the next line into `line`, or fail naming what is missing.
  const auto next = [&](std::string_view what) -> Result<void> {
    if (std::getline(file, line)) {
      return {};
    }
    return boost::leaf::new_error(std::format("POSCAR missing {}", what));
  };
  // Parse the start of `line` with p (trailing text, e.g. flags, ignored).
  const auto parse_line = [&](const auto &p, std::string_view error) {
    auto it = line.begin();
    return or_error(bp::prefix_parse(it, line.end(), p, bp::ws), error);
  };

  BOOST_LEAF_CHECK(next("comment line"));
  BOOST_LEAF_CHECK(next("scaling factor"));
  BOOST_LEAF_AUTO(scale,
                  parse_line(bp::double_, "POSCAR scaling factor parse error"));

  // Lines 3-5: lattice vectors a1, a2, a3 — stored as box columns.
  mat3_t lat;
  for (int i = 0; i < 3; ++i) {
    BOOST_LEAF_CHECK(next("lattice rows"));
    BOOST_LEAF_AUTO(row, parse_line(vec3_p, "POSCAR lattice parse error"));
    lat.col(i) = to_vec3(row);
  }
  // A negative scaling factor is the target cell volume.
  const double vol = std::abs(lat.determinant());
  const double s = scale >= 0.0 ? scale
                   : vol > 0.0  ? std::cbrt(-scale / vol)
                                : 1.0;
  out.box = lat * s;

  // Line 6: element symbols (VASP5); VASP4 has the counts here instead.
  BOOST_LEAF_CHECK(next("element line"));
  const auto word_p = bp::lexeme[+(bp::char_ - bp::char_(" \t\r\n"))];
  BOOST_LEAF_AUTO(symbols, or_error(bp::parse(line, +word_p, bp::ws),
                                    "POSCAR empty element line"));
  if (bp::parse(line, +bp::int_, bp::ws)) {
    return boost::leaf::new_error(std::string{
        "VASP4 POSCAR has no element symbols; add a VASP5 element line"});
  }

  BOOST_LEAF_CHECK(next("counts line"));
  BOOST_LEAF_AUTO(counts,
                  parse_line(+bp::int_, "POSCAR element/count line mismatch"));
  if (counts.size() != symbols.size()) {
    return boost::leaf::new_error(
        std::string{"POSCAR element/count line mismatch"});
  }

  // Optional "Selective dynamics", then the coordinate-mode line.
  BOOST_LEAF_CHECK(next("coordinate mode"));
  if (const auto t = detail::trim(line);
      t.starts_with('S') || t.starts_with('s')) {
    BOOST_LEAF_CHECK(next("coordinate mode after Selective"));
  }
  const std::string_view mode = detail::trim(line);
  const bool cartesian =
      !mode.empty() && std::string_view{"CcKk"}.contains(mode.front());

  // Coordinate lines, grouped per element.
  detail::StructureBuilder atoms;
  for (const auto &[sym, cnt] : std::views::zip(symbols, counts)) {
    const int z = seitz::data::atomic_number(sym).value_or(0);
    for (int n = 0; n < cnt; ++n) {
      BOOST_LEAF_CHECK(next("coordinate lines (fewer than declared)"));
      BOOST_LEAF_AUTO(xyz,
                      parse_line(vec3_p, "POSCAR coordinate line parse error"));
      const vec3_t c =
          cartesian ? vec3_t(s * to_vec3(xyz)) : vec3_t(out.box * to_vec3(xyz));
      atoms.add({c.x(), c.y(), c.z()}, z, sym);
    }
  }
  out.structure = std::move(atoms).finish();
  return out;
}

Result<void> write_vasp(const AtomicStructure &s, const mat3_t &box,
                        const std::filesystem::path &path) {
  std::ofstream f(path);
  if (!f) {
    return boost::leaf::new_error(
        std::string{"Cannot write VASP POSCAR file: " + path.string()});
  }
  // POSCAR requires atoms grouped by element (first-appearance order).
  const auto groups = detail::group_by_element(s);

  f << "POSCAR written by RMC\n1.0\n";
  // Lattice rows = box columns (the three lattice vectors a1, a2, a3).
  for (int c = 0; c < 3; ++c) {
    f << std::format("{:.10f} {:.10f} {:.10f}\n", box(0, c), box(1, c),
                     box(2, c));
  }
  for (const auto &[e, g] : groups | std::views::enumerate) {
    f << (e ? " " : "") << g.symbol;
  }
  f << "\n";
  for (const auto &[e, g] : groups | std::views::enumerate) {
    f << (e ? " " : "") << g.atoms.size();
  }
  f << "\nDirect\n";

  const mat3_t inv_box = box.inverse();
  for (const auto &g : groups) {
    for (const std::size_t i : g.atoms) {
      const vec3_t frac =
          inv_box * s.coordinates.row(static_cast<Eigen::Index>(i)).transpose();
      f << std::format("{:.10f} {:.10f} {:.10f}\n", frac.x(), frac.y(),
                       frac.z());
    }
  }
  return {};
}

} // namespace RMC::io
