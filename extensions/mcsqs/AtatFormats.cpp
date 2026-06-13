#include "AtatFormats.hpp"

#include <algorithm>
#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/parser/parser.hpp>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <optional>
#include <ranges>
#include <set>
#include <numbers>
#include <stdexcept>
#include <string>

namespace RMC::atat {
namespace {

namespace bp = boost::parser;

// Parses three space-separated reals into the components of a 3-vector.
constexpr auto vec3_p = bp::double_ >> bp::double_ >> bp::double_;

// Read the next line that contains non-whitespace; false at EOF.
bool next_nonempty_line(std::istream &in, std::string &line) {
  while (std::getline(in, line)) {
    if (!boost::algorithm::all(line, boost::is_space())) {
      return true;
    }
  }
  return false;
}

// Slurp an entire stream into a string (the sym.out / clusters.out token
// streams are small and parse most cleanly as one whitespace-skipped range).
std::string slurp(std::istream &in) {
  return std::string((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
}

// Parse exactly three reals from one line into an Eigen column.
vec3_t parse_vec3_line(const std::string &line) {
  const auto v = bp::parse(line, vec3_p, bp::ws);
  if (!v) {
    throw std::runtime_error("ATAT parse: expected 3 numbers: " + line);
  }
  const auto &[x, y, z] = *v;
  return vec3_t(x, y, z);
}

// Build lattice vectors (columns) from a,b,c,alpha,beta,gamma (degrees).
// Orientation is conventional; the absolute orientation is immaterial to SQS
// correlations (which live in the lattice/fractional frame).
mat3_t lattice_vectors(double a, double b, double c, double alpha, double beta,
                       double gamma) {
  const double d2r = std::numbers::pi / 180.0;
  const double ca = std::cos(alpha * d2r), cb = std::cos(beta * d2r),
               cg = std::cos(gamma * d2r), sg = std::sin(gamma * d2r);
  mat3_t m;
  m.col(0) = vec3_t(a, 0.0, 0.0);
  m.col(1) = vec3_t(b * cg, b * sg, 0.0);
  const double cx = c * cb;
  const double cy = c * (ca - cb * cg) / sg;
  const double cz2 = c * c - cx * cx - cy * cy;
  const double cz = std::sqrt(std::max(0.0, cz2));
  m.col(2) = vec3_t(cx, cy, cz);
  return m;
}

} // namespace

int AtatLattice::occupation_index(const std::string &species) const {
  const auto it = std::lower_bound(labels.begin(), labels.end(), species);
  if (it == labels.end() || *it != species) {
    return -1;
  }
  return static_cast<int>(it - labels.begin());
}

AtatLattice parse_lattice(std::istream &in) {
  AtatLattice lat;
  std::string line;

  // --- coordinate system (axes) ---
  if (!next_nonempty_line(in, line)) {
    throw std::runtime_error("rndstr: empty file");
  }
  // First line is either 'a b c al be ga' (6 numbers) or the first of three
  // axis rows (3 numbers each).
  if (const auto six = bp::parse(line, bp::repeat(6)[bp::double_], bp::ws)) {
    const auto &p = *six;
    lat.axes = lattice_vectors(p[0], p[1], p[2], p[3], p[4], p[5]);
  } else if (bp::parse(line, vec3_p, bp::ws)) {
    lat.axes.col(0) = parse_vec3_line(line);
    for (int i = 1; i < 3; ++i) {
      if (!next_nonempty_line(in, line)) {
        throw std::runtime_error("rndstr: truncated axes");
      }
      lat.axes.col(i) = parse_vec3_line(line);
    }
  } else {
    throw std::runtime_error("rndstr: first line must be 'a b c al be ga' or a "
                             "3-vector axis row");
  }

  // --- primitive cell (3 vectors, in axes coords) ---
  for (int i = 0; i < 3; ++i) {
    if (!next_nonempty_line(in, line)) {
      throw std::runtime_error("rndstr: truncated cell");
    }
    lat.cell.col(i) = parse_vec3_line(line);
  }

  // --- sites ---
  // A site line is "fx fy fz" followed by a species list. Species are
  // separated by any of " \t,;/" (the skipper below) and each token is either
  // "Sp" or "Sp=occ"; lexeme[] keeps each token contiguous (no skipping
  // inside).
  const auto sep = bp::char_(" \t\r\n,;/");
  const auto name = +(bp::char_ - bp::char_(" \t\r\n,;/="));
  const auto species_p = bp::lexeme[name >> -('=' >> bp::double_)];
  const auto site_p = bp::double_ >> bp::double_ >> bp::double_ >> +species_p;

  std::set<std::string> labelset;
  while (next_nonempty_line(in, line)) {
    const auto parsed = bp::parse(line, site_p, sep);
    if (!parsed) {
      throw std::runtime_error("rndstr: malformed site line: " + line);
    }
    const auto &[x, y, z, occ_list] = *parsed;
    LatticeSite site;
    site.frac = vec3_t(x, y, z);
    for (const auto &[sp, occ] : occ_list) {
      site.occ.emplace_back(sp, occ ? *occ : -1.0);
    }
    if (site.occ.empty()) {
      throw std::runtime_error("rndstr: site with no species: " + line);
    }
    // Default to equiatomic occupation when not specified.
    const bool has_occ = std::ranges::any_of(
        site.occ, [](const auto &p) { return p.second >= 0.0; });
    if (!has_occ) {
      const double u = 1.0 / static_cast<double>(site.occ.size());
      for (auto &p : site.occ) {
        p.second = u;
      }
    }
    for (const auto &[sp, _] : site.occ) {
      labelset.insert(sp);
    }
    lat.sites.push_back(std::move(site));
  }
  lat.labels.assign(labelset.begin(), labelset.end()); // std::set is sorted
  return lat;
}

AtatLattice parse_lattice(const std::filesystem::path &path) {
  std::ifstream f(path);
  if (!f) {
    throw std::runtime_error("Cannot open lattice file: " + path.string());
  }
  return parse_lattice(f);
}

std::vector<SymOp> parse_sym(std::istream &in) {
  const std::string text = slurp(in);
  auto it = text.begin();
  const auto end = text.end();

  const auto n =
      bp::prefix_parse(it, end, bp::uint_, bp::ws)
          .or_else([]() -> std::optional<unsigned> {
            throw std::runtime_error("sym.out: missing operation count");
          })
          .value();

  std::vector<SymOp> ops;
  ops.reserve(n);
  std::ranges::transform(
      std::views::iota(0u, n), std::back_inserter(ops), [&](unsigned) {
        SymOp op;

        const auto rot =
            bp::prefix_parse(it, end, bp::repeat(9)[bp::double_], bp::ws)
                .or_else([]() -> std::optional<std::vector<double>> {
                  throw std::runtime_error(
                      "sym.out: truncated point operation");
                });

        std::ranges::for_each(std::views::iota(0, 9), [&](int k) {
          op.rot(k / 3, k % 3) = (*rot)[static_cast<std::size_t>(k)];
        });

        const auto tr =
            bp::prefix_parse(it, end, bp::repeat(3)[bp::double_], bp::ws)
                .or_else([]() -> std::optional<std::vector<double>> {
                  throw std::runtime_error("sym.out: truncated translation");
                });

        op.trans = Eigen::Map<const Eigen::Vector3d>(tr->data());
        return op;
      });
  return ops;
}

std::vector<SymOp> parse_sym(const std::filesystem::path &path) {
  std::ifstream f(path);
  if (!f) {
    throw std::runtime_error("Cannot open sym file: " + path.string());
  }
  return parse_sym(f);
}

std::vector<RawOrbit> parse_clusters(std::istream &in) {
  // Block layout (whitespace/newline separated; blank lines are incidental):
  //   multiplicity  length  n_points  then n_points × (x y z site_type func)
  const std::string text = slurp(in);
  auto it = text.begin();
  const auto end = text.end();

  // Advance past whitespace; report whether the stream is exhausted.
  const auto at_end = [&] {
    it = std::ranges::find_if(
        it, end, [](unsigned char c) { return std::isspace(c) == 0; });
    return it == end;
  };

  // header: multiplicity length n_points ; point: x y z site_type func
  const auto header_p = bp::double_ >> bp::double_ >> bp::uint_;
  const auto point_p =
      bp::double_ >> bp::double_ >> bp::double_ >> bp::int_ >> bp::int_;

  std::vector<RawOrbit> orbits;
  while (!at_end()) {
    const auto hdr = bp::prefix_parse(it, end, header_p, bp::ws);
    if (!hdr) {
      throw std::runtime_error("clusters.out: malformed orbit header");
    }
    const auto &[mult, length, npts] = *hdr;

    RawOrbit o{.multiplicity = mult, .length = length, .points = {}};
    const auto pts =
        bp::prefix_parse(it, end, bp::repeat(npts)[point_p], bp::ws);
    if (!pts) {
      throw std::runtime_error("clusters.out: malformed point");
    }

    for (const auto &[x, y, z, site_type, func] : *pts) {
      ClusterPoint cp{
          .coord = vec3_t(x, y, z), .site_type = site_type, .func = func};
      o.points.push_back(cp);
    }
    orbits.push_back(std::move(o));
  }
  return orbits;
}

std::vector<RawOrbit> parse_clusters(const std::filesystem::path &path) {
  std::ifstream f(path);
  if (!f) {
    throw std::runtime_error("Cannot open clusters file: " + path.string());
  }
  return parse_clusters(f);
}

std::vector<double> parse_correlations(std::istream &in) {
  std::string line;
  if (!next_nonempty_line(in, line)) {
    throw std::runtime_error("corrdump: empty correlation output");
  }
  const auto out = bp::parse(line, +bp::double_, bp::ws);
  if (!out || out->empty()) {
    throw std::runtime_error("corrdump: no correlations parsed");
  }
  return *out;
}

void write_str_out(const std::filesystem::path &path, const mat3_t &axes,
                   const mat3_t &supercell,
                   const std::vector<vec3_t> &positions,
                   const std::vector<std::string> &species) {
  if (positions.size() != species.size()) {
    throw std::runtime_error("write_str_out: positions/species size mismatch");
  }
  std::ofstream f(path);
  if (!f) {
    throw std::runtime_error("Cannot write str.out: " + path.string());
  }
  f << std::setprecision(9);
  for (Eigen::Index i = 0; i < 3; ++i) {
    f << axes.col(i).transpose() << '\n';
  }
  for (Eigen::Index i = 0; i < 3; ++i) {
    f << supercell.col(i).transpose() << '\n';
  }
  for (auto &&[pos, sp] : std::views::zip(positions, species)) {
    f << pos(0) << ' ' << pos(1) << ' ' << pos(2) << ' ' << sp << '\n';
  }
}

} // namespace RMC::atat
