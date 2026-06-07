#include "AtatFormats.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <set>
#include <sstream>
#include <stdexcept>

namespace RMC::atat {
namespace {

// Split a string on whitespace.
std::vector<std::string> split_ws(const std::string &s) {
  std::vector<std::string> out;
  std::istringstream ss(s);
  std::string tok;
  while (ss >> tok) {
    out.push_back(tok);
  }
  return out;
}

// Read the next line that contains non-whitespace; false at EOF.
bool next_nonempty_line(std::istream &in, std::string &line) {
  while (std::getline(in, line)) {
    if (line.find_first_not_of(" \t\r\n") != std::string::npos) {
      return true;
    }
  }
  return false;
}

vec3_t parse_vec3(const std::vector<std::string> &toks, std::size_t off = 0) {
  if (toks.size() < off + 3) {
    throw std::runtime_error("ATAT parse: expected 3 numbers");
  }
  return vec3_t(std::stod(toks[off]), std::stod(toks[off + 1]),
                std::stod(toks[off + 2]));
}

// Build lattice vectors (columns) from a,b,c,alpha,beta,gamma (degrees).
// Orientation is conventional; the absolute orientation is immaterial to SQS
// correlations (which live in the lattice/fractional frame).
mat3_t lattice_vectors(double a, double b, double c, double alpha, double beta,
                       double gamma) {
  const double d2r = M_PI / 180.0;
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
  auto toks = split_ws(line);
  if (toks.size() == 6) {
    lat.axes = lattice_vectors(std::stod(toks[0]), std::stod(toks[1]),
                               std::stod(toks[2]), std::stod(toks[3]),
                               std::stod(toks[4]), std::stod(toks[5]));
  } else if (toks.size() == 3) {
    lat.axes.col(0) = parse_vec3(toks);
    for (int i = 1; i < 3; ++i) {
      if (!next_nonempty_line(in, line)) {
        throw std::runtime_error("rndstr: truncated axes");
      }
      lat.axes.col(i) = parse_vec3(split_ws(line));
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
    lat.cell.col(i) = parse_vec3(split_ws(line));
  }

  // --- sites ---
  std::set<std::string> labelset;
  while (next_nonempty_line(in, line)) {
    std::istringstream ss(line);
    double x, y, z;
    if (!(ss >> x >> y >> z)) {
      throw std::runtime_error("rndstr: malformed site line: " + line);
    }
    LatticeSite site;
    site.frac = vec3_t(x, y, z);
    std::string rest;
    std::getline(ss, rest);
    // Species are separated by any of " \t,;/"; each token is Sp or Sp=occ.
    std::string tok;
    auto flush_tok = [&] {
      if (tok.empty()) {
        return;
      }
      const auto eq = tok.find('=');
      std::string sp =
          (eq == std::string::npos) ? tok : tok.substr(0, eq);
      double occ = (eq == std::string::npos)
                       ? -1.0
                       : std::stod(tok.substr(eq + 1));
      site.occ.emplace_back(std::move(sp), occ);
      tok.clear();
    };
    for (const char ch : rest) {
      if (ch == ' ' || ch == '\t' || ch == ',' || ch == ';' || ch == '/' ||
          ch == '\r') {
        flush_tok();
      } else {
        tok.push_back(ch);
      }
    }
    flush_tok();
    if (site.occ.empty()) {
      throw std::runtime_error("rndstr: site with no species: " + line);
    }
    // Default to equiatomic occupation when not specified.
    const bool has_occ =
        std::any_of(site.occ.begin(), site.occ.end(),
                    [](const auto &p) { return p.second >= 0.0; });
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
  int n = 0;
  if (!(in >> n) || n < 0) {
    throw std::runtime_error("sym.out: missing operation count");
  }
  std::vector<SymOp> ops;
  ops.reserve(static_cast<std::size_t>(n));
  for (int k = 0; k < n; ++k) {
    SymOp op;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        if (!(in >> op.rot(i, j))) {
          throw std::runtime_error("sym.out: truncated point operation");
        }
      }
    }
    for (int i = 0; i < 3; ++i) {
      if (!(in >> op.trans(i))) {
        throw std::runtime_error("sym.out: truncated translation");
      }
    }
    ops.push_back(op);
  }
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
  std::vector<RawOrbit> orbits;
  double mult = 0.0;
  while (in >> mult) {
    RawOrbit o;
    o.multiplicity = mult;
    int npts = 0;
    if (!(in >> o.length) || !(in >> npts) || npts < 0) {
      throw std::runtime_error("clusters.out: malformed orbit header");
    }
    for (int p = 0; p < npts; ++p) {
      ClusterPoint cp;
      double x, y, z;
      if (!(in >> x >> y >> z >> cp.site_type >> cp.func)) {
        throw std::runtime_error("clusters.out: malformed point");
      }
      cp.coord = vec3_t(x, y, z);
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
  std::vector<double> out;
  std::istringstream ss(line);
  double v;
  while (ss >> v) {
    out.push_back(v);
  }
  if (out.empty()) {
    throw std::runtime_error("corrdump: no correlations parsed");
  }
  return out;
}

void write_str_out(const std::filesystem::path &path, const mat3_t &axes,
                   const mat3_t &supercell, const std::vector<vec3_t> &positions,
                   const std::vector<std::string> &species) {
  if (positions.size() != species.size()) {
    throw std::runtime_error("write_str_out: positions/species size mismatch");
  }
  std::ofstream f(path);
  if (!f) {
    throw std::runtime_error("Cannot write str.out: " + path.string());
  }
  f << std::setprecision(9);
  for (int i = 0; i < 3; ++i) {
    f << axes(0, i) << " " << axes(1, i) << " " << axes(2, i) << "\n";
  }
  for (int i = 0; i < 3; ++i) {
    f << supercell(0, i) << " " << supercell(1, i) << " " << supercell(2, i)
      << "\n";
  }
  for (std::size_t a = 0; a < positions.size(); ++a) {
    f << positions[a](0) << " " << positions[a](1) << " " << positions[a](2)
      << " " << species[a] << "\n";
  }
}

} // namespace RMC::atat
