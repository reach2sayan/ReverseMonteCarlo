#include "ClusterEnumerator.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <optional>
#include <random>
#include <ranges>
#include <stdexcept>
#include <string>
#include <vector>

namespace RMC::atat {
namespace {

constexpr double kTol = 1e-3; // ATAT zero_tolerance for site/symmetry matching

// Distance of x to the nearest integer (ATAT cylinder_norm, xtalutil.h:41).
double cyl(double x) {
  return std::fabs(std::fmod(std::fabs(x) + 0.5, 1.0) - 0.5);
}

bool in01(const vec3_t &v) {
  for (int i = 0; i < 3; ++i) {
    if (v(i) < 0.0 || v(i) >= 1.0) {
      return false;
    }
  }
  return true;
}

bool frac_zero(const vec3_t &d) {
  return cyl(d(0)) < kTol && cyl(d(1)) < kTol && cyl(d(2)) < kTol;
}

// Two positions equal modulo the (primitive or super) cell.
bool same_site(const vec3_t &a, const vec3_t &b, const mat3_t &inv_cell) {
  return frac_zero(inv_cell * (a - b));
}

// Index of the supercell site coincident with `pos` (mod supercell), or -1.
int which_atom(const vec3_t &pos, const std::vector<vec3_t> &sites,
               const mat3_t &inv_super) {
  for (std::size_t i = 0; i < sites.size(); ++i) {
    if (same_site(pos, sites[i], inv_super)) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

// Local mutable cluster (ATAT MultiCluster) in axes coords.
struct MC {
  std::vector<vec3_t> clus;
  std::vector<int> func;
  std::vector<int> site_type;
};

MC apply_sym(const SymOp &op, const MC &a) {
  MC b;
  b.func = a.func;
  b.site_type = a.site_type;
  b.clus.resize(a.clus.size());
  for (std::size_t i = 0; i < a.clus.size(); ++i) {
    b.clus[i] = op.rot * a.clus[i] + op.trans;
  }
  return b;
}

// Index in `clus` matching `pos` mod primitive cell, else -1.
int which_in_cluster(const std::vector<vec3_t> &clus, const vec3_t &pos,
                     const mat3_t &inv_cell) {
  for (std::size_t i = 0; i < clus.size(); ++i) {
    if (same_site(clus[i], pos, inv_cell)) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

// ATAT equivalent_mod_cell for MultiCluster (xtalutil.c++:181-203).
bool cluster_equiv(const MC &a, const MC &b, const mat3_t &inv_cell) {
  if (a.clus.size() != b.clus.size()) {
    return false;
  }
  if (a.clus.empty()) {
    return true;
  }
  for (std::size_t i = 0; i < b.clus.size(); ++i) {
    if (!same_site(a.clus[0], b.clus[i], inv_cell)) {
      continue;
    }
    const vec3_t shift = b.clus[i] - a.clus[0];
    bool ok = true;
    for (std::size_t j = 0; j < a.clus.size(); ++j) {
      const int at = which_in_cluster(b.clus, a.clus[j] + shift, inv_cell);
      if (at < 0 || b.func[static_cast<std::size_t>(at)] != a.func[j]) {
        ok = false;
        break;
      }
    }
    if (ok) {
      return true;
    }
  }
  return false;
}

// ATAT find_equivalent_clusters (calccorr.c++:186-208): symmetry images of the
// representative, deduped modulo the primitive cell. apply_symmetry preserves
// point order and func, so each image's point p carries the representative's
// func[p] — letting the orbit keep ONE per-point func/site_type array.
std::vector<MC> find_equivalent(const MC &rep, const std::vector<SymOp> &sym,
                                const mat3_t &inv_cell) {
  std::vector<MC> list;
  for (const auto &op : sym) {
    MC img = apply_sym(op, rep);
    const bool dup = std::any_of(list.begin(), list.end(), [&](const MC &e) {
      return cluster_equiv(e, img, inv_cell);
    });
    if (!dup) {
      list.push_back(std::move(img));
    }
  }
  return list;
}

vec3_t floor_vec(const vec3_t &v) {
  return vec3_t(std::floor(v(0)), std::floor(v(1)), std::floor(v(2)));
}

// ATAT LatticePointInCellIterator (xtalutil.c++:262-286): primitive lattice
// points inside the supercell (count = |det(supercell)/det(cell)|). Returns the
// points t = cell * (integer combo), in axes coords.
std::vector<vec3_t> enumerate_lattice_points(const mat3_t &cell,
                                             const mat3_t &supercell) {
  const mat3_t cell_to_super = supercell.inverse() * cell;
  const mat3_t super_to_cell = cell_to_super.inverse();
  const vec3_t shift = vec3_t::Constant(M_PI * kTol * 0.1);
  vec3_t mn = vec3_t::Constant(1e30);
  vec3_t mx = vec3_t::Constant(-1e30);
  for (int cx = 0; cx < 2; ++cx) {
    for (int cy = 0; cy < 2; ++cy) {
      for (int cz = 0; cz < 2; ++cz) {
        const vec3_t corner(cx, cy, cz);
        const vec3_t v = super_to_cell * (corner + shift);
        mn = mn.cwiseMin(v);
        mx = mx.cwiseMax(v);
      }
    }
  }
  const vec3_t lo = floor_vec(mn);
  const vec3_t hi = floor_vec(mx) + vec3_t::Constant(1.0); // ceil-ish bound
  std::vector<vec3_t> pts;
  for (int i = static_cast<int>(lo(0)); i <= static_cast<int>(hi(0)); ++i) {
    for (int j = static_cast<int>(lo(1)); j <= static_cast<int>(hi(1)); ++j) {
      for (int k = static_cast<int>(lo(2)); k <= static_cast<int>(hi(2)); ++k) {
        const vec3_t cur(i, j, k);
        if (in01(cell_to_super * cur + shift)) {
          pts.push_back(cell * cur);
        }
      }
    }
  }
  return pts;
}

// ATAT wrap_inside_cell (xtalutil.c++:41-52): canonicalise a position into the
// supercell with a tiny irrational shift to avoid boundary ties.
vec3_t wrap_inside(const vec3_t &pos, const mat3_t &super,
                   const mat3_t &inv_super) {
  const vec3_t shift = vec3_t::Constant(M_PI * kTol * 0.1);
  vec3_t f = inv_super * pos - shift;
  for (int i = 0; i < 3; ++i) {
    f(i) -= std::floor(f(i)); // mod1
  }
  return super * (f + shift);
}

// Random-state target = product of point correlations (mcsqs.c++:456-481).
double compute_target(const MC &rep, const AtatLattice &lat,
                      const CorrFuncTable &table,
                      const std::unordered_map<std::string, int> &occ_index,
                      const mat3_t &inv_cell) {
  double prod = 1.0;
  for (const auto [pt, st, fn] :
       std::views::zip(rep.clus, rep.site_type, rep.func)) {
    const auto site = std::ranges::find_if(lat.sites, [&](const auto &s) {
      return same_site(s.frac, pt, inv_cell);
    });
    if (site == lat.sites.end()) {
      throw std::runtime_error("cluster point does not coincide with a site");
    }
    double pc = 0.0;
    for (const auto &[sp, occ] : site->occ) {
      pc += occ * table.value(st, fn, occ_index.at(sp));
    }
    prod *= pc;
  }
  return prod;
}

} // namespace

EnumeratedSqs enumerate(const AtatLattice &lat, const std::vector<SymOp> &sym,
                        const std::vector<RawOrbit> &raw,
                        const Eigen::Matrix3i &sc_matrix, std::uint32_t seed) {
  // Single-sublattice guard: every active (multi-species) site must share one
  // species set, so the global alphabetical occupation index == within-site
  // one.
  std::optional<std::vector<std::string>> active_set;
  for (const auto &s : lat.sites) {
    if (s.occ.size() <= 1) {
      continue;
    }
    std::vector<std::string> set;
    for (const auto &p : s.occ) {
      set.push_back(p.first);
    }
    std::sort(set.begin(), set.end());
    if (!active_set) {
      active_set = set;
    } else if (*active_set != set) {
      throw std::runtime_error(
          "enumerate: heterogeneous sublattices (different species sets) are "
          "not yet supported; use one shared species set");
    }
  }

  const mat3_t cell = lat.cell;
  const mat3_t supercell = cell * sc_matrix.cast<double>();
  const mat3_t inv_super = supercell.inverse();
  const mat3_t inv_cell = cell.inverse();

  std::unordered_map<std::string, int> occ_index;
  for (std::size_t i = 0; i < lat.labels.size(); ++i) {
    occ_index[lat.labels[i]] = static_cast<int>(i);
  }
  int maxc = 2;
  for (const auto &s : lat.sites) {
    maxc = std::max(maxc, static_cast<int>(s.occ.size()));
  }
  CorrFuncTable table = CorrFuncTable::trigonometric(maxc);

  const std::vector<vec3_t> pts = enumerate_lattice_points(cell, supercell);
  const long det =
      std::lround(std::abs(sc_matrix.cast<double>().determinant()));
  if (static_cast<long>(pts.size()) != det) {
    throw std::runtime_error(
        "enumerate: lattice-point count != det(supercell)");
  }

  // Supercell sites: (for each lattice point) × (each primitive site).
  const std::size_t n_prim = lat.sites.size();
  std::vector<vec3_t> super_pos;
  std::vector<int> super_prim;
  super_pos.reserve(pts.size() * n_prim);
  super_prim.reserve(pts.size() * n_prim);
  for (const auto &t : pts) {
    for (std::size_t s = 0; s < n_prim; ++s) {
      super_pos.push_back(
          wrap_inside(t + lat.sites[s].frac, supercell, inv_super));
      super_prim.push_back(static_cast<int>(s));
    }
  }
  const std::size_t n_atoms = super_pos.size();

  // Random initial occupation, per sublattice, honouring composition (largest-
  // remainder rounding so per-sublattice counts are integer and sum exactly).
  std::vector<std::string> elements(n_atoms);
  std::mt19937 rng(seed);
  for (std::size_t s = 0; s < n_prim; ++s) {
    std::vector<std::size_t> idx;
    idx.reserve(n_atoms);
    std::ranges::copy_if(
        std::views::iota(std::size_t{0}, n_atoms), std::back_inserter(idx),
        [&](std::size_t a) { return super_prim[a] == static_cast<int>(s); });

    const auto &occ = lat.sites[s].occ;
    const std::size_t n = idx.size();
    std::vector<std::size_t> counts(occ.size(), 0);
    std::vector<double> frac(occ.size(), 0.0);
    std::size_t base = 0;
    for (std::size_t k = 0; k < occ.size(); ++k) {
      const double e = occ[k].second * static_cast<double>(n);
      counts[k] = static_cast<std::size_t>(std::floor(e));
      frac[k] = e - static_cast<double>(counts[k]);
      base += counts[k];
    }
    std::vector<std::size_t> order(occ.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::sort(order.begin(), order.end(),
              [&](std::size_t a, std::size_t b) { return frac[a] > frac[b]; });
    for (std::size_t r = 0; base < n && r < order.size(); ++r, ++base) {
      counts[order[r]]++;
    }
    std::vector<std::string> pool;
    pool.reserve(n);
    for (std::size_t k = 0; k < occ.size(); ++k) {
      for (std::size_t c = 0; c < counts[k]; ++c) {
        pool.push_back(occ[k].first);
      }
    }
    std::shuffle(pool.begin(), pool.end(), rng);
    for (std::size_t a = 0; a < idx.size() && a < pool.size(); ++a) {
      elements[idx[a]] = pool[a];
    }
  }

  // Assemble the AtomicStructure (coords are cosmetic — the constraint ignores
  // them; residue = sublattice id so SpeciesSwap respects sublattice
  // boundaries).
  EnumeratedSqs out;
  out.occ_index = occ_index;
  out.table = table;
  out.axes = lat.axes;
  out.supercell = supercell;
  out.frac_positions = super_pos;

  AtomicStructure &st = out.structure;
  st.coordinates.resize(static_cast<Eigen::Index>(n_atoms), 3);
  st.atomic_numbers.resize(static_cast<Eigen::Index>(n_atoms));
  st.elements.reserve(n_atoms);
  st.names.reserve(n_atoms);
  st.residues.reserve(n_atoms);
  st.molecule_ids.reserve(n_atoms);
  for (std::size_t a = 0; a < n_atoms; ++a) {
    const vec3_t cart = lat.axes * super_pos[a];
    st.coordinates.row(static_cast<Eigen::Index>(a)) = cart.transpose();
    st.atomic_numbers[static_cast<Eigen::Index>(a)] = occ_index[elements[a]];
    st.elements.push_back(elements[a]);
    st.names.push_back(elements[a]);
    st.residues.push_back("SL" + std::to_string(super_prim[a]));
    st.molecule_ids.push_back(a);
  }

  // Orbits: pairs and larger (corrdump is run with -noe -nop).
  for (const auto &r : raw) {
    if (r.points.size() < 2) {
      continue;
    }
    MC rep;
    for (const auto &p : r.points) {
      rep.clus.push_back(p.coord);
      rep.func.push_back(p.func);
      rep.site_type.push_back(p.site_type);
    }
    const std::vector<MC> images = find_equivalent(rep, sym, inv_cell);

    ClusterOrbit orbit;
    orbit.weight = 1.0;
    orbit.funcs = rep.func;
    orbit.site_types = rep.site_type;
    orbit.body = rep.clus.size();
    orbit.target = compute_target(rep, lat, table, occ_index, inv_cell);
    orbit.flat_sites.reserve(images.size() * pts.size() * orbit.body);
    for (const auto &img : images) {
      for (const auto &t : pts) {
        const std::size_t mark = orbit.flat_sites.size();
        bool ok = true;
        for (const auto &cp : img.clus) {
          const int site = which_atom(t + cp, super_pos, inv_super);
          if (site < 0) {
            ok = false;
            break;
          }
          orbit.flat_sites.push_back(static_cast<std::size_t>(site));
        }
        if (!ok) {
          orbit.flat_sites.resize(mark); // discard the partial instance
        }
      }
    }
    out.orbits.push_back(std::move(orbit));
  }

  return out;
}

} // namespace RMC::atat
