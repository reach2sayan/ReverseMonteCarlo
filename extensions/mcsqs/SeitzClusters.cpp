#include "SeitzClusters.hpp"

#include <seitz/alloy/clusters_pool.hpp>
#include <seitz/alloy/parent_lattice.hpp>
#include <seitz/alloy/site_basis.hpp>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <random>
#include <ranges>
#include <stdexcept>
#include <string>
#include <vector>

namespace RMC::atat {
namespace {

namespace alloy = seitz::alloy;

// seitz reports failures as LEAF results; the mcsqs pipeline throws.
template <class T> T must(seitz::Result<T> r, const char *what) {
  if (!r) {
    throw std::runtime_error(std::string{"enumerate: "} + what);
  }
  return std::move(r).value();
}

// True if the fractional offset d is a lattice vector (to kClusterPrec).
bool on_lattice(const vec3_t &d) {
  return (d.array() - d.array().round()).abs().maxCoeff() < alloy::kClusterPrec;
}

// n labels drawn at the site's fractions: counts rounded by largest remainder
// (integers summing to n), then shuffled.
std::vector<std::string> random_fill(const LatticeSite &site, std::size_t n,
                                     std::mt19937 &rng) {
  struct Share {
    std::string label;
    std::size_t count;
    double remainder;
  };
  std::vector<Share> shares;
  std::size_t placed = 0;
  for (const auto &[label, x] : site.occ) {
    const double exact = x * static_cast<double>(n);
    const auto count = static_cast<std::size_t>(exact);
    shares.push_back({label, count, exact - static_cast<double>(count)});
    placed += count;
  }
  std::ranges::stable_sort(shares, std::greater{}, &Share::remainder);
  const auto leftover = static_cast<std::ptrdiff_t>(n > placed ? n - placed : 0);
  for (Share &s : shares | std::views::take(leftover)) {
    ++s.count;
  }
  std::vector<std::string> pool;
  for (const Share &s : shares) {
    pool.insert(pool.end(), s.count, s.label);
  }
  std::ranges::shuffle(pool, rng);
  return pool;
}

} // namespace

EnumeratedSqs enumerate(const AtatLattice &lat, const Eigen::Matrix3i &sc,
                        const Diameters &diameters, std::uint32_t seed) {
  EnumeratedSqs out;
  for (const auto [i, label] : lat.labels | std::views::enumerate) {
    out.occ_index[label] = static_cast<int>(i);
  }

  // Parent lattice: a site's species are the global ranks of its labels (seitz
  // only compares these sets, to find the sublattices).
  std::vector<alloy::SiteSpec> sites;
  for (const LatticeSite &site : lat.sites) {
    alloy::Species species;
    for (const std::string &label : site.occ | std::views::keys) {
      species.push_back(out.occ_index.at(label));
    }
    std::ranges::sort(species);
    sites.emplace_back(vec3_t(lat.cell.inverse() * site.frac),
                       std::move(species));
  }
  const auto parent = must(alloy::ParentLattice::from_sites(
                               seitz::Lattice{mat3_t(lat.axes * lat.cell)}, sites),
                           "invalid parent lattice");
  const auto pool = must(
      alloy::ClustersPool::generate(parent, {.radii = diameters,
                                             .include_empty = false,
                                             .include_points = false}),
      "cluster enumeration failed");
  const seitz::Cell &prim = parent.cell();
  const seitz::Cell super = must(prim.transformed(sc), "singular supercell");

  // Site basis, one block per sublattice j with k species: the species listed
  // l-th on its first site (corrdump numbers them in listed order) gets
  // block(f, global rank) = θ_f(l); zero on labels the sublattice cannot hold.
  // Sublattice ids are first-seen in site order, so a new id is the next block.
  const auto basis =
      alloy::SiteBasis::trigonometric(std::max(2, parent.max_species()));
  for (const auto &[s, site] : lat.sites | std::views::enumerate) {
    if (prim.type(s) != static_cast<int>(out.table.size())) {
      continue;
    }
    const int k = static_cast<int>(site.occ.size());
    mat_t block = mat_t::Zero(std::max(k - 1, 0),
                              static_cast<Eigen::Index>(lat.labels.size()));
    for (int f = 0; f + 1 < k; ++f) {
      for (const auto &[l, label] : site.occ | std::views::keys |
                                        std::views::enumerate) {
        block(f, out.occ_index.at(label)) = basis[k, f, static_cast<int>(l)];
      }
    }
    out.table.push_back(std::move(block));
  }

  // The supercell: atoms ordered parent-site-major over its n_cells lattice
  // points (seitz::Cell::transformed); residue "SL<j>" keeps species swaps
  // inside sublattice j.
  const seitz::Index n_cells = super.size() / prim.size();
  out.axes = lat.axes;
  out.supercell = lat.cell * sc.cast<double>();
  AtomicStructure &st = out.structure;
  st.coordinates = super.positions() * super.lattice().matrix().transpose();
  st.atomic_numbers.resize(super.size());
  st.elements.resize(static_cast<std::size_t>(super.size()));
  st.molecule_ids.resize(st.elements.size());
  std::ranges::iota(st.molecule_ids, std::size_t{0});
  for (const auto &[position, type] : super.atoms()) {
    st.residues.push_back("SL" + std::to_string(type));
    out.frac_positions.push_back(out.supercell * position);
  }

  // Random occupation per sublattice, at the fractions of its first site.
  std::mt19937 rng(seed);
  for (int j = 0; j < static_cast<int>(out.table.size()); ++j) {
    const auto members =
        std::views::iota(seitz::Index{0}, super.size()) |
        std::views::filter([&](seitz::Index i) { return super.type(i) == j; }) |
        std::ranges::to<std::vector>();
    const auto first = std::ranges::find(prim.types(), j) - prim.types().begin();
    for (const auto &[i, label] : std::views::zip(
             members, random_fill(lat.sites[static_cast<std::size_t>(first)],
                                  members.size(), rng))) {
      st.elements[static_cast<std::size_t>(i)] = label;
      st.atomic_numbers[i] = out.occ_index.at(label);
    }
  }
  st.names = st.elements;

  // Orbit instances: every image of the representative at every lattice point
  // t_k of the supercell. A point p (parent fractional) of parent site s lands
  // on one of s's n_cells images.
  const mat3_t to_super = sc.cast<double>().inverse();
  const auto parent_site = [&](const alloy::ClusterPoint &p) {
    for (seitz::Index s = 0; s < prim.size(); ++s) {
      if (on_lattice(p.position - prim.position(s))) {
        return s;
      }
    }
    throw std::runtime_error("enumerate: cluster point on no lattice site");
  };
  std::vector<vec3_t> translations; // t_k, from the images of parent site 0
  for (seitz::Index k = 0; k < n_cells; ++k) {
    translations.push_back(super.position(k) - to_super * prim.position(0));
  }
  const auto site_of = [&](const alloy::ClusterPoint &p, const vec3_t &t) {
    const vec3_t f = to_super * p.position + t;
    const seitz::Index first = parent_site(p) * n_cells;
    for (seitz::Index i = first; i < first + n_cells; ++i) {
      if (on_lattice(f - super.position(i))) {
        return static_cast<std::size_t>(i);
      }
    }
    throw std::runtime_error("enumerate: cluster point on no supercell site");
  };

  for (const alloy::Orbit &orbit : pool) {
    ClusterOrbit co;
    co.body = static_cast<std::size_t>(orbit.representative.size());
    co.target = 1.0;
    for (const alloy::ClusterPoint &p : orbit.representative) {
      const seitz::Index s = parent_site(p);
      const int j = prim.type(s);
      co.funcs.push_back(p.function);
      co.site_types.push_back(j);
      // Random-state target: Π over points of Σ x·θ(label) (ATAT
      // mcsqs.c++:456-481).
      double point = 0.0;
      for (const auto &[label, x] : lat.sites[static_cast<std::size_t>(s)].occ) {
        point += x * out.table[static_cast<std::size_t>(j)](
                         p.function, out.occ_index.at(label));
      }
      co.target *= point;
    }
    co.flat_sites.reserve(orbit.images.size() * translations.size() * co.body);
    for (const alloy::Cluster &image : orbit.images) {
      for (const vec3_t &t : translations) {
        for (const alloy::ClusterPoint &p : image) {
          co.flat_sites.push_back(site_of(p, t));
        }
      }
    }
    out.orbits.push_back(std::move(co));
  }
  return out;
}

} // namespace RMC::atat
