#pragma once
#include <RMC/constraints/Constraint.hpp>
#include <RMC/core/Structure.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <numeric>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace RMC {

// One symmetry-equivalent instance of a cluster: a specific set of site
// indices in the supercell whose occupation product contributes to the orbit's
// correlation. The index order matches the orbit's per-point func/site_type.
struct ClusterInstance {
  std::vector<std::size_t> sites; // site indices (size = cluster body count)
};

// All symmetry-equivalent instances of one cluster type (orbit).
struct ClusterOrbit {
  std::vector<ClusterInstance> instances;
  double target = 0.0; // target correlation (0 = random equiatomic binary)
  double weight = 1.0; // weight in objective function
  // Multicomponent (ATAT) metadata, per cluster point. Empty ⇒ binary defaults
  // (site_type 0, func 0) so legacy binary orbits keep working unchanged.
  std::vector<int> funcs;
  std::vector<int> site_types;
};

// ATAT trigonometric (Chebyshev-like) site-basis table: value indexed by
// [site_type][func][occupation], where site_type = (#components − 2). For binary
// it reduces to {occ 0 → −1, occ 1 → +1}. Port of TrigoCorrFuncTable::init
// (atat/src/calccorr.c++:163-180).
struct CorrFuncTable {
  std::vector<std::vector<std::vector<double>>> t; // [site_type][func][occ]

  [[nodiscard]] double value(int site_type, int func, int occ) const {
    return t[static_cast<std::size_t>(site_type)][static_cast<std::size_t>(func)]
            [static_cast<std::size_t>(occ)];
  }

  [[nodiscard]] static CorrFuncTable trigonometric(int max_components) {
    CorrFuncTable tab;
    if (max_components < 2) {
      max_components = 2;
    }
    tab.t.resize(static_cast<std::size_t>(max_components - 1));
    for (int m = 2; m <= max_components; ++m) {
      auto &site = tab.t[static_cast<std::size_t>(m - 2)];
      site.resize(static_cast<std::size_t>(m - 1));
      for (int f = 0; f < m - 1; ++f) {
        site[static_cast<std::size_t>(f)].assign(static_cast<std::size_t>(m),
                                                 0.0);
      }
      for (int s = 0; s < m; ++s) {
        for (int k = 1; k <= m / 2; ++k) {
          site[static_cast<std::size_t>(2 * k - 2)][static_cast<std::size_t>(s)] =
              -std::cos(2.0 * M_PI * s * k / m);
        }
        for (int k = 1; k <= (m + 1) / 2 - 1; ++k) {
          site[static_cast<std::size_t>(2 * k - 1)][static_cast<std::size_t>(s)] =
              -std::sin(2.0 * M_PI * s * k / m);
        }
      }
    }
    return tab;
  }
};

// Constraint for Special Quasi-random Structure (SQS) search.
//
// Tracks the weighted χ² deviation of multi-body cluster correlations from
// target values. Works with SpeciesSwapGenerator: coordinates are ignored; only
// structure.elements (species labels) are read.
//
// Two ways to define the site basis:
//   • SpeciesMap — element → scalar σ (binary / simple linear encoding):
//       {{"Cu", +1.0}, {"Au", -1.0}}
//   • CorrFuncTable + occupation-index map — the full ATAT (m−1) trigonometric
//     basis for correct multicomponent (m>2) correlations; supplied by the
//     ClusterEnumerator from corrdump's clusters.out.
//
// Cluster instances are pre-computed (ATAT corrdump + symmetry enumeration) and
// passed in as ClusterOrbit objects.
class ClusterCorrelationConstraint
    : public ConstraintBase<ClusterCorrelationConstraint> {
public:
  // Maps element symbol → occupation function value σ (typically ±1).
  using SpeciesMap = std::unordered_map<std::string, double>;

  // Binary / linear-encoding constructor (back-compatible). Builds a degenerate
  // single-function table whose value for each element is its scalar σ.
  ClusterCorrelationConstraint(const AtomicStructure &structure,
                               const SpeciesMap &species_map,
                               std::vector<ClusterOrbit> orbits)
      : structure_(structure), orbits_(std::move(orbits)) {
    table_.t.resize(1);
    table_.t[0].resize(1);
    table_.t[0][0].resize(species_map.size(), 0.0);
    for (const auto [i, kv] : std::views::enumerate(species_map)) {
      const auto &[elem, sigma] = kv;
      occ_index_[elem] = static_cast<int>(i);
      table_.t[0][0][static_cast<std::size_t>(i)] = sigma;
    }
  }

  // Multicomponent constructor: explicit occupation-index map (element →
  // 0..m−1) and a CorrFuncTable (e.g. CorrFuncTable::trigonometric(m)).
  ClusterCorrelationConstraint(const AtomicStructure &structure,
                               std::unordered_map<std::string, int> occ_index,
                               CorrFuncTable table,
                               std::vector<ClusterOrbit> orbits)
      : structure_(structure), occ_index_(std::move(occ_index)),
        table_(std::move(table)), orbits_(std::move(orbits)) {}

  // Required by ConstraintBase<Derived> — called by compute_before/after_move.
  // Ignores coords and moved (correlations are global, not incremental).
  [[nodiscard]] double compute_error(const coords_t &,
                                     std::span<const std::size_t>) {
    return std::ranges::fold_left(
        orbits_, 0.0, [this](double total, const ClusterOrbit &orbit) {
          const double dev = orbit_correlation(orbit) - orbit.target;
          return total + orbit.weight * dev * dev;
        });
  }

  [[nodiscard]] static constexpr std::string_view name() noexcept {
    return "ClusterCorrelation";
  }

  // More expensive than cheap geometric constraints.
  [[nodiscard]] double computation_cost(Constraint::Token) const noexcept {
    return static_cast<double>(total_instances_) * 10.0;
  }

  [[nodiscard]] std::vector<double> current_correlations() const {
    auto corr = orbits_ | std::views::transform([this](const auto &orbit) {
                  return orbit_correlation(orbit);
                });
    return {corr.begin(), corr.end()};
  }

  [[nodiscard]] const std::vector<ClusterOrbit> &orbits() const noexcept {
    return orbits_;
  }

private:
  [[nodiscard]] double orbit_correlation(const ClusterOrbit &orbit) const {
    if (orbit.instances.empty()) {
      return 0.0;
    }
    double sum = 0.0;
    for (const auto &inst : orbit.instances) {
      double prod = 1.0;
      for (const auto [p, site] : std::views::enumerate(inst.sites)) {
        const auto it = occ_index_.find(structure_.elements[site]);
        if (it == occ_index_.end()) {
          prod = 0.0; // unknown species → zero contribution (legacy behaviour)
          break;
        }
        const int st = orbit.site_types.empty() ? 0 : orbit.site_types[p];
        const int fn = orbit.funcs.empty() ? 0 : orbit.funcs[p];
        prod *= table_.value(st, fn, it->second);
      }
      sum += prod;
    }
    return sum / static_cast<double>(orbit.instances.size());
  }

  const AtomicStructure &structure_;
  std::unordered_map<std::string, int> occ_index_; // element → occupation index
  CorrFuncTable table_;
  std::vector<ClusterOrbit> orbits_;
  std::size_t total_instances_ = [this] {
    return std::transform_reduce(
        orbits_.begin(), orbits_.end(), std::size_t{0}, std::plus<>{},
        [](const auto &o) { return o.instances.size(); });
  }();
};

static_assert(CConstraint<ClusterCorrelationConstraint>,
              "ClusterCorrelationConstraint must satisfy the CConstraint concept");

} // namespace RMC
