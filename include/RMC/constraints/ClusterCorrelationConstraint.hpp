#pragma once
#include <RMC/constraints/Constraint.hpp>
#include <RMC/core/Structure.hpp>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace RMC {

// One symmetry-equivalent instance of a cluster: a specific set of site
// indices in the supercell whose occupation product contributes to the orbit's
// correlation.
struct ClusterInstance {
  std::vector<std::size_t> sites; // site indices (size = cluster body count)
};

// All symmetry-equivalent instances of one cluster type (orbit).
struct ClusterOrbit {
  std::vector<ClusterInstance> instances;
  double target = 0.0; // target correlation (0 = random alloy)
  double weight = 1.0; // weight in objective function
};

// Metropolis constraint for Special Quasi-random Structure (SQS) search.
//
// Tracks the weighted χ² deviation of multi-body cluster correlations from
// target values.  Designed to work with SpeciesSwapGenerator: coordinates are
// ignored; only structure.elements (species labels) are read.
//
// Occupation functions are defined by a SpeciesMap:
//   {{"Cu", +1.0}, {"Au", -1.0}}  — binary alloy
//   {{"A", +1.0}, {"B", 0.0}, {"C", -1.0}}  — ternary (linear encoding)
//
// Cluster instances are pre-computed externally (e.g., by ATAT's corrdump or
// equivalent symmetry analysis) and passed in as ClusterOrbit objects.
class ClusterCorrelationConstraint
    : public ConstraintBase<ClusterCorrelationConstraint> {
public:
  // Maps element symbol → occupation function value σ (typically ±1).
  using SpeciesMap = std::unordered_map<std::string, double>;

  ClusterCorrelationConstraint(const AtomicStructure &structure,
                               SpeciesMap species_map,
                               std::vector<ClusterOrbit> orbits)
      : structure_(structure), species_map_(std::move(species_map)),
        orbits_(std::move(orbits)) {}

  // Required by ConstraintBase<Derived> — called by compute_before/after_move.
  // Ignores coords and moved (correlations are global, not incremental).
  [[nodiscard]] double compute_error(const coords_t &,
                                     std::span<const std::size_t>) {
    double total = 0.0;
    for (const auto &orbit : orbits_) {
      const double corr = orbit_correlation(orbit);
      const double dev = corr - orbit.target;
      total += orbit.weight * dev * dev;
    }
    return total;
  }

  [[nodiscard]] static constexpr std::string_view name() noexcept {
    return "ClusterCorrelation";
  }

  // More expensive than cheap geometric constraints.
  [[nodiscard]] double computation_cost(Constraint::Token) const noexcept {
    return static_cast<double>(total_instances_) * 10.0;
  }

  [[nodiscard]] std::vector<double> current_correlations() const {
    std::vector<double> out;
    out.resize(orbits_.size());
    std::transform(
        orbits_.begin(), orbits_.end(), out.begin(),
        [this](const auto &orbit) { return orbit_correlation(orbit); });
    return out;
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
      for (const std::size_t site : inst.sites) {
        const auto it = species_map_.find(structure_.elements[site]);
        prod *= (it != species_map_.end()) ? it->second : 0.0;
      }
      sum += prod;
    }
    return sum / static_cast<double>(orbit.instances.size());
  }

  const AtomicStructure &structure_;
  SpeciesMap species_map_;
  std::vector<ClusterOrbit> orbits_;
  std::size_t total_instances_ = [this] {
    std::size_t n = 0;
    for (const auto &o : orbits_) {
      n += o.instances.size();
    }
    return n;
  }();
};

} // namespace RMC
