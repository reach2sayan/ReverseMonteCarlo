#pragma once
#include <RMC/constraints/Constraint.hpp>
#include <RMC/constraints/IncrementalCache.hpp>
#include <algorithm>
#include <array>
#include <string>
#include <vector>

namespace RMC {

// Enforces bond-length bounds for explicit atom-index pairs.
// std_err = Σ violations, where violation > 0 means bond is outside [lo, hi].
//
// Incremental: on a single-atom move, only bonds touching that atom are
// recomputed via ItemCache. All others retain their cached contribution.
class BondConstraint : public RigidConstraintBase<BondConstraint> {
public:
  struct BondItem {
    std::size_t i, j;
    double lo, hi;
  };

  void add_bond(std::size_t i, std::size_t j, double lo, double hi) {
    bonds_.push_back({std::min(i, j), std::max(i, j), lo, hi});
    cache_.invalidate();
  }

  [[nodiscard]] static constexpr std::string_view name() noexcept {
    return "BondConstraint";
  }

  [[nodiscard]] double compute_error(const coords_t &coords,
                                     std::span<const std::size_t> moved) const {
    return cache_.compute(
        bonds_, [](const BondItem &b) { return std::array{b.i, b.j}; },
        [this](const coords_t &c, const BondItem &b) {
          return range_violation(distance(c, b.i, b.j), b.lo, b.hi);
        },
        coords, moved);
  }

private:
  std::vector<BondItem> bonds_;
  mutable ItemCache<BondItem> cache_;
};

} // namespace RMC
