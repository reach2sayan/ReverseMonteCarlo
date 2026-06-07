#pragma once
#include <RMC/constraints/Constraint.hpp>
#include <RMC/constraints/IncrementalCache.hpp>
#include <vector>

namespace RMC {

// Enforces bond-angle bounds for atom triplets (i–j–k), angle at j.
//
// Incremental: on a single-atom move, only triplets touching that atom are
// recomputed via ItemCache. All others retain their cached contribution.
class AngleConstraint : public RigidConstraintBase<AngleConstraint> {
public:
  struct Triplet {
    std::size_t i, j, k;
    double lo, hi;
  };

  void add_angle(std::size_t i, std::size_t j, std::size_t k, double lo_rad,
                 double hi_rad) {
    triplets_.push_back({i, j, k, lo_rad, hi_rad});
    cache_.invalidate();
  }

  [[nodiscard]] static constexpr std::string_view name() noexcept {
    return "AngleConstraint";
  }
  [[nodiscard]] double compute_error(const coords_t &coords,
                                     std::span<const std::size_t> moved) const;

private:
  double triplet_error(const coords_t &coords, const Triplet &t) const noexcept;
  std::vector<Triplet> triplets_;
  mutable ItemCache<Triplet> cache_;
};

static_assert(CConstraint<AngleConstraint>,
              "AngleConstraint must satisfy the CConstraint concept");

} // namespace RMC
