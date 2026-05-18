#pragma once
#include <RMC/constraints/Constraint.hpp>
#include <RMC/constraints/IncrementalCache.hpp>
#include <array>
#include <cmath>
#include <string>
#include <vector>

namespace RMC {

// Enforces bond-angle bounds for atom triplets (i–j–k), angle at j.
//
// Incremental: on a single-atom move, only triplets touching that atom are
// recomputed via ItemCache. All others retain their cached contribution.
class AngleConstraint : public ConstraintBase<AngleConstraint> {
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

  [[nodiscard]] std::string name() const { return "AngleConstraint"; }

  [[nodiscard]] double compute_error(const coords_t &coords,
                                     std::span<const std::size_t> moved) const {
    return cache_.compute(
        triplets_,
        [](const Triplet &t) { return std::array{t.i, t.j, t.k}; },
        [this](const coords_t &c, const Triplet &t) {
          return triplet_error(c, t);
        },
        coords, moved);
  }

private:
  double triplet_error(const coords_t &coords, const Triplet &t) const {
    vec3_t v1 = (coords.row(t.i) - coords.row(t.j)).transpose();
    vec3_t v2 = (coords.row(t.k) - coords.row(t.j)).transpose();
    if (bc_) {
      v1 = bc_min_image(*bc_, v1);
      v2 = bc_min_image(*bc_, v2);
    }
    const double cos_a = v1.dot(v2) / (v1.norm() * v2.norm() + 1e-30);
    const double angle = std::acos(std::clamp(cos_a, -1.0, 1.0));
    if (angle < t.lo)
      return t.lo - angle;
    if (angle > t.hi)
      return angle - t.hi;
    return 0.0;
  }

  std::vector<Triplet> triplets_;
  mutable ItemCache<Triplet> cache_;
};

} // namespace RMC
