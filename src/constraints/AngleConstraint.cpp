#include <RMC/constraints/AngleConstraint.hpp>
#include <algorithm>
#include <array>
#include <cmath>

namespace RMC {

double AngleConstraint::compute_error(const coords_t &coords,
                                      std::span<const std::size_t> moved) const {
  return cache_.compute(
      triplets_, [](const Triplet &t) { return std::array{t.i, t.j, t.k}; },
      [this](const coords_t &c, const Triplet &t) {
        return triplet_error(c, t);
      },
      coords, moved);
}

double AngleConstraint::triplet_error(const coords_t &coords,
                                      const Triplet &t) const noexcept {
  vec3_t v1 = (coords.row(t.i) - coords.row(t.j)).transpose();
  vec3_t v2 = (coords.row(t.k) - coords.row(t.j)).transpose();
  if (bc_) {
    v1 = bc_min_image(*bc_, v1);
    v2 = bc_min_image(*bc_, v2);
  }
  const double cos_a = v1.dot(v2) / (v1.norm() * v2.norm() + 1e-30);
  const double angle = std::acos(std::clamp(cos_a, -1.0, 1.0));
  return range_violation(angle, t.lo, t.hi);
}

} // namespace RMC
