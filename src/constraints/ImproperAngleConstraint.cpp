#include <RMC/constraints/ImproperAngleConstraint.hpp>
#include <array>
#include <cmath>

namespace RMC {

double ImproperAngleConstraint::compute_error(
    const coords_t &coords, std::span<const std::size_t> moved) const {
  return cache_.compute(
      quads_, [](const Quad &q) { return std::array{q.i, q.j, q.k, q.l}; },
      [this](const coords_t &c, const Quad &q) { return quad_error(c, q); },
      coords, moved);
}

double ImproperAngleConstraint::quad_error(const coords_t &c,
                                           const Quad &q) const noexcept {
  vec3_t b1 = c.row(q.j).transpose() - c.row(q.i).transpose();
  vec3_t b2 = c.row(q.k).transpose() - c.row(q.j).transpose();
  vec3_t b3 = c.row(q.l).transpose() - c.row(q.k).transpose();
  if (bc_) {
    b1 = bc_min_image(*bc_, b1);
    b2 = bc_min_image(*bc_, b2);
    b3 = bc_min_image(*bc_, b3);
  }
  const vec3_t n1 = b1.cross(b2);
  const vec3_t n2 = b2.cross(b3);
  const double phi =
      std::atan2((n1.cross(n2)).dot(b2.normalized()), n1.dot(n2));
  return range_violation(phi, q.lo, q.hi);
}

} // namespace RMC
