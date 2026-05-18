#pragma once
#include <RMC/constraints/Constraint.hpp>
#include <RMC/constraints/IncrementalCache.hpp>
#include <array>
#include <cmath>
#include <string>
#include <vector>

namespace RMC {

// Improper dihedral: enforces planarity / chirality for quadruplets (i,j,k,l)
// where the angle is between plane(i,j,k) and plane(j,k,l).
//
// Incremental: on a single-atom move, only quads touching that atom are
// recomputed via ItemCache. All others retain their cached contribution.
class ImproperAngleConstraint : public ConstraintBase<ImproperAngleConstraint> {
public:
  struct Quad {
    std::size_t i, j, k, l;
    double lo, hi;
  };

  void add_improper(std::size_t i, std::size_t j, std::size_t k, std::size_t l,
                    double lo_rad, double hi_rad) {
    quads_.push_back({i, j, k, l, lo_rad, hi_rad});
    cache_.invalidate();
  }

  [[nodiscard]] constexpr std::string name() const { return "ImproperAngleConstraint"; }
  [[nodiscard]] constexpr double compute_error(const coords_t &coords,
                                     std::span<const std::size_t> moved) const {
    return cache_.compute(
        quads_, [](const Quad &q) { return std::array{q.i, q.j, q.k, q.l}; },
        [this](const coords_t &c, const Quad &q) { return quad_error(c, q); },
        coords, moved);
  }

private:
  double quad_error(const coords_t &c, const Quad &q) const noexcept {
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
    if (phi < q.lo) {
      return q.lo - phi;
    }
    if (phi > q.hi) {
      return phi - q.hi;
    }
    return 0.0;
  }

  std::vector<Quad> quads_;
  mutable ItemCache<Quad> cache_;
};

} // namespace RMC
