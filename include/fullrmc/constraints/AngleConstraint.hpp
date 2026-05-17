#pragma once
#include <cmath>
#include <fullrmc/constraints/Constraint.hpp>
#include <vector>

namespace fullrmc {

// Enforces bond-angle bounds for atom triplets (i–j–k), angle at j.
class AngleConstraint : public ConstraintBase<AngleConstraint> {

public:
  struct Triplet {
    std::size_t i, j, k;
    double lo, hi;
  };

  void add_angle(std::size_t i, std::size_t j, std::size_t k, double lo_rad,
                 double hi_rad) {
    triplets_.push_back({i, j, k, lo_rad, hi_rad});
  }

  [[nodiscard]] std::string name() const override { return "AngleConstraint"; }

  [[nodiscard]] double
  compute_error(const coords_t &coords,
                std::span<const std::size_t> /*moved*/) const {
    double err = 0.0;
    for (auto &t : triplets_) {
      vec3_t v1 = (coords.row(t.i) - coords.row(t.j)).transpose();
      vec3_t v2 = (coords.row(t.k) - coords.row(t.j)).transpose();
      if (bc_) {
        v1 = bc_min_image(*bc_, v1);
        v2 = bc_min_image(*bc_, v2);
      }
      double cos_a = v1.dot(v2) / (v1.norm() * v2.norm() + 1e-30);
      cos_a = std::clamp(cos_a, -1.0, 1.0);
      double angle = std::acos(cos_a);
      if (angle < t.lo)
        err += (t.lo - angle);
      else if (angle > t.hi)
        err += (angle - t.hi);
    }
    return err;
  }

private:
  std::vector<Triplet> triplets_;
};

} // namespace fullrmc
