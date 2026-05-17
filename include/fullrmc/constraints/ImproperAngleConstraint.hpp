#pragma once
#include <cmath>
#include <fullrmc/constraints/Constraint.hpp>
#include <vector>

namespace fullrmc {

// Improper dihedral: enforces planarity / chirality for quadruplets (i,j,k,l)
// where the angle is between plane(i,j,k) and plane(j,k,l).
class ImproperAngleConstraint : public ConstraintBase<ImproperAngleConstraint> {
public:
  struct Quad {
    std::size_t i, j, k, l;
    double lo, hi;
  };

  void add_improper(std::size_t i, std::size_t j, std::size_t k, std::size_t l,
                    double lo_rad, double hi_rad) {
    quads_.push_back({i, j, k, l, lo_rad, hi_rad});
  }

  [[nodiscard]] std::string name() const override {
    return "ImproperAngleConstraint";
  }

  [[nodiscard]] double
  compute_error(const coords_t &coords,
                std::span<const std::size_t> /*moved*/) const {
    double err = 0.0;
    for (auto &q : quads_) {
      double phi = improper(coords, q.i, q.j, q.k, q.l);
      if (phi < q.lo)
        err += (q.lo - phi);
      else if (phi > q.hi)
        err += (phi - q.hi);
    }
    return err;
  }

private:
  std::vector<Quad> quads_;

  // Same formula as dihedral – improper is just a dihedral with a
  // different atom ordering chosen to measure out-of-plane distortion.
  [[nodiscard]] double improper(const coords_t &c, std::size_t i, std::size_t j,
                                std::size_t k, std::size_t l) const noexcept {
    vec3_t b1 = c.row(j).transpose() - c.row(i).transpose();
    vec3_t b2 = c.row(k).transpose() - c.row(j).transpose();
    vec3_t b3 = c.row(l).transpose() - c.row(k).transpose();
    if (bc_) {
      b1 = bc_min_image(*bc_, b1);
      b2 = bc_min_image(*bc_, b2);
      b3 = bc_min_image(*bc_, b3);
    }
    vec3_t n1 = b1.cross(b2);
    vec3_t n2 = b2.cross(b3);
    double x = n1.dot(n2);
    double y = (n1.cross(n2)).dot(b2.normalized());
    return std::atan2(y, x);
  }
};

} // namespace fullrmc
