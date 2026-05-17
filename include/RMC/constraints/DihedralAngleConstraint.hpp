#pragma once
#include <RMC/constraints/Constraint.hpp>
#include <cmath>
#include <vector>

namespace RMC {

// Enforces dihedral-angle bounds for atom quadruplets (i–j–k–l).
class DihedralAngleConstraint : public ConstraintBase<DihedralAngleConstraint> {
public:
  struct Quad {
    std::size_t i, j, k, l;
    double lo, hi;
  };

  constexpr void add_dihedral(std::size_t i, std::size_t j, std::size_t k,
                              std::size_t l, double lo_rad, double hi_rad) {
    quads_.push_back({i, j, k, l, lo_rad, hi_rad});
  }

  [[nodiscard]] std::string name() const { return "DihedralAngleConstraint"; }

  [[nodiscard]] double
  compute_error(const coords_t &coords,
                std::span<const std::size_t> /*moved*/) const {
    double err = 0.0;
    std::ranges::for_each(quads_, [&](const Quad &q) {
      double phi = dihedral(coords, q.i, q.j, q.k, q.l);
      if (phi < q.lo) {
        err += (q.lo - phi);
      } else if (phi > q.hi) {
        err += (phi - q.hi);
      }
    });
    return err;
  }

private:
  std::vector<Quad> quads_;

  [[nodiscard]] double dihedral(const coords_t &c, std::size_t i, std::size_t j,
                                std::size_t k, std::size_t l) const noexcept {
    auto get = [&](std::size_t a) -> vec3_t { return c.row(a).transpose(); };
    vec3_t b1 = get(j) - get(i);
    vec3_t b2 = get(k) - get(j);
    vec3_t b3 = get(l) - get(k);
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

} // namespace RMC
