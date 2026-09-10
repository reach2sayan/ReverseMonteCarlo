#pragma once
#include <RMC/constraints/Constraint.hpp>
#include <RMC/constraints/IncrementalCache.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <string_view>
#include <vector>

namespace RMC {

// Kernels: a scalar of the min-image bond vectors b[n] = r[n+1] − r[n] along a
// chain of `arity` atoms.
namespace geom {
struct Distance {
  static constexpr std::size_t arity = 2;
  static double value(const std::array<vec3_t, 1> &b) { return b[0].norm(); }
};
// Bond angle at the middle atom of i–j–k.
struct Angle {
  static constexpr std::size_t arity = 3;
  static double value(const std::array<vec3_t, 2> &b) {
    const vec3_t u = -b[0];
    return std::acos(
        std::clamp(u.dot(b[1]) / (u.norm() * b[1].norm() + 1e-30), -1.0, 1.0));
  }
};
// Torsion of i–j–k–l: the angle between planes (i,j,k) and (j,k,l). Proper and
// improper dihedrals are the same quantity.
struct Dihedral {
  static constexpr std::size_t arity = 4;
  static double value(const std::array<vec3_t, 3> &b) {
    const vec3_t n1 = b[0].cross(b[1]);
    const vec3_t n2 = b[1].cross(b[2]);
    return std::atan2(n1.cross(n2).dot(b[1].normalized()), n1.dot(n2));
  }
};
} // namespace geom

// Rigid constraint keeping Kernel(atoms) inside [lo, hi] for a list of atom
// tuples; the error is the summed range violation. Incremental: a move
// recomputes only the tuples touching a moved atom (ItemCache).
template <class Kernel>
class GeometricRangeConstraint
    : public RigidConstraintBase<GeometricRangeConstraint<Kernel>> {
public:
  using Atoms = std::array<std::size_t, Kernel::arity>;
  struct Item {
    Atoms atoms;
    double lo, hi;
  };

  void add(const Atoms &atoms, double lo, double hi) {
    items_.push_back({atoms, lo, hi});
    cache_.invalidate();
  }

  [[nodiscard]] double compute_error(const coords_t &coords,
                                     std::span<const std::size_t> moved) const {
    return cache_.compute(
        items_, [](const Item &it) { return it.atoms; },
        [this](const coords_t &c, const Item &it) {
          // A term through a removed atom no longer exists.
          if (std::ranges::any_of(it.atoms,
                                  [this](std::size_t a) { return this->absent(a); })) {
            return 0.0;
          }
          std::array<vec3_t, Kernel::arity - 1> b;
          for (std::size_t n = 0; n < b.size(); ++n) {
            b[n] = c.row(it.atoms[n + 1]).transpose() -
                   c.row(it.atoms[n]).transpose();
            if (this->bc_) {
              b[n] = this->bc_->min_image(b[n]);
            }
          }
          return range_violation(Kernel::value(b), it.lo, it.hi);
        },
        coords, moved);
  }

private:
  std::vector<Item> items_;
  mutable ItemCache<Item> cache_;
};

// Bond-length bounds for atom pairs.
struct BondConstraint : GeometricRangeConstraint<geom::Distance> {
  void add_bond(std::size_t i, std::size_t j, double lo, double hi) {
    add({std::min(i, j), std::max(i, j)}, lo, hi);
  }
  [[nodiscard]] static constexpr std::string_view name() noexcept {
    return "BondConstraint";
  }
};

// Bond-angle bounds for triplets (i–j–k), angle at j.
struct AngleConstraint : GeometricRangeConstraint<geom::Angle> {
  void add_angle(std::size_t i, std::size_t j, std::size_t k, double lo_rad,
                 double hi_rad) {
    add({i, j, k}, lo_rad, hi_rad);
  }
  [[nodiscard]] static constexpr std::string_view name() noexcept {
    return "AngleConstraint";
  }
};

// Dihedral-angle bounds for quadruplets (i–j–k–l).
struct DihedralAngleConstraint : GeometricRangeConstraint<geom::Dihedral> {
  void add_dihedral(std::size_t i, std::size_t j, std::size_t k, std::size_t l,
                    double lo_rad, double hi_rad) {
    add({i, j, k, l}, lo_rad, hi_rad);
  }
  [[nodiscard]] static constexpr std::string_view name() noexcept {
    return "DihedralAngleConstraint";
  }
};

// Improper dihedral: planarity / chirality of (i,j,k,l), the angle between
// plane(i,j,k) and plane(j,k,l).
struct ImproperAngleConstraint : GeometricRangeConstraint<geom::Dihedral> {
  void add_improper(std::size_t i, std::size_t j, std::size_t k, std::size_t l,
                    double lo_rad, double hi_rad) {
    add({i, j, k, l}, lo_rad, hi_rad);
  }
  [[nodiscard]] static constexpr std::string_view name() noexcept {
    return "ImproperAngleConstraint";
  }
};

} // namespace RMC
