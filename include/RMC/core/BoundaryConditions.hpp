#pragma once
#include <RMC/core/Types.hpp>
#include <seitz/core/lattice.hpp>
#include <seitz/core/periodicity.hpp>

namespace RMC {

// A periodic cell over a seitz::Lattice (box columns are the cell vectors), or
// open space with a caller-supplied volume. PeriodicBC / InfiniteBC name the
// two constructors; they add no state and slice to this type on copy.
class BoundaryConditions {
public:
  explicit BoundaryConditions(const mat3_t &box)
      : lattice_{box}, inv_box_{box.inverse()},
        periodicity_{seitz::all_periodic()}, volume_{lattice_.volume()} {}
  explicit BoundaryConditions(double volume = 1.0)
      : periodicity_{seitz::none_periodic()}, volume_{volume} {}

  [[nodiscard]] bool periodic() const noexcept {
    return periodicity_ != seitz::none_periodic();
  }
  [[nodiscard]] const mat3_t &box() const noexcept { return lattice_.matrix(); }
  [[nodiscard]] const mat3_t &inv_box() const noexcept { return inv_box_; }
  [[nodiscard]] double volume() const noexcept { return volume_; }

  // Fold a Cartesian position into the cell ([0,1) fractional).
  [[nodiscard]] vec3_t wrap(const vec3_t &r) const noexcept {
    return periodic() ? box() * seitz::wrap(inv_box_ * r, periodicity_) : r;
  }
  // Minimum-image displacement.
  [[nodiscard]] vec3_t min_image(const vec3_t &d) const noexcept {
    return periodic() ? box() * seitz::minimal_image(inv_box_ * d, periodicity_)
                      : d;
  }

private:
  seitz::Lattice lattice_;
  mat3_t inv_box_{mat3_t::Identity()};
  seitz::CellPeriodicity periodicity_;
  double volume_;
};

struct PeriodicBC : BoundaryConditions {
  explicit PeriodicBC(const mat3_t &box) : BoundaryConditions{box} {}
};
struct InfiniteBC : BoundaryConditions {
  explicit InfiniteBC(double volume = 1.0) : BoundaryConditions{volume} {}
};

} // namespace RMC
