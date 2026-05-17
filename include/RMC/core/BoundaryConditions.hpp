#pragma once
#include <cmath>
#include <RMC/core/Types.hpp>
#include <variant>

namespace RMC {

class PeriodicBC {
public:
  explicit PeriodicBC(const mat3_t &box) { set_box(box); }
  void set_box(const mat3_t &box) {
    box_ = box;
    inv_box_ = box.inverse();
    volume_ = std::abs(box.determinant());
  }

  [[nodiscard]] constexpr const mat3_t &box() const noexcept { return box_; }
  [[nodiscard]] constexpr const mat3_t &inv_box() const noexcept { return inv_box_; }

  // Wrap a Cartesian position back into the unit cell [0,1)^3 in fractional
  // coords.
  [[nodiscard]] vec3_t wrap(const vec3_t &r) const noexcept {
    vec3_t frac = inv_box_ * r;
    frac = frac.array() - frac.array().floor(); // [0,1)
    return box_ * frac;
  }

  [[nodiscard]] vec3_t min_image(const vec3_t &delta) const noexcept {
    vec3_t frac = inv_box_ * delta;
    frac = frac.array() - frac.array().round(); // [-0.5, 0.5)
    return box_ * frac;
  }

  [[nodiscard]] constexpr double volume() const noexcept { return volume_; }
private:
  mat3_t box_;
  mat3_t inv_box_;
  double volume_{0.0};
};

class InfiniteBC {
public:
  constexpr explicit InfiniteBC(double volume = 1.0) : volume_(volume) {}
  [[nodiscard]] vec3_t wrap(const vec3_t &r) const noexcept { return r; }
  [[nodiscard]] vec3_t min_image(const vec3_t &delta) const noexcept {
    return delta;
  }
  [[nodiscard]] constexpr double volume() const noexcept { return volume_; }
  constexpr void set_volume(double v) noexcept { volume_ = v; }

private:
  double volume_;
};

using BoundaryConditions = std::variant<PeriodicBC, InfiniteBC>;
inline vec3_t bc_wrap(const BoundaryConditions &bc, const vec3_t &r) {
  return std::visit([&](const auto &b) { return b.wrap(r); }, bc);
}
inline vec3_t bc_min_image(const BoundaryConditions &bc, const vec3_t &d) {
  return std::visit([&](const auto &b) { return b.min_image(d); }, bc);
}
inline double bc_volume(const BoundaryConditions &bc) {
  return std::visit([](const auto &b) { return b.volume(); }, bc);
}

} // namespace RMC
