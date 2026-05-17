#pragma once
#include <RMC/constraints/Constraint.hpp>
#include <vector>

namespace RMC {

class ConstraintCollection {
public:
  constexpr void add(IConstraint c) {
    if (bc_.has_value()) {
      c.set_boundary_conditions(bc_.value());
    }
    constraints_.push_back(std::move(c));
  }

  constexpr void
  set_boundary_conditions(const BoundaryConditions &bc) noexcept {
    bc_ = bc;
    std::ranges::for_each(
        constraints_, [&](IConstraint &c) { c.set_boundary_conditions(bc); });
  }

  constexpr void compute_before_move(const coords_t &coords,
                                     std::span<const std::size_t> moved) {
    std::ranges::for_each(constraints_, [&](IConstraint &c) {
      c.compute_before_move(coords, moved);
    });
  }

  constexpr void compute_after_move(const coords_t &coords,
                                    std::span<const std::size_t> moved) {
    std::ranges::for_each(constraints_, [&](IConstraint &c) {
      c.compute_after_move(coords, moved);
    });
  }

  [[nodiscard]] constexpr bool should_reject() const noexcept {
    return std::ranges::any_of(
        constraints_, [](const IConstraint &c) { return c.should_reject(); });
  }

  constexpr void accept() noexcept {
    std::ranges::for_each(constraints_, [](IConstraint &c) { c.accept(); });
  }
  constexpr void reject() noexcept {
    std::ranges::for_each(constraints_, [](IConstraint &c) { c.reject(); });
  }

  [[nodiscard]] constexpr double total_error() const noexcept {
    return std::ranges::fold_left(
        constraints_, 0.0,
        [](double acc, const auto &c) { return acc + c.standard_error(); });
  }

  [[nodiscard]] constexpr std::size_t size() const noexcept {
    return constraints_.size();
  }

  [[nodiscard]] constexpr IConstraint &operator[](std::size_t i) {
    return constraints_[i];
  }
  [[nodiscard]] constexpr const IConstraint &operator[](std::size_t i) const {
    return constraints_[i];
  }

private:
  std::vector<IConstraint> constraints_;
  std::optional<BoundaryConditions> bc_{std::nullopt};
};

} // namespace RMC
