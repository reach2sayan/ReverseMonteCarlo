#pragma once
#include <RMC/constraints/Constraint.hpp>
#include <algorithm>
#include <boost/assert.hpp>
#include <vector>

namespace RMC {

class ConstraintCollection {
public:
  // Insert in ascending computation_cost() order so cheap constraints
  // (bonds, angles) run before expensive O(N²) ones (PDF, S(Q)).
  constexpr void add(Constraint c) {
    if (c.is_singular()) {
      BOOST_ASSERT_MSG(std::ranges::none_of(constraints_,
                                            [&c](const Constraint &e) {
                                              return e.is_singular() &&
                                                     e.name() == c.name();
                                            }),
                       "ConstraintCollection: duplicate singular constraint");
    }
    if (bc_.has_value()) {
      c.set_boundary_conditions(bc_.value());
    }
    const double cost = c.computation_cost();
    auto it = std::lower_bound(constraints_.begin(), constraints_.end(), cost,
                               [](const Constraint &x, double v) {
                                 return x.computation_cost() < v;
                               });
    constraints_.insert(it, std::move(c));
  }

  constexpr void
  set_boundary_conditions(const BoundaryConditions &bc) noexcept {
    bc_ = bc;
    std::ranges::for_each(
        constraints_, [&](Constraint &c) { c.set_boundary_conditions(bc); });
  }

  constexpr void compute_before_move(const coords_t &coords,
                                     std::span<const std::size_t> moved) {
    std::ranges::for_each(constraints_, [&](Constraint &c) {
      c.compute_before_move(coords, moved);
    });
  }

  // Run constraints cheapest-first. Stop as soon as one would reject — the
  // rest are skipped (their reject() resets err_after_ to err_before_ so
  // total_error() stays consistent).
  constexpr void compute_after_move(const coords_t &coords,
                                    std::span<const std::size_t> moved) {
    for (auto &c : constraints_) {
      c.compute_after_move(coords, moved);
      if (c.should_reject()) {
        break;
      }
    }
  }

  [[nodiscard]] constexpr bool should_reject() const noexcept {
    return std::ranges::any_of(
        constraints_, [](const Constraint &c) { return c.should_reject(); });
  }

  constexpr void accept() noexcept {
    std::ranges::for_each(constraints_, [](Constraint &c) { c.accept(); });
  }
  constexpr void reject() noexcept {
    std::ranges::for_each(constraints_, [](Constraint &c) { c.reject(); });
  }

  [[nodiscard]] constexpr double total_error() const noexcept {
    return std::ranges::fold_left(
        constraints_, 0.0,
        [](double acc, const auto &c) { return acc + c.standard_error(); });
  }

  constexpr void set_n_frames(std::size_t n) noexcept {
    std::ranges::for_each(constraints_,
                          [n](Constraint &c) { c.set_n_frames(n); });
  }
  constexpr void set_active_frame(std::size_t k) noexcept {
    std::ranges::for_each(constraints_,
                          [k](Constraint &c) { c.set_active_frame(k); });
  }

  [[nodiscard]] constexpr std::size_t size() const noexcept {
    return constraints_.size();
  }

  [[nodiscard]] constexpr Constraint &operator[](std::size_t i) {
    return constraints_[i];
  }
  [[nodiscard]] constexpr const Constraint &operator[](std::size_t i) const {
    return constraints_[i];
  }

private:
  std::vector<Constraint> constraints_;
  std::optional<BoundaryConditions> bc_{std::nullopt};
};

} // namespace RMC
