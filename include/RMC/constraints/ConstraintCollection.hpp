#pragma once
#include <RMC/constraints/Constraint.hpp>
#include <algorithm>
#include <boost/assert.hpp>
#include <boost/stl_interfaces/view_interface.hpp>
#include <vector>

namespace RMC {

class ConstraintCollection
    : public boost::stl_interfaces::view_interface<
          ConstraintCollection,
          boost::stl_interfaces::element_layout::contiguous> {
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
    auto it = std::lower_bound(
        constraints_.begin(), constraints_.end(), cost,
        [](const Constraint &x, double v) { return x.computation_cost() < v; });
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

  // Run constraints cheapest-first. A RIGID constraint that worsens is a hard
  // gate, so once one rejects the remaining (often expensive) constraints are
  // skipped — their reject() resets err_after_ to err_before_ so total_error()
  // stays consistent. SOFT constraints never short-circuit: they are always
  // fully evaluated so the engine's Sampler sees the exact post-move total.
  constexpr void compute_after_move(const coords_t &coords,
                                    std::span<const std::size_t> moved) {
    for (auto &c : constraints_) {
      c.compute_after_move(coords, moved);
      if (c.is_rigid() && c.should_reject()) {
        break;
      }
    }
  }

  // Legacy per-constraint veto (any constraint worsened). Retained for tests
  // and callers that want the strict gate; the engine now uses
  // rigid_should_reject() plus the Sampler instead.
  [[nodiscard]] constexpr bool should_reject() const noexcept {
    return std::ranges::any_of(
        constraints_, [](const Constraint &c) { return c.should_reject(); });
  }

  // Hard gate for the engine: any RIGID constraint that worsened. Soft
  // constraints are deferred to the Sampler via total_error[_before]().
  [[nodiscard]] constexpr bool rigid_should_reject() const noexcept {
    return std::ranges::any_of(constraints_, [](const Constraint &c) {
      return c.is_rigid() && c.should_reject();
    });
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

  // Soft total error BEFORE the proposed move (rigid constraints contribute 0).
  // Paired with total_error() to form the ΔE handed to the Sampler.
  [[nodiscard]] constexpr double total_error_before() const noexcept {
    return std::ranges::fold_left(constraints_, 0.0,
                                  [](double acc, const auto &c) {
                                    return acc + c.standard_error_before();
                                  });
  }

  // Safety net: run each constraint's one-time initialise() (no-op for those
  // without one; idempotent for pair constraints). Lets the engine self-heal a
  // forgotten client-side pdc.initialise().
  constexpr void initialise_all() {
    std::ranges::for_each(constraints_, [](Constraint &c) { c.initialise(); });
  }

  constexpr void set_n_frames(std::size_t n) noexcept {
    std::ranges::for_each(constraints_,
                          [n](Constraint &c) { c.set_n_frames(n); });
  }
  constexpr void set_active_frame(std::size_t k) noexcept {
    std::ranges::for_each(constraints_,
                          [k](Constraint &c) { c.set_active_frame(k); });
  }

  // Read-only range surface. boost::stl_interfaces::view_interface synthesizes
  // size(), empty(), operator[], front(), back(), and data() from these. Only
  // const iterators are exposed, so callers cannot reorder or mutate elements
  // and thereby break the cost-ordering / BC-propagation invariants that add()
  // maintains. Internal logic iterates the member vector directly instead.
  [[nodiscard]] constexpr auto begin() const noexcept {
    return constraints_.cbegin();
  }
  [[nodiscard]] constexpr auto end() const noexcept {
    return constraints_.cend();
  }

private:
  std::vector<Constraint> constraints_;
  std::optional<BoundaryConditions> bc_{std::nullopt};
};

} // namespace RMC
