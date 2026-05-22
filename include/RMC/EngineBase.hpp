#pragma once
#include <RMC/constraints/ConstraintCollection.hpp>
#include <RMC/core/BoundaryConditions.hpp>
#include <RMC/core/Group.hpp>
#include <RMC/core/Structure.hpp>
#include <RMC/core/Types.hpp>

#include <boost/assert.hpp>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <vector>

namespace RMC {

using StepCallback = std::function<void(std::uint64_t, std::uint64_t,
                                        std::uint64_t, double,
                                        const AtomicStructure &)>;

// CRTP base for Engine and MultiFrameEngine.
// Holds shared state (bc_, groups_, constraints_, stats) and provides
// add_group, add_constraint, run, run_until, and two protected helpers:
//   stage()       — lifts void(T&) into optional<T>(T) for and_then chaining
//   apply_pbc_to() — wraps atom coordinates of a structure into the unit cell
template <typename Derived> class EngineBase {
public:
  constexpr void add_group(Group g) { groups_.push_back(std::move(g)); }
  void add_constraint(Constraint c) {
    c.set_boundary_conditions(bc_);
    constraints_.add(std::move(c));
  }

  constexpr void run(std::uint64_t n_steps) {
    BOOST_ASSERT_MSG(!groups_.empty(), "no groups defined");
    for (std::uint64_t i = 0; i < n_steps; ++i) {
      self().step();
    }
  }
  constexpr void run_until(double target_chi2, std::uint64_t max_steps = 0) {
    BOOST_ASSERT_MSG(!groups_.empty(), "no groups defined");
    std::uint64_t s = 0;
    while (constraints_.total_error() > target_chi2) {
      self().step();
      ++s;
      if (max_steps > 0 && s >= max_steps) {
        break;
      }
    }
  }

  [[nodiscard]] constexpr ConstraintCollection &constraints() noexcept {
    return constraints_;
  }
  [[nodiscard]] constexpr const BoundaryConditions &boundary() const noexcept {
    return bc_;
  }
  [[nodiscard]] constexpr double total_error() const noexcept {
    return constraints_.total_error();
  }
  [[nodiscard]] constexpr std::uint64_t steps_total() const noexcept {
    return n_steps_total_;
  }
  [[nodiscard]] constexpr std::uint64_t steps_accepted() const noexcept {
    return n_steps_accepted_;
  }

  void set_step_callback(StepCallback cb, std::uint64_t log_every = 1000) {
    step_cb_ = std::move(cb);
    log_every_ = log_every;
  }

protected:
  constexpr explicit EngineBase(BoundaryConditions bc) : bc_(std::move(bc)) {}

  // Lifts void(T&) into optional<T>(T) for and_then chaining.
  static constexpr auto stage(auto &&fn) {
    return [fn = std::forward<decltype(fn)>(fn)](
               auto c) -> std::optional<decltype(c)> {
      std::invoke(fn, c);
      return c;
    };
  }

  constexpr void apply_pbc_to(AtomicStructure &s,
                              std::span<const std::size_t> moved) {
    std::ranges::for_each(moved, [&](auto i) {
      vec3_t r = s.coordinates.row(static_cast<Eigen::Index>(i)).transpose();
      r = bc_wrap(bc_, r);
      s.coordinates.row(static_cast<Eigen::Index>(i)) = r.transpose();
    });
  }

  constexpr void maybe_log(const AtomicStructure &current) {
    if (step_cb_ && (n_steps_total_ % log_every_ == 0)) {
      std::invoke(step_cb_, n_steps_total_, n_steps_accepted_, n_steps_tried_,
                  constraints_.total_error(), current);
    }
  }

  BoundaryConditions bc_;
  std::vector<Group> groups_;
  ConstraintCollection constraints_;
  std::uint64_t n_steps_total_{0};
  std::uint64_t n_steps_tried_{0};
  std::uint64_t n_steps_accepted_{0};
  StepCallback step_cb_;
  std::uint64_t log_every_{1000};

private:
  constexpr Derived &self() { return static_cast<Derived &>(*this); }
};

} // namespace RMC
