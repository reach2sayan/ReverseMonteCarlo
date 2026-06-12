#pragma once
#include <RMC/constraints/Constraint.hpp>
#include <boost/stl_interfaces/view_interface.hpp>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

namespace RMC {

class ConstraintCollection
    : public boost::stl_interfaces::view_interface<
          ConstraintCollection,
          boost::stl_interfaces::element_layout::contiguous> {
public:
  // Insert in ascending computation_cost() order so cheap constraints
  // (bonds, angles) run before expensive O(N²) ones (PDF, S(Q)).
  void add(Constraint c);
  void set_boundary_conditions(const BoundaryConditions &bc) noexcept;
  void set_collector(const AtomsCollector *c) noexcept;
  void compute_before_move(const coords_t &coords,
                           std::span<const std::size_t> moved);

  // Run constraints cheapest-first. A RIGID constraint that worsens is a hard
  // gate, so once one rejects the remaining (often expensive) constraints are
  // skipped — their reject() resets err_after_ to err_before_ so total_error()
  // stays consistent. SOFT constraints never short-circuit: they are always
  // fully evaluated so the engine's Sampler sees the exact post-move total.
  void compute_after_move(const coords_t &coords,
                          std::span<const std::size_t> moved);

  // Hard gate for the engine: any RIGID constraint that worsened. Soft
  // constraints are deferred to the Sampler via total_error[_before]().
  [[nodiscard]] bool rigid_should_reject() const noexcept;
  void accept() noexcept;
  void reject() noexcept;

  [[nodiscard]] double total_error() const noexcept;
  // Soft total error BEFORE the proposed move (rigid constraints contribute 0).
  // Paired with total_error() to form the ΔE handed to the Sampler.
  [[nodiscard]] double total_error_before() const noexcept;

  [[nodiscard]] std::vector<std::pair<std::string_view, double>>
  error_breakdown() const;

  void initialise_all();

  void set_n_frames(std::size_t n) noexcept;
  void set_active_frame(std::size_t k) noexcept;
  [[nodiscard]] constexpr auto begin() const noexcept {
    return constraints_.cbegin();
  }
  [[nodiscard]] constexpr auto end() const noexcept {
    return constraints_.cend();
  }

private:
  std::vector<Constraint> constraints_;
  std::optional<BoundaryConditions> bc_{std::nullopt};
  const AtomsCollector *collector_{nullptr};
};

} // namespace RMC
