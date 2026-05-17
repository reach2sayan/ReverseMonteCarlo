#pragma once
#include <fullrmc/constraints/ConstraintCollection.hpp>
#include <fullrmc/core/AtomsCollector.hpp>
#include <fullrmc/core/BoundaryConditions.hpp>
#include <fullrmc/core/Group.hpp>
#include <fullrmc/core/Structure.hpp>
#include <fullrmc/core/Types.hpp>
#include <fullrmc/io/Checkpoint.hpp>
#include <fullrmc/selectors/GroupSelector.hpp>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/count.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/log/trivial.hpp>

#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <vector>

namespace fullrmc {

using StepCallback =
    std::function<void(std::uint64_t, std::uint64_t, std::uint64_t, double)>;

class Engine {
public:
  explicit Engine(AtomicStructure structure, BoundaryConditions bc);
  constexpr void add_group(Group g) { groups_.push_back(std::move(g)); }
  void build_atomic_groups(double min_amp = 0.0, double max_amp = 0.2,
                           std::uint32_t seed = 42);

  constexpr void set_selector(IGroupSelector s) { selector_ = std::move(s); }
  constexpr void add_constraint(IConstraint c) {
    c.set_boundary_conditions(bc_);
    constraints_.add(std::move(c));
  }

  // Optional: save a checkpoint every `every` accepted steps.
  void set_checkpoint(std::filesystem::path path, std::uint64_t every = 5000);
  void set_step_callback(StepCallback cb, std::uint64_t log_every = 1000);

  void run(std::uint64_t n_steps);
  void run_until(double target_chi2, std::uint64_t max_steps = 0);

  [[nodiscard]] constexpr const AtomicStructure &structure() const noexcept {
    return structure_;
  }
  [[nodiscard]] constexpr AtomicStructure &structure() noexcept {
    return structure_;
  }
  [[nodiscard]] constexpr const BoundaryConditions &boundary() const noexcept {
    return bc_;
  }
  [[nodiscard]] constexpr io::EngineStats stats() const noexcept {
    return io::EngineStats{.steps_total = n_steps_total_,
                           .steps_accepted = n_steps_accepted_,
                           .steps_tried = n_steps_tried_,
                           .last_total_err = constraints_.total_error()};
  }
  [[nodiscard]] constexpr ConstraintCollection &constraints() noexcept {
    return constraints_;
  }

private:
  // Execute a single MC trial.
  void step();

  // Apply periodic wrapping to the atoms that just moved.
  void apply_pbc(std::span<const std::size_t> moved);

  AtomicStructure structure_;
  BoundaryConditions bc_;
  std::vector<Group> groups_;
  IGroupSelector selector_;
  ConstraintCollection constraints_;
  AtomsCollector collector_;

  // Statistics
  std::uint64_t n_steps_total_{0};
  std::uint64_t n_steps_tried_{0};
  std::uint64_t n_steps_accepted_{0};

  // Checkpoint state
  std::optional<std::filesystem::path> checkpoint_path_;
  std::uint64_t checkpoint_every_{5000};

  // Logging callback
  StepCallback step_cb_;
  std::uint64_t log_every_{1000};
};

} // namespace fullrmc
