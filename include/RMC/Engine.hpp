#pragma once
#include <RMC/constraints/ConstraintCollection.hpp>
#include <RMC/core/AtomsCollector.hpp>
#include <RMC/core/BoundaryConditions.hpp>
#include <RMC/core/Group.hpp>
#include <RMC/core/Structure.hpp>
#include <RMC/core/Types.hpp>
#include <RMC/io/Checkpoint.hpp>
#include <RMC/selectors/GroupSelector.hpp>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/count.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/log/trivial.hpp>

#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <vector>

namespace RMC {

using StepCallback =
    std::function<void(std::uint64_t, std::uint64_t, std::uint64_t, double)>;

// Lifts void(T&) into optional<T>(T) for and_then chaining.
// T is deduced at the call site by optional::and_then — never named here.
namespace {
constexpr auto stage(auto &&fn) {
  return [fn = std::forward<decltype(fn)>(fn)](
             auto c) -> std::optional<decltype(c)> {
    std::invoke(fn, c);
    return c;
  };
}
} // namespace

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

  constexpr void run(std::uint64_t n_steps) {
    BOOST_ASSERT_MSG(!groups_.empty(), "Engine::run: no groups defined");
    for (std::uint64_t i = 0; i < n_steps; ++i) {
      step();
    }
  }
  constexpr void run_until(double target_chi2, std::uint64_t max_steps = 0);

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
  struct TrialCtx {
    std::size_t gi;
    Group *group;
  };

  // Pipeline stages — each takes/returns TrialCtx through and_then.
  constexpr std::optional<TrialCtx> select_group() {
    const std::size_t gi = selector_.select(groups_.size());
    Group &g = groups_[gi];
    if (!g.refine || g.empty() || !g.generator) {
      return std::nullopt;
    }
    return TrialCtx{gi, &g};
  }
  constexpr void snapshot_and_score_before(TrialCtx &c) {
    ++n_steps_tried_;
    structure_.save_snapshot(c.group->span());
    if (c.group->generator->modifies_species())
      structure_.save_species_snapshot();
    constraints_.compute_before_move(structure_.coordinates, c.group->span());
  }
  constexpr void propose_move(TrialCtx &c) {
    c.group->generator->generate(structure_.coordinates, c.group->span());
    apply_pbc(c.group->span());
  }
  constexpr void score_after(TrialCtx &c) {
    constraints_.compute_after_move(structure_.coordinates, c.group->span());
  }

  constexpr void settle(TrialCtx &c) {
    // Let gradient-based generators (HMC/leapfrog) supply their own
    // accept/reject; fall back to the standard Metropolis criterion from
    // constraints otherwise.
    bool rejected;
    if (auto override_rej = c.group->generator->rejection_override()) {
      rejected = *override_rej;
    } else {
      rejected = constraints_.should_reject();
    }
    if (!rejected && !collector_.pending().empty()) {
      collector_.commit_removal();
    } else {
      collector_.rollback_removal();
    }
    if (rejected) {
      structure_.restore_snapshot(c.group->span());
      structure_.restore_species_snapshot();
      constraints_.reject();
    } else {
      constraints_.accept();
      ++n_steps_accepted_;
    }
    selector_.feedback(c.gi, !rejected);
  }
  constexpr void maybe_log() {
    if (step_cb_ && (n_steps_total_ % log_every_ == 0)) {
      step_cb_(n_steps_total_, n_steps_accepted_, n_steps_tried_,
               constraints_.total_error());
    }
  }
  constexpr void maybe_checkpoint() {
    if (checkpoint_path_ && n_steps_accepted_ > 0 &&
        n_steps_accepted_ % checkpoint_every_ == 0) {
      if (auto r = io::save_checkpoint(structure_, stats(), *checkpoint_path_);
          !r) {
        BOOST_LOG_TRIVIAL(warning) << "Checkpoint save failed";
      }
    }
  }

  constexpr void step() {
    ++n_steps_total_;
    select_group()
        .and_then(stage([&](TrialCtx &c) { snapshot_and_score_before(c); }))
        .and_then(stage([&](TrialCtx &c) { propose_move(c); }))
        .and_then(stage([&](TrialCtx &c) { score_after(c); }))
        .and_then(stage([&](TrialCtx &c) { settle(c); }));
    maybe_log();
    maybe_checkpoint();
  }

  constexpr void apply_pbc(std::span<const std::size_t> moved) {
    std::ranges::for_each(moved, [&](auto i) {
      vec3_t r = structure_.coordinates.row(i).transpose();
      r = bc_wrap(bc_, r);
      structure_.coordinates.row(i) = r.transpose();
    });
  }

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

} // namespace RMC
