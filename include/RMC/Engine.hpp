#pragma once
#include <RMC/EngineBase.hpp>
#include <RMC/core/AtomsCollector.hpp>
#include <RMC/io/Checkpoint.hpp>
#include <RMC/selectors/GroupSelector.hpp>

#include <boost/log/trivial.hpp>

#include <cstdint>
#include <filesystem>
#include <optional>

namespace RMC {

class Engine : public EngineBase<Engine> {
public:
  explicit Engine(AtomicStructure structure, BoundaryConditions bc);
  void build_atomic_groups(double min_amp = 0.0, double max_amp = 0.2,
                           std::uint32_t seed = 42);

  constexpr void set_selector(GroupSelector s) { selector_ = std::move(s); }

  // Optional: save a checkpoint every `every` accepted steps.
  void set_checkpoint(std::filesystem::path path, std::uint64_t every = 5000);

  [[nodiscard]] constexpr const AtomicStructure &structure() const noexcept {
    return structure_;
  }
  [[nodiscard]] constexpr AtomicStructure &structure() noexcept {
    return structure_;
  }
  [[nodiscard]] constexpr io::EngineStats stats() const noexcept {
    return io::EngineStats{.steps_total = n_steps_total_,
                           .steps_accepted = n_steps_accepted_,
                           .steps_tried = n_steps_tried_,
                           .last_total_err = constraints_.total_error()};
  }

private:
  friend class EngineBase<Engine>;

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
    apply_pbc_to(structure_, c.group->span());
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

  AtomicStructure structure_;
  GroupSelector selector_;
  AtomsCollector collector_;

  // Checkpoint state
  std::optional<std::filesystem::path> checkpoint_path_;
  std::uint64_t checkpoint_every_{5000};
};

} // namespace RMC
