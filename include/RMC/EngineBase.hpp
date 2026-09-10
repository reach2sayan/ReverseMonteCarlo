#pragma once
#include <RMC/EnginePolicies.hpp>
#include <RMC/constraints/ConstraintCollection.hpp>
#include <RMC/core/BoundaryConditions.hpp>
#include <RMC/core/Group.hpp>
#include <RMC/core/RngGenerator.hpp>
#include <RMC/core/Structure.hpp>
#include <RMC/core/Types.hpp>
#include <RMC/io/Checkpoint.hpp>
#include <RMC/sampling/Sampler.hpp>

#include <boost/assert.hpp>

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <vector>

namespace RMC {

using StepCallback = std::function<void(std::uint64_t, std::uint64_t,
                                        std::uint64_t, double,
                                        const AtomicStructure &)>;

// CRTP base for the refinement engine. step() runs one trial (select, snapshot,
// score before, propose, score after, settle) with three-tier acceptance
// (decide_rejection); per-engine specifics come from CRTP customization points:
//   select()                -> std::optional<TrialCtx>
//   store()                 -> indexed frame storage (frame 0 = primary)
//   species_policy() / feedback_policy() / collector_policy() /
//   checkpoint_policy()     -> compile-time feature policies
//   group_sel_for_feedback()-> selector receiving adaptive feedback
//   do_initialise()         -> one-time frame/constraint priming
// Derived must `friend class EngineBase<Derived>;`.
template <typename Derived> class EngineBase {
public:
  // One trial's frame and group (single-frame fixes fi == 0).
  struct TrialCtx {
    std::size_t fi;
    std::size_t gi;
    AtomicStructure *frame;
    Group *group;
  };

  void add_group(Group g) { groups_.push_back(std::move(g)); }
  void add_constraint(Constraint c) {
    c.set_boundary_conditions(*bc_);
    constraints_.add(std::move(c));
  }

  void initialise() { ensure_initialised(); }

  void run(std::uint64_t n_steps) {
    BOOST_ASSERT_MSG(!groups_.empty(), "no groups defined");
    ensure_initialised();
    for ([[maybe_unused]] auto _ : std::views::iota(std::uint64_t{0}, n_steps)) {
      self().step();
    }
  }
  void run_until(double target_chi2, std::uint64_t max_steps = 0) {
    BOOST_ASSERT_MSG(!groups_.empty(), "no groups defined");
    ensure_initialised();
    for (std::uint64_t s = 0; constraints_.total_error() > target_chi2 &&
                              (max_steps == 0 || s < max_steps);
         ++s) {
      self().step();
    }
  }

  [[nodiscard]] constexpr ConstraintCollection &constraints() noexcept {
    return constraints_;
  }
  [[nodiscard]] const BoundaryConditions &boundary() const noexcept {
    return *bc_;
  }
  [[nodiscard]] double total_error() const noexcept {
    return constraints_.total_error();
  }
  [[nodiscard]] constexpr std::uint64_t steps_total() const noexcept {
    return n_steps_total_;
  }
  [[nodiscard]] constexpr std::uint64_t steps_accepted() const noexcept {
    return n_steps_accepted_;
  }
  [[nodiscard]] io::EngineStats make_stats() const noexcept {
    return io::EngineStats{.steps_total = n_steps_total_,
                           .steps_accepted = n_steps_accepted_,
                           .steps_tried = n_steps_tried_,
                           .last_total_err = constraints_.total_error()};
  }

  void set_step_callback(StepCallback cb, std::uint64_t log_every = 1000) {
    step_cb_ = std::move(cb);
    log_every_ = log_every;
  }

  // Replace the move-acceptance policy (default GreedySampler, strict downhill).
  // `seed` seeds the RNG for Metropolis/annealing uniforms; give each replica a
  // distinct seed for independent streams.
  void set_sampler(Sampler s, std::uint32_t seed = 0xACCE55u) {
    sampler_ = std::move(s);
    accept_rng_ = Rng{seed};
  }

protected:
  explicit EngineBase(BoundaryConditions bc)
      : bc_(std::make_unique<BoundaryConditions>(std::move(bc))) {}

  // One step: a trial when select() finds an eligible group, then best-state
  // tracking, logging and checkpointing.
  void step() {
    ++n_steps_total_;
    const std::optional<TrialCtx> ctx = self().select();
    if (ctx) {
      trial(*ctx);
    }
    const AtomicStructure &cur = ctx ? *ctx->frame : self().store().front();
    self().best_policy().update(cur, constraints_.total_error());
    maybe_log(cur);
    self().checkpoint_policy().maybe(self().store().front(), make_stats(),
                                     n_steps_accepted_);
  }

  void trial(const TrialCtx &c) {
    ++n_steps_tried_;
    const auto idx = c.group->span();
    constraints_.set_active_frame(c.fi); // no-op for non-frame constraints
    c.frame->save_snapshot(idx);
    self().species_policy().save_before(*c.frame, *c.group);
    constraints_.compute_before_move(c.frame->coordinates, idx);
    c.group->generator->generate(c.frame->coordinates, idx);
    for (const auto i : idx) {
      c.frame->coordinates.row(i) =
          bc_->wrap(c.frame->coordinates.row(i).transpose()).transpose();
    }
    constraints_.compute_after_move(c.frame->coordinates, idx);
    settle(c);
  }

  [[nodiscard]] bool decide_rejection(const TrialCtx &c) {
    // Gradient-based generators (HMC/leapfrog) own their accept/reject.
    if (auto override_rej = c.group->generator->rejection_override()) {
      return *override_rej;
    }
    // Otherwise any RIGID constraint is a hard rejection...
    if (constraints_.rigid_should_reject()) {
      return true;
    }
    // ...and the Sampler decides the rest.
    return !sampler_.accept(constraints_.total_error_before(),
                            constraints_.total_error(), n_steps_total_,
                            accept_rng_.uniform());
  }

  void settle(const TrialCtx &c) {
    const bool rejected = decide_rejection(c);
    self().collector_policy().commit_or_rollback(!rejected);
    if (rejected) {
      c.frame->restore_snapshot(c.group->span());
      self().species_policy().restore_on_reject(*c.frame);
      constraints_.reject();
    } else {
      constraints_.accept();
      ++n_steps_accepted_;
    }
    self().feedback_policy().feedback(self().group_sel_for_feedback(), c.gi,
                                      !rejected);
  }

  void ensure_initialised() {
    if (initialised_) {
      return;
    }
    constraints_.set_collector(self().collector_policy().collector());
    constraints_.initialise_all();
    self().do_initialise();
    initialised_ = true;
  }

  void maybe_log(const AtomicStructure &current) {
    if (step_cb_ && (n_steps_total_ % log_every_ == 0)) {
      std::invoke(step_cb_, n_steps_total_, n_steps_accepted_, n_steps_tried_,
                  constraints_.total_error(), current);
    }
  }

  // Heap-stable so the raw `const BoundaryConditions*` each constraint holds
  // survives the engine being moved (e.g. into run_ensemble's vector).
  std::unique_ptr<BoundaryConditions> bc_;
  std::vector<Group> groups_;
  ConstraintCollection constraints_;
  Sampler sampler_{GreedySampler{}};
  Rng accept_rng_{0xACCE55u};
  std::uint64_t n_steps_total_{0};
  std::uint64_t n_steps_tried_{0};
  std::uint64_t n_steps_accepted_{0};
  StepCallback step_cb_;
  std::uint64_t log_every_{1000};
  bool initialised_{false};

private:
  constexpr Derived &self() { return static_cast<Derived &>(*this); }
};

} // namespace RMC
