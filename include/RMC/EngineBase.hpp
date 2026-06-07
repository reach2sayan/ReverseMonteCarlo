#pragma once
#include <RMC/EnginePolicies.hpp>
#include <RMC/constraints/ConstraintCollection.hpp>
#include <RMC/core/BoundaryConditions.hpp>
#include <RMC/core/Group.hpp>
#include <RMC/core/RngGenerator.hpp>
#include <RMC/core/Structure.hpp>
#include <RMC/core/Types.hpp>
#include <RMC/io/Checkpoint.hpp>
#include <RMC/sampling/GreedySampler.hpp>
#include <RMC/sampling/Sampler.hpp>

#include <boost/assert.hpp>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <optional>
#include <ranges>
#include <span>
#include <vector>

namespace RMC {

using StepCallback = std::function<void(std::uint64_t, std::uint64_t,
                                        std::uint64_t, double,
                                        const AtomicStructure &)>;

// CRTP base for Engine and MultiFrameEngine.
//
// Owns the shared MC pipeline so the two engines duplicate nothing: step(),
// the four pipeline stages, the three-tier acceptance (decide_rejection) and
// settle() all live here and reach the per-engine specifics through CRTP
// customization points the Derived class supplies:
//   select()                -> std::optional<TrialCtx>  (1 or 2 selectors)
//   store()                 -> indexed frame storage (frame 0 = primary)
//   species_policy() / feedback_policy() / collector_policy() /
//   checkpoint_policy()     -> orthogonal compile-time feature policies
//   group_sel_for_feedback()-> the selector that receives adaptive feedback
//   do_initialise()         -> one-time frame/constraint priming
// Derived must `friend class EngineBase<Derived>;` to expose these.
//
// Shared state: bc_, groups_, constraints_, sampler_, RNG, stats. Helpers:
//   stage()        — lifts void(T&) into optional<T>(T) for and_then chaining
//   apply_pbc_to() — wraps atom coordinates of a structure into the unit cell
template <typename Derived> class EngineBase {
public:
  // Ephemeral per-step context threaded through the and_then pipeline. The
  // frame pointer is always populated (single-frame fixes fi == 0).
  struct TrialCtx {
    std::size_t fi;
    std::size_t gi;
    AtomicStructure *frame;
    Group *group;
  };

  void add_group(Group g) { groups_.push_back(std::move(g)); }
  void add_constraint(Constraint c) {
    c.set_boundary_conditions(bc_);
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

  // Replace the move-acceptance policy. Default is GreedySampler (strict
  // downhill — RMC's historical behaviour). `seed` seeds the dedicated RNG the
  // engine draws Metropolis/annealing uniforms from; give each ensemble replica
  // a distinct seed for independent stochastic streams.
  void set_sampler(Sampler s, std::uint32_t seed = 0xACCE55u) {
    sampler_ = std::move(s);
    accept_rng_ = RngBuffer<>{seed};
  }

protected:
  explicit EngineBase(BoundaryConditions bc) : bc_(std::move(bc)) {}

  // One trial step: select → snapshot/score-before → propose → score-after →
  // settle, then log and (policy-gated) checkpoint. select() short-circuits the
  // and_then chain when no eligible group is available.
  void step() {
    ++n_steps_total_;
    auto ctx = self().select();
    ctx.and_then(stage([&](TrialCtx &c) { snapshot_and_score_before(c); }))
        .and_then(stage([&](TrialCtx &c) { propose_move(c); }))
        .and_then(stage([&](TrialCtx &c) { score_after(c); }))
        .and_then(stage([&](TrialCtx &c) { settle(c); }));
    const AtomicStructure &cur = ctx ? *ctx->frame : self().store()[0];
    self().best_policy().update(cur, constraints_.total_error());
    maybe_log(cur);
    self().checkpoint_policy().maybe(self().store()[0], make_stats(),
                                     n_steps_accepted_);
  }

  void snapshot_and_score_before(TrialCtx &c) {
    ++n_steps_tried_;
    constraints_.set_active_frame(c.fi); // no-op for non-frame constraints
    c.frame->save_snapshot(c.group->span());
    self().species_policy().save_before(*c.frame, *c.group);
    constraints_.compute_before_move(c.frame->coordinates, c.group->span());
  }
  void propose_move(TrialCtx &c) {
    c.group->generator->generate(c.frame->coordinates, c.group->span());
    apply_pbc_to(*c.frame, c.group->span());
  }
  void score_after(TrialCtx &c) {
    constraints_.compute_after_move(c.frame->coordinates, c.group->span());
  }

  // Three-tier acceptance:
  //   1. gradient-based generators (HMC/leapfrog) own their accept/reject;
  //   2. otherwise any worsened RIGID constraint is a hard rejection;
  //   3. otherwise the Sampler decides from the soft total error — greedy
  //      (strict downhill) by default, Metropolis/annealing when configured.
  [[nodiscard]] bool decide_rejection(TrialCtx &c) {
    if (auto override_rej = c.group->generator->rejection_override()) {
      return *override_rej;
    }
    if (constraints_.rigid_should_reject()) {
      return true;
    }
    return !sampler_.accept(constraints_.total_error_before(),
                            constraints_.total_error(), n_steps_total_,
                            accept_rng_.uniform());
  }

  void settle(TrialCtx &c) {
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

  // One-time setup, run lazily on the first run()/run_until() so clients never
  // call an engine-level initialise(). Sweeps the constraints first (safety net
  // for a forgotten per-constraint initialise()), then the engine-specific
  // frame priming. Order matters: priming calls compute_error, which consumes
  // the per-constraint state initialise_all() fills.
  void ensure_initialised() {
    if (initialised_) {
      return;
    }
    constraints_.initialise_all();
    self().do_initialise();
    initialised_ = true;
  }

  // Lifts void(T&) into optional<T>(T) for and_then chaining.
  static constexpr auto stage(auto &&fn) {
    return [fn = std::forward<decltype(fn)>(fn)](
               auto c) -> std::optional<decltype(c)> {
      std::invoke(fn, c);
      return c;
    };
  }

  void apply_pbc_to(AtomicStructure &s, std::span<const std::size_t> moved) {
    std::ranges::for_each(moved, [&](auto i) {
      vec3_t r = s.coordinates.row(static_cast<Eigen::Index>(i)).transpose();
      r = bc_wrap(bc_, r);
      s.coordinates.row(static_cast<Eigen::Index>(i)) = r.transpose();
    });
  }

  void maybe_log(const AtomicStructure &current) {
    if (step_cb_ && (n_steps_total_ % log_every_ == 0)) {
      std::invoke(step_cb_, n_steps_total_, n_steps_accepted_, n_steps_tried_,
                  constraints_.total_error(), current);
    }
  }

  BoundaryConditions bc_;
  std::vector<Group> groups_;
  ConstraintCollection constraints_;
  Sampler sampler_{GreedySampler{}};
  RngBuffer<> accept_rng_{0xACCE55u};
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
