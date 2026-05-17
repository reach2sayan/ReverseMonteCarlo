#include <boost/log/trivial.hpp>
#include <fullrmc/Engine.hpp>
#include <fullrmc/generators/Translations.hpp>
#include <fullrmc/selectors/RandomSelector.hpp>

namespace fullrmc {

Engine::Engine(AtomicStructure structure, BoundaryConditions bc)
    : structure_(std::move(structure)), bc_(std::move(bc)),
      selector_(RandomSelector{}) {}

void Engine::build_atomic_groups(double min_amp, double max_amp,
                                 std::uint32_t seed) {
  groups_.clear();
  for (std::size_t i = 0; i < structure_.size(); ++i) {
    auto default_generator = TranslationGenerator(
        min_amp, max_amp, seed + static_cast<std::uint32_t>(i));
    groups_.emplace_back(Group{.name = "atom_" + std::to_string(i),
                               .indices = {static_cast<std::size_t>(i)},
                               .generator = std::move(default_generator)});
  }
}

void Engine::set_checkpoint(std::filesystem::path path, std::uint64_t every) {
  checkpoint_path_ = std::move(path);
  checkpoint_every_ = every;
}

void Engine::set_step_callback(StepCallback cb, std::uint64_t log_every) {
  step_cb_ = std::move(cb);
  log_every_ = log_every;
}

void Engine::run(std::uint64_t n_steps) {
  BOOST_ASSERT_MSG(!groups_.empty(), "Engine::run: no groups defined");
  for (std::uint64_t i = 0; i < n_steps; ++i) {
    step();
  }
}

void Engine::run_until(double target_chi2, std::uint64_t max_steps) {
  BOOST_ASSERT_MSG(!groups_.empty(), "Engine::run_until: no groups defined");
  std::uint64_t s = 0;
  while (constraints_.total_error() > target_chi2) {
    step();
    ++s;
    if (max_steps > 0 && s >= max_steps)
      break;
  }
}

// Lifts void(T&) into optional<T>(T) for and_then chaining.
// T is deduced at the call site by optional::and_then — never named here.
namespace {
template <typename F>
auto stage(F &&fn) {
  return [fn = std::forward<F>(fn)](auto c) -> std::optional<decltype(c)> {
    std::invoke(fn, c);
    return c;
  };
}
} // namespace

std::optional<Engine::TrialCtx> Engine::select_group() {
  const std::size_t gi = selector_.select(groups_.size());
  Group &g = groups_[gi];
  if (!g.refine || g.empty() || !g.generator)
    return std::nullopt;
  return TrialCtx{gi, &g};
}

void Engine::snapshot_and_score_before(TrialCtx &c) {
  ++n_steps_tried_;
  structure_.save_snapshot(c.group->span());
  constraints_.compute_before_move(structure_.coordinates, c.group->span());
}

void Engine::propose_move(TrialCtx &c) {
  c.group->generator->generate(structure_.coordinates, c.group->span());
  apply_pbc(c.group->span());
}

void Engine::score_after(TrialCtx &c) {
  constraints_.compute_after_move(structure_.coordinates, c.group->span());
}

void Engine::settle(TrialCtx &c) {
  const bool rejected = constraints_.should_reject();
  if (!rejected && !collector_.pending().empty())
    collector_.commit_removal();
  else
    collector_.rollback_removal();
  if (rejected) {
    structure_.restore_snapshot(c.group->span());
    constraints_.reject();
  } else {
    constraints_.accept();
    ++n_steps_accepted_;
  }
  selector_.feedback(c.gi, !rejected);
}

void Engine::maybe_log() {
  if (step_cb_ && (n_steps_total_ % log_every_ == 0))
    step_cb_(n_steps_total_, n_steps_accepted_, n_steps_tried_,
             constraints_.total_error());
}

void Engine::maybe_checkpoint() {
  if (checkpoint_path_ && n_steps_accepted_ > 0 &&
      n_steps_accepted_ % checkpoint_every_ == 0)
    if (auto r = io::save_checkpoint(structure_, stats(), *checkpoint_path_); !r)
      BOOST_LOG_TRIVIAL(warning) << "Checkpoint save failed";
}

void Engine::step() {
  ++n_steps_total_;
  select_group()
      .and_then(stage([&](TrialCtx &c) { snapshot_and_score_before(c); }))
      .and_then(stage([&](TrialCtx &c) { propose_move(c); }))
      .and_then(stage([&](TrialCtx &c) { score_after(c); }))
      .and_then(stage([&](TrialCtx &c) { settle(c); }));
  maybe_log();
  maybe_checkpoint();
}

void Engine::apply_pbc(std::span<const std::size_t> moved) {
  for (auto i : moved) {
    vec3_t r = structure_.coordinates.row(i).transpose();
    r = bc_wrap(bc_, r);
    structure_.coordinates.row(i) = r.transpose();
  }
}

} // namespace fullrmc
