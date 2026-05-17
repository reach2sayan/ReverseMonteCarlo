#include <algorithm>
#include <boost/log/trivial.hpp>
#include <fullrmc/Engine.hpp>
#include <fullrmc/generators/Translations.hpp>
#include <fullrmc/selectors/RandomSelector.hpp>
#include <stdexcept>

namespace fullrmc {

Engine::Engine(AtomicStructure structure, BoundaryConditions bc)
    : structure_(std::move(structure)), bc_(std::move(bc)) {
  selector_ = std::make_unique<RandomSelector>();
}

void Engine::add_group(Group g) { groups_.push_back(std::move(g)); }

void Engine::build_atomic_groups(double min_amp, double max_amp,
                                 std::uint32_t seed) {
  groups_.clear();
  const std::size_t N = structure_.size();
  for (std::size_t i = 0; i < N; ++i) {
    Group g;
    g.name = "atom_" + std::to_string(i);
    g.indices = {static_cast<std::size_t>(i)};
    g.generator = std::make_shared<TranslationGenerator>(
        min_amp, max_amp, seed + static_cast<std::uint32_t>(i));
    groups_.push_back(std::move(g));
  }
}

void Engine::set_selector(std::unique_ptr<IGroupSelector> s) {
  selector_ = std::move(s);
}

void Engine::add_constraint(std::unique_ptr<IConstraint> c) {
  constraints_.add(std::move(c));
}

void Engine::set_checkpoint(std::filesystem::path path, std::uint64_t every) {
  checkpoint_path_ = std::move(path);
  checkpoint_every_ = every;
}

void Engine::set_step_callback(StepCallback cb, std::uint64_t log_every) {
  step_cb_ = std::move(cb);
  log_every_ = log_every;
}

io::EngineStats Engine::stats() const noexcept {
  return {n_steps_total_, n_steps_accepted_, n_steps_tried_,
          constraints_.total_error()};
}

void Engine::run(std::uint64_t n_steps) {
  if (groups_.empty())
    throw std::runtime_error("Engine::run: no groups defined");
  if (!selector_)
    selector_ = std::make_unique<RandomSelector>();

  for (std::uint64_t i = 0; i < n_steps; ++i)
    step();
}

void Engine::run_until(double target_chi2, std::uint64_t max_steps) {
  if (groups_.empty())
    throw std::runtime_error("Engine::run_until: no groups defined");

  std::uint64_t s = 0;
  while (constraints_.total_error() > target_chi2) {
    step();
    ++s;
    if (max_steps > 0 && s >= max_steps)
      break;
  }
}

void Engine::step() {
  ++n_steps_total_;

  // 1. Select group.
  const std::size_t gi = selector_->select(groups_.size());
  Group &g = groups_[gi];
  if (!g.refine || g.empty())
    return;
  if (!g.generator)
    return;

  ++n_steps_tried_;

  // 2. Snapshot positions of the moving atoms.
  structure_.save_snapshot(g.span());

  // 3. Evaluate constraints before the move.
  constraints_.compute_before_move(structure_.coordinates, g.span());

  // 4. Apply the generator.
  g.generator->generate(structure_.coordinates, g.span());

  // 5. Wrap back into the simulation box.
  apply_pbc(g.span());

  // 6. Evaluate constraints after the move.
  constraints_.compute_after_move(structure_.coordinates, g.span());

  // 7. Accept / reject.
  bool rejected = constraints_.should_reject();

  // Handle staged atom removals.
  if (!rejected && !collector_.pending().empty())
    collector_.commit_removal();
  else
    collector_.rollback_removal();

  if (rejected) {
    structure_.restore_snapshot(g.span());
    constraints_.reject();
  } else {
    constraints_.accept();
    ++n_steps_accepted_;
  }

  // 8. Inform selector about outcome.
  selector_->feedback(gi, !rejected);

  // 9. Logging.
  if (step_cb_ && (n_steps_total_ % log_every_ == 0))
    step_cb_(n_steps_total_, n_steps_accepted_, n_steps_tried_,
             constraints_.total_error());

  // 10. Checkpoint.
  if (checkpoint_path_ && n_steps_accepted_ > 0 &&
      n_steps_accepted_ % checkpoint_every_ == 0) {
    if (auto result = io::save_checkpoint(structure_, stats(), *checkpoint_path_);
        !result)
      BOOST_LOG_TRIVIAL(warning) << "Checkpoint save failed";
  }
}

void Engine::apply_pbc(std::span<const std::size_t> moved) {
  for (auto i : moved) {
    vec3_t r = structure_.coordinates.row(i).transpose();
    r = bc_wrap(bc_, r);
    structure_.coordinates.row(i) = r.transpose();
  }
}

} // namespace fullrmc
