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

constexpr void Engine::run_until(double target_chi2, std::uint64_t max_steps) {
  BOOST_ASSERT_MSG(!groups_.empty(), "Engine::run_until: no groups defined");
  std::uint64_t s = 0;
  while (constraints_.total_error() > target_chi2) {
    step();
    ++s;
    if (max_steps > 0 && s >= max_steps)
      break;
  }
}

} // namespace fullrmc
