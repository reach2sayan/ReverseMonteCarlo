#include <RMC/Engine.hpp>
#include <RMC/generators/Translations.hpp>
#include <RMC/selectors/RandomSelector.hpp>
#include <boost/log/trivial.hpp>

namespace RMC {

Engine::Engine(AtomicStructure structure, BoundaryConditions bc)
    : EngineBase<Engine>(std::move(bc)), structure_(std::move(structure)),
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

} // namespace RMC
