#include <RMC/Engine.hpp>
#include <RMC/generators/Removes.hpp>
#include <RMC/generators/Translations.hpp>
#include <RMC/selectors/RandomSelector.hpp>
#include <boost/log/trivial.hpp>
#include <ranges>
#include <utility>

namespace RMC {

Engine::Engine(AtomicStructure structure, BoundaryConditions bc)
    : EngineBase<Engine>(std::move(bc)), store_(std::move(structure)),
      selector_(RandomSelector{}) {}

void Engine::build_atomic_groups(double min_amp, double max_amp,
                                 std::uint32_t seed) {
  groups_.clear();
  for (auto i : std::views::iota(std::size_t{0}, store_.primary().size())) {
    auto default_generator = TranslationGenerator(
        min_amp, max_amp, seed + static_cast<std::uint32_t>(i));
    groups_.emplace_back(Group{.name = "atom_" + std::to_string(i),
                               .indices = {i},
                               .generator = std::move(default_generator)});
  }
}

void Engine::add_removal_group(std::string name,
                               std::vector<std::size_t> indices) {
  add_group(Group{.name = std::move(name),
                  .indices = std::move(indices),
                  .generator = RemoveGenerator{col_.shared_collector()}});
}

void Engine::set_checkpoint(std::filesystem::path path, std::uint64_t every) {
  ckpt_.configure(std::move(path), every);
}

} // namespace RMC
