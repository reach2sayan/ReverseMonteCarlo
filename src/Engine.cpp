#include <RMC/Engine.hpp>
#include <RMC/generators/LangevinTranslationGenerator.hpp>
#include <RMC/generators/LeapfrogTranslationGenerator.hpp>
#include <RMC/generators/Removes.hpp>
#include <RMC/generators/Translations.hpp>
#include <RMC/selectors/RandomSelector.hpp>
#include <boost/assert.hpp>
#include <numeric>
#include <ranges>
#include <utility>
#include <vector>

namespace RMC {

Engine::Engine(AtomicStructure structure, BoundaryConditions bc,
               std::uint32_t frame_rng_seed)
    : EngineBase<Engine>(std::move(bc)), selector_(RandomSelector{}),
      frame_selector_(RandomSelector{frame_rng_seed}) {
  store_.add(std::move(structure)); // frame 0
}

void Engine::do_initialise() {
  BOOST_ASSERT_MSG(store_.size() > 0, "Engine: no frames");
  const std::size_t n_frames = store_.size();
  constraints_.set_n_frames(n_frames);

  // Prime each frame's histogram so the averaged sum is correct from step one.
  const std::size_t n_atoms = store_[0].size();
  std::vector<std::size_t> all_idx(n_atoms);
  std::iota(all_idx.begin(), all_idx.end(), std::size_t{0});

  for (const auto k : std::views::iota(std::size_t{0}, n_frames)) {
    constraints_.set_active_frame(k);
    constraints_.compute_before_move(store_[k].coordinates, all_idx);
  }
  constraints_.set_active_frame(0); // arbitrary; reset per step
}

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

void Engine::build_langevin_groups(double step_size, std::uint32_t seed) {
  groups_.clear();
  for (auto i : std::views::iota(std::size_t{0}, store_.primary().size())) {
    LangevinTranslationGenerator gen(step_size, constraints_,
                                     seed + static_cast<std::uint32_t>(i));
    groups_.emplace_back(Group{.name = "atom_" + std::to_string(i),
                               .indices = {i},
                               .generator = std::move(gen)});
  }
}

void Engine::build_leapfrog_groups(double step_size, int n_steps,
                                   std::uint32_t seed) {
  groups_.clear();
  for (auto i : std::views::iota(std::size_t{0}, store_.primary().size())) {
    LeapfrogTranslationGenerator gen(n_steps, step_size, constraints_,
                                     seed + static_cast<std::uint32_t>(i));
    groups_.emplace_back(Group{.name = "atom_" + std::to_string(i),
                               .indices = {i},
                               .generator = std::move(gen)});
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
