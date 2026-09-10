#include <RMC/Engine.hpp>
#include <RMC/generators/LangevinTranslationGenerator.hpp>
#include <RMC/generators/LeapfrogTranslationGenerator.hpp>
#include <RMC/generators/Removes.hpp>
#include <RMC/generators/Translations.hpp>
#include <boost/assert.hpp>
#include <ranges>
#include <utility>
#include <vector>

namespace RMC {

Engine::Engine(AtomicStructure structure, BoundaryConditions bc,
               std::uint32_t frame_rng_seed)
    : EngineBase<Engine>(std::move(bc)), selector_(RandomSelector{}),
      frame_selector_(RandomSelector{frame_rng_seed}) {
  frames_.push_back(std::move(structure)); // frame 0
}

void Engine::do_initialise() {
  BOOST_ASSERT_MSG(!frames_.empty(), "Engine: no frames");
  constraints_.set_n_frames(frames_.size());
  // Prime each frame's histogram so the averaged sum is correct from step one.
  const auto all = std::views::iota(std::size_t{0}, frames_.front().size()) |
                   std::ranges::to<std::vector>();
  for (const auto [k, frame] : frames_ | std::views::enumerate) {
    constraints_.set_active_frame(static_cast<std::size_t>(k));
    constraints_.compute_before_move(frame.coordinates, all);
  }
  constraints_.set_active_frame(0); // arbitrary; reset per step
}

template <class MakeGen>
void Engine::build_per_atom_groups(std::uint32_t seed, MakeGen make) {
  groups_.clear();
  for (const auto i : std::views::iota(std::size_t{0}, frames_.front().size())) {
    groups_.push_back(Group{.name = "atom_" + std::to_string(i),
                            .indices = {i},
                            .generator = make(seed + static_cast<std::uint32_t>(i))});
  }
}

void Engine::build_atomic_groups(double min_amp, double max_amp,
                                 std::uint32_t seed) {
  build_per_atom_groups(seed, [&](std::uint32_t s) {
    return TranslationGenerator(min_amp, max_amp, s);
  });
}

void Engine::build_langevin_groups(double step_size, std::uint32_t seed) {
  build_per_atom_groups(seed, [&](std::uint32_t s) {
    return LangevinTranslationGenerator(step_size, constraints_, s);
  });
}

void Engine::build_leapfrog_groups(double step_size, int n_steps,
                                   std::uint32_t seed) {
  build_per_atom_groups(seed, [&](std::uint32_t s) {
    return LeapfrogTranslationGenerator(n_steps, step_size, constraints_, s);
  });
}

void Engine::add_removal_group(std::string name,
                               std::vector<std::size_t> indices) {
  add_group(Group{.name = std::move(name),
                  .indices = std::move(indices),
                  .generator = RemoveGenerator{col_.collector()}});
}

void Engine::set_checkpoint(std::filesystem::path path, std::uint64_t every) {
  ckpt_.configure(std::move(path), every);
}

} // namespace RMC
