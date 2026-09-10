#pragma once
#include <RMC/EngineBase.hpp>
#include <RMC/FrameStore.hpp>
#include <RMC/selectors/GroupSelector.hpp>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace RMC {

// Reverse Monte Carlo (RMC) structural refinement.
//   [1] McGreevy & Pusztai, Mol. Simul. 1, 359-367 (1988). doi:10.1080/08927028808080958
//   [2] McGreevy, J. Phys.: Condens. Matter 13, R877-R913 (2001). doi:10.1088/0953-8984/13/46/201
//   [3] Aoun, "Fullrmc," J. Comput. Chem. 37, 1102-1111 (2016). doi:10.1002/jcc.24304

// The RMC refinement engine. With one frame this is ordinary RMC; with N frames
// they are refined simultaneously against the AVERAGED computed profile (averaging
// lives in the constraints). Each step selects one frame, then one group within it.
class Engine : public EngineBase<Engine> {
public:
  explicit Engine(AtomicStructure structure, BoundaryConditions bc,
                  std::uint32_t frame_rng_seed = 1729);
  Engine(const Engine &) = delete;
  Engine &operator=(const Engine &) = delete;
  Engine(Engine &&) = default;
  Engine &operator=(Engine &&) = default;

  // Add an additional structural frame (frame 0 is the ctor structure). All
  // frames are refined against the averaged computed profile.
  void add_frame(AtomicStructure s) { store_.add(std::move(s)); }
  void set_frame_selector(GroupSelector s) { frame_selector_ = std::move(s); }

  void build_atomic_groups(double min_amp = 0.0, double max_amp = 0.2,
                           std::uint32_t seed = 42);

  // Rebuild per-atom groups with gradient-driven proposals (−∇χ²; Langevin/MALA
  // or HMC leapfrog). Call AFTER add_constraint and AFTER the engine is in its
  // final location: the generators hold a pointer into constraints(), which
  // moving the engine would invalidate.
  void build_langevin_groups(double step_size, std::uint32_t seed = 42);
  void build_leapfrog_groups(double step_size, int n_steps = 10,
                             std::uint32_t seed = 42);

  // Add a move group whose proposal removes `indices`. Bound to this engine's
  // collector, so settle() commits/rolls back the staged removal.
  void add_removal_group(std::string name, std::vector<std::size_t> indices);

  // The engine's shared removal collector. Bind your own RemoveGenerator to it
  // (RemoveGenerator{engine.collector()}) when building groups by hand.
  [[nodiscard]] std::shared_ptr<AtomsCollector> collector() noexcept {
    return col_.shared_collector();
  }

  void set_selector(GroupSelector s) { selector_ = std::move(s); }

  // Optional: save a checkpoint every `every` accepted steps.
  void set_checkpoint(std::filesystem::path path, std::uint64_t every = 5000);

  [[nodiscard]] const AtomicStructure &structure() const noexcept {
    return store_.primary();
  }
  [[nodiscard]] AtomicStructure &structure() noexcept {
    return store_.primary();
  }

  // Track the lowest-error configuration seen during the run (see
  // WithBestTracking). Off by default.
  constexpr void set_track_best(bool on = true) noexcept { best_.set_track(on); }
  [[nodiscard]] double best_error() const noexcept {
    return best_.best_error(constraints_.total_error());
  }
  [[nodiscard]] const AtomicStructure &best_structure() const noexcept {
    return best_.best_structure(store_.primary());
  }

  [[nodiscard]] io::EngineStats stats() const noexcept { return make_stats(); }

private:
  friend class EngineBase<Engine>;

  // CRTP customization points called by EngineBase.
  std::optional<TrialCtx> select() {
    // Single-frame: skip the frame draw and fix fi = 0 to keep the group RNG
    // stream identical to the pre-multi-frame engine.
    const std::size_t fi =
        store_.size() == 1 ? std::size_t{0} : frame_selector_.select(store_.size());
    const std::size_t gi = selector_.select(groups_.size());
    Group &g = groups_[gi];
    if (!g.refine || g.empty() || !g.generator) {
      return std::nullopt;
    }
    return TrialCtx{fi, gi, &store_[fi], &g};
  }
  // Prime constraint frame count + each frame's histogram. Defined in Engine.cpp.
  void do_initialise();

  [[nodiscard]] constexpr FrameStore &store() noexcept { return store_; }
  [[nodiscard]] constexpr WithSpecies &species_policy() noexcept { return sp_; }
  [[nodiscard]] constexpr WithFeedback &feedback_policy() noexcept {
    return fb_;
  }
  [[nodiscard]] constexpr WithCollector &collector_policy() noexcept {
    return col_;
  }
  [[nodiscard]] constexpr WithBestTracking &best_policy() noexcept {
    return best_;
  }
  [[nodiscard]] constexpr WithCheckpoint &checkpoint_policy() noexcept {
    return ckpt_;
  }
  [[nodiscard]] constexpr GroupSelector &group_sel_for_feedback() noexcept {
    return selector_;
  }

  FrameStore store_;
  GroupSelector selector_;
  GroupSelector frame_selector_;
  [[no_unique_address]] WithSpecies sp_;
  [[no_unique_address]] WithFeedback fb_;
  [[no_unique_address]] WithCollector col_;
  WithBestTracking best_;
  WithCheckpoint ckpt_;
};

} // namespace RMC
