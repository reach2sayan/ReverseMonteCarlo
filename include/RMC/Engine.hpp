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
//
// Method references:
//   [1] R. L. McGreevy and L. Pusztai, "Reverse Monte Carlo Simulation: A New
//       Technique for the Determination of Disordered Structures,"
//       Mol. Simul. 1, 359-367 (1988). doi:10.1080/08927028808080958
//   [2] R. L. McGreevy, "Reverse Monte Carlo modelling,"
//       J. Phys.: Condens. Matter 13, R877-R913 (2001).
//       doi:10.1088/0953-8984/13/46/201
//
// See also, as an independent reference implementation of the method:
//   [3] B. Aoun, "Fullrmc, a Rigid Body Reverse Monte Carlo Modeling Package
//       Enabled with Machine Learning and Artificial Intelligence,"
//       J. Comput. Chem. 37, 1102-1111 (2016). doi:10.1002/jcc.24304
//
// This is an independent C++ implementation of the published RMC method [1,2].

// The RMC refinement engine. One or more structural frames are refined against
// the experimental data; with a single frame this is ordinary RMC, with N
// frames they are refined simultaneously against the AVERAGED computed profile
// (the averaging lives inside the pair/angle constraints). Add extra frames with
// add_frame(); each step selects one frame, then one group within it.
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

  // Rebuild the per-atom move groups with gradient-driven proposals that steer
  // each atom along −∇χ² of the attached constraints (Langevin/MALA, or HMC via
  // leapfrog). Call AFTER add_constraint and AFTER the engine is in its final
  // location: the generators hold a pointer to this engine's constraint
  // collection, which moving the engine would invalidate.
  void build_langevin_groups(double step_size, std::uint32_t seed = 42);
  void build_leapfrog_groups(double step_size, int n_steps = 10,
                             std::uint32_t seed = 42);

  // Add a move group whose proposal removes `indices` from the system. The
  // RemoveGenerator is bound to this engine's collector, so the staged removal
  // is the one settle() commits/rolls back and the constraints skip.
  void add_removal_group(std::string name, std::vector<std::size_t> indices);

  // The engine's shared removal collector. Bind your own RemoveGenerator to it
  // (RemoveGenerator{engine.collector()}) when building groups by hand; the
  // staged removals must land in this collector to be committed and skipped.
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
    // Single-frame: skip the frame draw and fix fi = 0 so the group RNG stream
    // (and thus single-frame behaviour) is identical to the pre-multi-frame
    // engine.
    const std::size_t fi =
        store_.size() == 1 ? std::size_t{0} : frame_selector_.select(store_.size());
    const std::size_t gi = selector_.select(groups_.size());
    Group &g = groups_[gi];
    if (!g.refine || g.empty() || !g.generator) {
      return std::nullopt;
    }
    return TrialCtx{fi, gi, &store_[fi], &g};
  }
  // Inform the constraints of the frame count and prime each frame's histogram
  // so the averaged profile is correct from the first step. Run once via
  // EngineBase::ensure_initialised(); defined in Engine.cpp.
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
