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

// Single-frame RMC refinement: the N=1 specialisation of the shared engine
// pipeline (EngineBase), with the species-snapshot, adaptive-feedback,
// pending-removal, best-ever-tracking and checkpoint feature policies switched
// ON. The pipeline itself lives in EngineBase; this class supplies only the
// frame storage, selector, policy members and the per-engine customization
// points the base reaches through CRTP.
class Engine : public EngineBase<Engine> {
public:
  explicit Engine(AtomicStructure structure, BoundaryConditions bc);

  // Move-only. The structure lives on the heap at a STABLE address, so moving an
  // Engine (e.g. into the ensemble's vector) leaves any references held by its
  // constraints / move generators — e.g. SQS's ClusterCorrelationConstraint and
  // SpeciesSwapGenerator — pointing at the same live structure. Copying would
  // alias the source's structure, so it is deleted.
  Engine(const Engine &) = delete;
  Engine &operator=(const Engine &) = delete;
  Engine(Engine &&) = default;
  Engine &operator=(Engine &&) = default;

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
    const std::size_t gi = selector_.select(groups_.size());
    Group &g = groups_[gi];
    if (!g.refine || g.empty() || !g.generator) {
      return std::nullopt;
    }
    return TrialCtx{0, gi, &store_.primary(), &g};
  }
  // Single-frame pair constraints self-prime their histogram on the first
  // compute_before_move, so no per-frame priming is needed here.
  constexpr void do_initialise() noexcept {}

  [[nodiscard]] constexpr SingleFrameStore &store() noexcept { return store_; }
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

  SingleFrameStore store_;
  GroupSelector selector_;
  [[no_unique_address]] WithSpecies sp_;
  [[no_unique_address]] WithFeedback fb_;
  [[no_unique_address]] WithCollector col_;
  WithBestTracking best_;
  WithCheckpoint ckpt_;
};

} // namespace RMC
