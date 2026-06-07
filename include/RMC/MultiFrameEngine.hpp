#pragma once
#include <RMC/EngineBase.hpp>
#include <RMC/FrameStore.hpp>
#include <RMC/selectors/GroupSelector.hpp>
#include <RMC/selectors/RandomSelector.hpp>

#include <boost/assert.hpp>

#include <cstdint>
#include <numeric>
#include <vector>

namespace RMC {

// Runs a multi-frame RMC refinement: N structural frames are refined
// simultaneously against a single experimental dataset. Each MC step selects
// one frame and one group within that frame, proposes a move, and accepts or
// rejects it based on whether the chi² of the AVERAGED computed profile
// (averaged across all N frames) improves.
//
// The averaging is handled inside PairFunctionConstraint (and its derivatives)
// via per-frame histograms. All other constraints (bonds, angles, etc.) are
// evaluated only against the moved frame.
//
// Usage:
//   MultiFrameEngine eng(bc);
//   eng.add_frame(frame0); eng.add_frame(frame1); ...
//   eng.add_group(g);
//   eng.add_constraint(pdf_constraint);
//   eng.run(100'000);   // per-frame histograms are primed automatically
//
// This is the general (N-frame) form of the shared engine pipeline
// (EngineBase). All feature policies (species/feedback/collector/best/
// checkpoint) are switched OFF; the only per-engine specifics are the two
// selectors and the frame-histogram priming in do_initialise().
class MultiFrameEngine : public EngineBase<MultiFrameEngine> {
public:
  explicit MultiFrameEngine(BoundaryConditions bc,
                            std::uint32_t frame_rng_seed = 42,
                            std::uint32_t group_rng_seed = 1729)
      : EngineBase<MultiFrameEngine>(bc),
        frame_selector_(RandomSelector{frame_rng_seed}),
        group_selector_(RandomSelector{group_rng_seed}) {}

  void add_frame(AtomicStructure s) { store_.add(std::move(s)); }

  constexpr void set_frame_selector(GroupSelector s) {
    frame_selector_ = std::move(s);
  }
  constexpr void set_group_selector(GroupSelector s) {
    group_selector_ = std::move(s);
  }

  [[nodiscard]] constexpr const AtomicStructure &best_frame() const {
    // Returns frame closest to lowest total error (all frames contribute
    // equally to the averaged constraint, so we just return frame 0; callers
    // who need the absolute best should compare external metrics).
    return store_[0];
  }

private:
  friend class EngineBase<MultiFrameEngine>;

  // CRTP customization points called by EngineBase.
  std::optional<TrialCtx> select() {
    const std::size_t fi = frame_selector_.select(store_.size());
    const std::size_t gi = group_selector_.select(groups_.size());
    Group &g = groups_[gi];
    if (!g.refine || g.empty() || !g.generator) {
      return std::nullopt;
    }
    return TrialCtx{fi, gi, &store_[fi], &g};
  }

  // Inform pair constraints of the frame count and pre-populate their per-frame
  // histograms (one compute_before_move pass per frame) so the averaged
  // sum_hist_ is correct from the first step. Run automatically on the first
  // run()/run_until() via EngineBase::ensure_initialised().
  void do_initialise() {
    BOOST_ASSERT_MSG(store_.size() > 0, "MultiFrameEngine: no frames added");
    const std::size_t N = store_.size();
    constraints_.set_n_frames(N);

    const std::size_t n_atoms = store_[0].size();
    std::vector<std::size_t> all_idx(n_atoms);
    std::iota(all_idx.begin(), all_idx.end(), std::size_t{0});

    for (std::size_t k = 0; k < N; ++k) {
      constraints_.set_active_frame(k);
      constraints_.compute_before_move(store_[k].coordinates, all_idx);
    }
    // Leave active_frame at 0 (arbitrary; reset per step).
    constraints_.set_active_frame(0);
  }

  [[nodiscard]] constexpr MultiFrameStore &store() noexcept { return store_; }
  [[nodiscard]] constexpr NoSpecies &species_policy() noexcept { return sp_; }
  [[nodiscard]] constexpr NoFeedback &feedback_policy() noexcept { return fb_; }
  [[nodiscard]] constexpr NoCollector &collector_policy() noexcept {
    return col_;
  }
  [[nodiscard]] constexpr NoBestTracking &best_policy() noexcept {
    return best_;
  }
  [[nodiscard]] constexpr NoCheckpoint &checkpoint_policy() noexcept {
    return ckpt_;
  }
  [[nodiscard]] constexpr GroupSelector &group_sel_for_feedback() noexcept {
    return group_selector_;
  }

  MultiFrameStore store_;
  GroupSelector frame_selector_;
  GroupSelector group_selector_;
  [[no_unique_address]] NoSpecies sp_;
  [[no_unique_address]] NoFeedback fb_;
  [[no_unique_address]] NoCollector col_;
  [[no_unique_address]] NoBestTracking best_;
  [[no_unique_address]] NoCheckpoint ckpt_;
};

} // namespace RMC
