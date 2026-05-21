#pragma once
#include <RMC/EngineBase.hpp>
#include <RMC/selectors/GroupSelector.hpp>
#include <RMC/selectors/RandomSelector.hpp>

#include <boost/assert.hpp>
#include <boost/log/trivial.hpp>

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
//   eng.add_constraint(pdf_constraint);  // must call set_n_frames after add
//   eng.initialise();                    // pre-populates frame histograms
//   eng.run(100'000);
class MultiFrameEngine : public EngineBase<MultiFrameEngine> {
public:
  explicit MultiFrameEngine(BoundaryConditions bc,
                            std::uint32_t frame_rng_seed = 42,
                            std::uint32_t group_rng_seed = 1729)
      : EngineBase<MultiFrameEngine>(bc),
        frame_selector_(RandomSelector{frame_rng_seed}),
        group_selector_(RandomSelector{group_rng_seed}) {}

  void add_frame(AtomicStructure s) { frames_.push_back(std::move(s)); }

  void set_frame_selector(GroupSelector s) { frame_selector_ = std::move(s); }
  void set_group_selector(GroupSelector s) { group_selector_ = std::move(s); }

  // Call once after all frames/groups/constraints have been added.
  // Informs pair constraints of the frame count and pre-populates their
  // per-frame histograms by doing one compute_before_move pass per frame.
  void initialise() {
    BOOST_ASSERT_MSG(!frames_.empty(), "MultiFrameEngine: no frames added");
    BOOST_ASSERT_MSG(!groups_.empty(), "MultiFrameEngine: no groups added");

    const std::size_t N = frames_.size();
    constraints_.set_n_frames(N);

    // Build a full all-atoms index list used for initial histogram population.
    const std::size_t n_atoms = frames_[0].size();
    std::vector<std::size_t> all_idx(n_atoms);
    std::iota(all_idx.begin(), all_idx.end(), std::size_t{0});

    for (std::size_t k = 0; k < N; ++k) {
      constraints_.set_active_frame(k);
      constraints_.compute_before_move(frames_[k].coordinates, all_idx);
    }
    // Leave active_frame pointing at frame 0 (arbitrary; will be set per step).
    constraints_.set_active_frame(0);
  }

  [[nodiscard]] constexpr const std::vector<AtomicStructure> &
  frames() const noexcept {
    return frames_;
  }
  [[nodiscard]] constexpr const AtomicStructure &best_frame() const {
    // Returns frame closest to lowest total error (all frames contribute
    // equally to the averaged constraint, so we just return frame 0; callers
    // who need the absolute best should compare external metrics).
    return frames_[0];
  }

private:
  friend class EngineBase<MultiFrameEngine>;

  struct TrialCtx {
    std::size_t fi;
    std::size_t gi;
    AtomicStructure *frame;
    Group *group;
  };

  std::optional<TrialCtx> select_frame_and_group() {
    const std::size_t fi = frame_selector_.select(frames_.size());
    const std::size_t gi = group_selector_.select(groups_.size());
    Group &g = groups_[gi];
    if (!g.refine || g.empty() || !g.generator) {
      return std::nullopt;
    }
    return TrialCtx{fi, gi, &frames_[fi], &g};
  }
  void snapshot_and_score_before(TrialCtx &c) {
    ++n_steps_tried_;
    constraints_.set_active_frame(c.fi);
    c.frame->save_snapshot(c.group->span());
    constraints_.compute_before_move(c.frame->coordinates, c.group->span());
  }
  void propose_move(TrialCtx &c) {
    c.group->generator->generate(c.frame->coordinates, c.group->span());
    apply_pbc_to(*c.frame, c.group->span());
  }
  void score_after(TrialCtx &c) {
    constraints_.compute_after_move(c.frame->coordinates, c.group->span());
  }
  void settle(TrialCtx &c) {
    if (constraints_.should_reject()) {
      c.frame->restore_snapshot(c.group->span());
      constraints_.reject();
    } else {
      constraints_.accept();
      ++n_steps_accepted_;
    }
  }

  void step() {
    ++n_steps_total_;
    select_frame_and_group()
        .and_then(stage([&](TrialCtx &c) { snapshot_and_score_before(c); }))
        .and_then(stage([&](TrialCtx &c) { propose_move(c); }))
        .and_then(stage([&](TrialCtx &c) { score_after(c); }))
        .and_then(stage([&](TrialCtx &c) { settle(c); }));
    maybe_log();
  }

  std::vector<AtomicStructure> frames_;
  GroupSelector frame_selector_;
  GroupSelector group_selector_;
};

} // namespace RMC
