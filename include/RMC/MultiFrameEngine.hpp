#pragma once
#include <RMC/constraints/ConstraintCollection.hpp>
#include <RMC/core/BoundaryConditions.hpp>
#include <RMC/core/Group.hpp>
#include <RMC/core/Structure.hpp>
#include <RMC/core/Types.hpp>
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
class MultiFrameEngine {
public:
  explicit MultiFrameEngine(BoundaryConditions bc,
                            std::uint32_t frame_rng_seed = 42,
                            std::uint32_t group_rng_seed = 43)
      : bc_(bc), frame_selector_(RandomSelector{frame_rng_seed}),
        group_selector_(RandomSelector{group_rng_seed}) {}

  void add_frame(AtomicStructure s) { frames_.push_back(std::move(s)); }

  void add_group(Group g) { groups_.push_back(std::move(g)); }

  void add_constraint(Constraint c) {
    c.set_boundary_conditions(bc_);
    constraints_.add(std::move(c));
  }

  void set_frame_selector(IGroupSelector s) {
    frame_selector_ = std::move(s);
  }
  void set_group_selector(IGroupSelector s) {
    group_selector_ = std::move(s);
  }

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

  void run(std::uint64_t n_steps) {
    BOOST_ASSERT_MSG(!frames_.empty(),
                     "MultiFrameEngine::run: call initialise() first");
    for (std::uint64_t i = 0; i < n_steps; ++i) {
      step();
    }
  }

  void run_until(double target_chi2, std::uint64_t max_steps = 0) {
    std::uint64_t i = 0;
    while (constraints_.total_error() > target_chi2) {
      if (max_steps > 0 && i >= max_steps)
        break;
      step();
      ++i;
    }
  }

  [[nodiscard]] const std::vector<AtomicStructure> &frames() const noexcept {
    return frames_;
  }
  [[nodiscard]] const AtomicStructure &best_frame() const {
    // Returns frame closest to lowest total error (all frames contribute
    // equally to the averaged constraint, so we just return frame 0; callers
    // who need the absolute best should compare external metrics).
    return frames_[0];
  }
  [[nodiscard]] double total_error() const noexcept {
    return constraints_.total_error();
  }
  [[nodiscard]] std::uint64_t steps_total() const noexcept {
    return n_steps_total_;
  }
  [[nodiscard]] std::uint64_t steps_accepted() const noexcept {
    return n_steps_accepted_;
  }

private:
  void step() {
    ++n_steps_total_;

    // Pick frame and group.
    const std::size_t fi = frame_selector_.select(frames_.size());
    const std::size_t gi = group_selector_.select(groups_.size());
    Group &g = groups_[gi];
    if (!g.refine || g.empty() || !g.generator)
      return;

    AtomicStructure &frame = frames_[fi];

    // Activate this frame in constraints, save snapshot, score before.
    constraints_.set_active_frame(fi);
    frame.save_snapshot(g.span());
    constraints_.compute_before_move(frame.coordinates, g.span());

    // Propose move.
    g.generator->generate(frame.coordinates, g.span());
    apply_pbc(fi, g.span());

    // Score after.
    constraints_.compute_after_move(frame.coordinates, g.span());

    // Accept / reject.
    if (constraints_.should_reject()) {
      frame.restore_snapshot(g.span());
      constraints_.reject();
    } else {
      constraints_.accept();
      ++n_steps_accepted_;
    }
  }

  void apply_pbc(std::size_t fi, std::span<const std::size_t> moved) {
    for (std::size_t i : moved) {
      vec3_t r = frames_[fi].coordinates.row(
                     static_cast<Eigen::Index>(i)).transpose();
      r = bc_wrap(bc_, r);
      frames_[fi].coordinates.row(static_cast<Eigen::Index>(i)) =
          r.transpose();
    }
  }

  std::vector<AtomicStructure> frames_;
  std::vector<Group> groups_;
  ConstraintCollection constraints_;
  BoundaryConditions bc_;
  IGroupSelector frame_selector_;
  IGroupSelector group_selector_;

  std::uint64_t n_steps_total_{0};
  std::uint64_t n_steps_accepted_{0};
};

} // namespace RMC
