#pragma once
#include <RMC/core/Types.hpp>
#include <cstddef>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace RMC {

// Shared single-/multi-frame incremental histogram state machine used by every
// pair/angle constraint. It owns the running raw-count buffers and the
// before-move / after-move delta bookkeeping; the constraint supplies only the
// problem-specific kernels (full build, moved-atom delta) and any auxiliary
// hooks (e.g. neighbour-grid relocation) as callables, so the O(K·N)
// incremental MC machinery lives in exactly one place.
//
// Lifecycle per MC step (see update()):
//   * before-move call (moved staged, incremental_ready_ == false): snapshot the
//     active histogram, lazily full-build it on first touch, and record the
//     moved-atom delta at the OLD coords.
//   * after-move call (incremental_ready_ == true): recompute the moved-atom
//     delta at the NEW coords and patch  new = saved − D_old + D_new.
//   * full-evaluation call (empty `moved`): always takes the before path.
struct IncrementalHistogram {
  int len_{0};
  std::size_t n_frames_{1};
  std::size_t active_frame_{0};

  // Per-frame running counts: one slot per frame, engaged once that frame's full
  // histogram has been built. A disengaged slot means "full rebuild pending" and
  // contributes nothing to sum_hist_. Single-frame is simply n_frames_ == 1 (one
  // slot); there is no separate fast path.
  mutable std::vector<std::optional<vec_t>> frame_hists_;
  mutable vec_t sum_hist_;        // running Σ over the engaged frame slots
  mutable vec_t saved_frame_hist_; // pre-move snapshot for rollback

  // Incremental neighbour-delta scratch (pre-sized by set_length()).
  mutable vec_t saved_moved_delta_; // moved-atom contribution at old coords
  mutable vec_t scratch_delta_;     // reused after-move delta buffer
  mutable bool incremental_ready_{false}; // set by before-move, cleared after

  // Pre-size the per-step scratch buffers so the hot path can setZero() in
  // place instead of allocating each step. Call once the bin count is known.
  void set_length(int len) {
    len_ = len;
    saved_moved_delta_.resize(len);
    scratch_delta_.resize(len);
  }

  void set_n_frames(std::size_t n) {
    n_frames_ = n;
    frame_hists_.assign(n, std::nullopt); // all pending full rebuild
    sum_hist_ = vec_t::Zero(len_);
    saved_frame_hist_.resize(len_);
    incremental_ready_ = false;
  }

  void set_active_frame(std::size_t k) noexcept {
    active_frame_ = k;
    incremental_ready_ = false; // frame switch invalidates any pending delta
  }

  // Force the active frame to full-rebuild on its next compute (drift resync).
  // Drop its contribution from the running sum so the slot can be rebuilt clean.
  void invalidate_active() noexcept {
    if (auto &slot = frame_hists_[active_frame_]; slot) {
      sum_hist_ -= *slot;
      slot.reset();
    }
  }

  // Run one MC-step update and write the frame-averaged (Σ frames / N) counts
  // into `out`. The four callables are:
  //   build_full(vec_t& hist)              — full O(N²) histogram build
  //   accum_moved(vec_t& delta, span moved) — moved-atom contribution
  //   before_move(span moved)              — pre-delta hook (old coords)
  //   after_move(span moved)               — pre-delta hook (new coords)
  template <class BuildFull, class AccumMoved, class BeforeMove, class AfterMove>
  void update(vec_t &out, std::span<const std::size_t> moved,
              BuildFull &&build_full, AccumMoved &&accum_moved,
              BeforeMove &&before_move, AfterMove &&after_move) const {
    auto &slot = frame_hists_[active_frame_];
    if (incremental_ready_ && !moved.empty()) {
      // Slot is engaged here: incremental_ready_ is only set after a before-move
      // pass, which always builds the slot.
      after_move(moved);
      scratch_delta_.setZero();
      accum_moved(scratch_delta_, moved);
      vec_t new_frame_hist =
          saved_frame_hist_ - saved_moved_delta_ + scratch_delta_;
      sum_hist_ += new_frame_hist - *slot;
      slot = std::move(new_frame_hist);
      incremental_ready_ = false;
    } else {
      // Lazily full-build a pending slot, monadically: keep the engaged value,
      // otherwise build one and fold it into the running sum (the or_else body
      // runs only on a disengaged slot, so the sum is credited exactly once).
      slot = std::move(slot).or_else([&] {
        vec_t tmp = vec_t::Zero(len_);
        build_full(tmp);
        sum_hist_ += tmp;
        return std::optional<vec_t>{std::move(tmp)};
      });
      saved_frame_hist_ = *slot;
      if (!moved.empty()) {
        before_move(moved);
        saved_moved_delta_.setZero();
        accum_moved(saved_moved_delta_, moved);
        incremental_ready_ = true;
      }
    }
    out = sum_hist_ / static_cast<double>(n_frames_);
  }

  // Restore the active frame to its pre-move snapshot (constraint reject()).
  // `on_rollback` runs any auxiliary undo (e.g. grid cell restore).
  template <class OnRollback>
  void rollback(OnRollback &&on_rollback) const noexcept {
    // The active slot is engaged here (a compute always builds it before a
    // reject); swap its sum contribution back to the pre-move snapshot.
    auto &slot = frame_hists_[active_frame_];
    sum_hist_ += saved_frame_hist_ - *slot;
    slot = saved_frame_hist_;
    incremental_ready_ = false;
    on_rollback();
  }
};

} // namespace RMC
