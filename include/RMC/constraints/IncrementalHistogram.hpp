#pragma once
#include <RMC/core/Types.hpp>
#include <boost/assert.hpp>
#include <cstddef>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace RMC {

// Single-/multi-frame incremental histogram engine (lifecycle per MC step in update()).
struct IncrementalHistogram {
  int len_{0};
  std::size_t n_frames_{1};
  std::size_t active_frame_{0};

  // Per-frame running counts; a disengaged slot means "full rebuild pending"
  // and contributes nothing to sum_hist_.
  mutable std::vector<std::optional<vec_t>> frame_hists_;
  mutable vec_t sum_hist_;         // running Σ over the engaged frame slots
  mutable vec_t saved_frame_hist_; // pre-move snapshot for rollback

  // Incremental neighbour-delta scratch (pre-sized by set_length()).
  mutable vec_t saved_moved_delta_; // moved-atom contribution at old coords
  mutable vec_t scratch_delta_;     // reused after-move delta buffer
  mutable bool incremental_ready_{false}; // set by before-move, cleared after

  // No-op before/after-move hook (the pair path needs none).
  struct NoHook {
    void operator()(std::span<const std::size_t>) const noexcept {}
  };

  // Pre-size the per-step scratch buffers. Call once the bin count is known.
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

  void invalidate_active() noexcept {
    if (auto &slot = frame_hists_[active_frame_]; slot.has_value()) {
      sum_hist_ -= slot.value();
      slot.reset();
    }
  }

  template <class BuildFull, class AccumMoved, class BeforeMove = NoHook,
            class AfterMove = NoHook>
  void update(vec_t &out, std::span<const std::size_t> moved,
              BuildFull &&build_full, AccumMoved &&accum_moved,
              BeforeMove &&before_move = {}, AfterMove &&after_move = {}) const {
    auto &slot = frame_hists_[active_frame_];
    if (incremental_ready_ && !moved.empty()) {
      // after-move: patch new = saved − D_old + D_new (delta at NEW coords),
      // built in the reused delta buffer and swapped into the slot.
      BOOST_ASSERT_MSG(slot.has_value(), "Slot is Disengaged");
      after_move(moved);
      scratch_delta_.setZero();
      accum_moved(scratch_delta_, moved);
      scratch_delta_ += saved_frame_hist_ - saved_moved_delta_;
      sum_hist_ += scratch_delta_ - *slot;
      slot->swap(scratch_delta_);
      incremental_ready_ = false;
    } else {
      // before-move (or full eval when `moved` empty): snapshot, lazily full-build
      // a pending slot.
      slot = std::move(slot).or_else([&] {
        vec_t tmp = vec_t::Zero(len_);
        build_full(tmp);
        sum_hist_ += tmp;
        return std::optional<vec_t>{std::move(tmp)};
      });
      saved_frame_hist_ = slot.value();
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
    auto &slot = frame_hists_[active_frame_];
    BOOST_ASSERT_MSG(slot.has_value(), "slot is disengaged");
    sum_hist_ += saved_frame_hist_ - slot.value();
    slot = saved_frame_hist_;
    incremental_ready_ = false;
    on_rollback();
  }
};

} // namespace RMC
