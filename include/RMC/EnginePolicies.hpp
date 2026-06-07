#pragma once
#include <RMC/core/AtomsCollector.hpp>
#include <RMC/core/Group.hpp>
#include <RMC/core/Structure.hpp>
#include <RMC/io/Checkpoint.hpp>
#include <RMC/selectors/GroupSelector.hpp>

#include <boost/log/trivial.hpp>

#include <cstdint>
#include <filesystem>
#include <limits>
#include <optional>

namespace RMC {

// Orthogonal, compile-time engine features. Each axis has an "on" type carrying
// any needed state and a stateless "off" type whose methods are no-ops. The
// engines hold these as [[no_unique_address]] members, so the "off" variants add
// zero size and the calls inline away. The shared pipeline in EngineBase invokes
// them at fixed hook points.

// --- Species snapshots (atom-swap moves) -----------------------------------
struct WithSpecies {
  static void save_before(AtomicStructure &f, Group &g) {
    if (g.generator->modifies_species()) {
      f.save_species_snapshot();
    }
  }
  static void restore_on_reject(AtomicStructure &f) {
    f.restore_species_snapshot(); // no-op if save was never called
  }
};
struct NoSpecies {
  static constexpr void save_before(AtomicStructure &, Group &) noexcept {}
  static constexpr void restore_on_reject(AtomicStructure &) noexcept {}
};

struct WithFeedback {
  static void feedback(GroupSelector &sel, std::size_t gi, bool accepted) {
    sel.feedback(gi, accepted);
  }
};
struct NoFeedback {
  static constexpr void feedback(GroupSelector &, std::size_t, bool) noexcept {}
};

// --- Pending atom removal (RemoveGenerator) --------------------------------
struct WithCollector {
  AtomsCollector collector_;
  void commit_or_rollback(bool accepted) {
    if (accepted && !collector_.pending().empty()) {
      collector_.commit_removal();
    } else {
      collector_.rollback_removal();
    }
  }
  [[nodiscard]] constexpr AtomsCollector &collector() noexcept {
    return collector_;
  }
};
struct NoCollector {
  constexpr void commit_or_rollback(bool) noexcept {}
};

// --- Best-ever configuration tracking --------------------------------------
// With simulated annealing the FINAL MC state is deliberately not the minimum,
// so SQS wants the best-ever structure. Off by default: greedy fitting's final
// state is already its best, and tracking costs an O(N) copy per improvement.
struct WithBestTracking {
  bool track_{false};
  double best_error_{std::numeric_limits<double>::max()};
  std::optional<AtomicStructure> best_structure_;

  constexpr void set_track(bool on) noexcept { track_ = on; }
  void update(const AtomicStructure &cur, double err) {
    if (track_ && err < best_error_) {
      best_error_ = err;
      best_structure_ = cur; // snapshot the new best configuration
    }
  }
  // Lowest error seen (or the live error when not tracking — so ensemble
  // winner-selection is correct either way).
  [[nodiscard]] constexpr double best_error(double live) const noexcept {
    return track_ ? best_error_ : live;
  }
  [[nodiscard]] constexpr const AtomicStructure &
  best_structure(const AtomicStructure &cur) const noexcept {
    return best_structure_ ? *best_structure_ : cur;
  }
};
struct NoBestTracking {
  void update(const AtomicStructure &, double) noexcept {}
};

// --- Periodic checkpointing ------------------------------------------------
struct WithCheckpoint {
  std::optional<std::filesystem::path> path_;
  std::uint64_t every_{5000};

  void configure(std::filesystem::path path, std::uint64_t every) {
    path_ = std::move(path);
    every_ = every;
  }
  void maybe(const AtomicStructure &s, const io::EngineStats &stats,
             std::uint64_t n_accepted) {
    if (path_ && n_accepted > 0 && n_accepted % every_ == 0) {
      if (auto r = io::save_checkpoint(s, stats, *path_); !r) {
        BOOST_LOG_TRIVIAL(warning) << "Checkpoint save failed";
      }
    }
  }
};
struct NoCheckpoint {
  void maybe(const AtomicStructure &, const io::EngineStats &,
             std::uint64_t) noexcept {}
};

} // namespace RMC
