#pragma once
#include <RMC/core/AtomsCollector.hpp>
#include <RMC/core/Group.hpp>
#include <RMC/core/Structure.hpp>
#include <RMC/io/Checkpoint.hpp>
#include <RMC/selectors/GroupSelector.hpp>

#include <spdlog/spdlog.h>

#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>

namespace RMC {

// Orthogonal, compile-time engine features held as [[no_unique_address]]
// members and invoked at fixed hook points by EngineBase's pipeline.

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

struct WithFeedback {
  static void feedback(GroupSelector &sel, std::size_t gi, bool accepted) {
    sel.feedback(gi, accepted);
  }
};

// --- Pending atom removal (RemoveGenerator) --------------------------------
// Heap-stable so RemoveGenerators and constraints can hold a raw pointer that
// survives the engine being moved.
struct WithCollector {
  std::unique_ptr<AtomsCollector> collector_{std::make_unique<AtomsCollector>()};
  void commit_or_rollback(bool accepted) {
    accepted ? collector_->commit_removal() : collector_->rollback_removal();
  }
  [[nodiscard]] AtomsCollector *collector() const noexcept {
    return collector_.get();
  }
};

// --- Best-ever configuration tracking --------------------------------------
// For annealing, where the final MC state is not the minimum. Off by default
// (greedy's final state is already its best; tracking costs an O(N) copy per improvement).
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
  // Lowest error seen, or the live error when not tracking.
  [[nodiscard]] constexpr double best_error(double live) const noexcept {
    return track_ ? best_error_ : live;
  }
  [[nodiscard]] constexpr const AtomicStructure &
  best_structure(const AtomicStructure &cur) const noexcept {
    return best_structure_ ? *best_structure_ : cur;
  }
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
        spdlog::warn("Checkpoint save failed");
      }
    }
  }
};

} // namespace RMC
