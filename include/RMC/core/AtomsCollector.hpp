#pragma once
#include <RMC/core/Types.hpp>
#include <algorithm>
#include <cstddef>
#include <span>
#include <vector>

namespace RMC {

// Tracks atoms temporarily removed from the system (RemoveGenerator); queried
// by constraints to skip removed atoms. Membership is a flag array indexed by
// atom id in [0, total) (is_removed O(1)); pending_ is one move's index list.
class AtomsCollector {
public:
  constexpr void stage_removal(std::span<const index_t> indices) {
    pending_.assign(indices.begin(), indices.end());
  }
  constexpr void commit_removal() {
    for (const index_t i : pending_) {
      if (i >= removed_.size()) {
        removed_.resize(i + 1, 0);
      }
      if (removed_[i] == 0) {
        removed_[i] = 1;
        ++n_removed_;
      }
    }
    pending_.clear();
  }
  constexpr void rollback_removal() noexcept { pending_.clear(); }

  // Committed removal: survives accepted moves.
  [[nodiscard]] constexpr bool is_removed(index_t i) const noexcept {
    return i < removed_.size() && removed_[i] != 0;
  }
  // Staged this move, not yet committed.
  [[nodiscard]] constexpr bool is_pending(index_t i) const noexcept {
    return std::ranges::contains(pending_, i);
  }
  // Currently absent: committed set during compute_before_move (pending_ empty),
  // plus staged removals during compute_after_move. One predicate, both phases.
  [[nodiscard]] constexpr bool absent(index_t i) const noexcept {
    return is_removed(i) || is_pending(i);
  }

  [[nodiscard]] constexpr const std::vector<index_t> &pending() const noexcept {
    return pending_;
  }
  [[nodiscard]] constexpr std::size_t n_removed() const noexcept {
    return n_removed_;
  }
  [[nodiscard]] constexpr std::size_t
  n_active(std::size_t total) const noexcept {
    return total - n_removed_;
  }

  constexpr void clear() noexcept {
    removed_.clear();
    pending_.clear();
    n_removed_ = 0;
  }

private:
  std::vector<char> removed_;    // removed_[i] != 0  <=>  atom i committed-removed
  std::vector<index_t> pending_; // staged this move (small)
  std::size_t n_removed_{0};     // count of set flags (== distinct removed ids)
};

} // namespace RMC
