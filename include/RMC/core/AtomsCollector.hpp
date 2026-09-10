#pragma once
#include <RMC/core/Types.hpp>
#include <algorithm>
#include <boost/container/small_vector.hpp>
#include <boost/dynamic_bitset.hpp>
#include <cstddef>
#include <span>

namespace RMC {

// Tracks atoms temporarily removed from the system (RemoveGenerator); queried
// by constraints to skip removed atoms. Committed removals are a bitset over
// atom ids; pending_ is one move's staged index list.
class AtomsCollector {
public:
  void stage_removal(std::span<const index_t> indices) {
    pending_.assign(indices.begin(), indices.end());
  }
  void commit_removal() {
    for (const index_t i : pending_) {
      if (i >= removed_.size()) {
        removed_.resize(i + 1);
      }
      removed_.set(i);
    }
    pending_.clear();
  }
  void rollback_removal() noexcept { pending_.clear(); }

  // Committed removal: survives accepted moves.
  [[nodiscard]] bool is_removed(index_t i) const noexcept {
    return i < removed_.size() && removed_[i];
  }
  // Staged this move, not yet committed.
  [[nodiscard]] bool is_pending(index_t i) const noexcept {
    return std::ranges::contains(pending_, i);
  }
  // Currently absent: committed set during compute_before_move (pending_ empty),
  // plus staged removals during compute_after_move. One predicate, both phases.
  [[nodiscard]] bool absent(index_t i) const noexcept {
    return is_removed(i) || is_pending(i);
  }

  [[nodiscard]] const auto &pending() const noexcept { return pending_; }
  [[nodiscard]] std::size_t n_removed() const noexcept { return removed_.count(); }
  [[nodiscard]] std::size_t n_active(std::size_t total) const noexcept {
    return total - n_removed();
  }

  void clear() noexcept {
    removed_.clear();
    pending_.clear();
  }

private:
  boost::dynamic_bitset<> removed_;
  boost::container::small_vector<index_t, 8> pending_;
};

} // namespace RMC
