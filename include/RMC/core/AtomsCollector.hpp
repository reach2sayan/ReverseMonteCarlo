#pragma once
#include <RMC/core/Types.hpp>
#include <algorithm>
#include <iterator>
#include <vector>

namespace RMC {

// Tracks atoms temporarily removed from the system (RemoveGenerator).
// Constraints query this to skip removed atoms without reallocating arrays.
class AtomsCollector {
public:
  constexpr void stage_removal(std::span<const index_t> indices) {
    pending_.assign(indices.begin(), indices.end());
  }
  constexpr void commit_removal() {
    std::ranges::copy(pending_, std::back_inserter(removed_));
    pending_.clear();
    std::ranges::sort(removed_);
  }
  constexpr void rollback_removal() noexcept { pending_.clear(); }
  [[nodiscard]] constexpr bool is_removed(index_t i) const noexcept {
    return std::ranges::binary_search(removed_, i);
  }
  [[nodiscard]] constexpr bool is_pending(index_t i) const noexcept {
    return std::ranges::contains(pending_, i);
  }
  [[nodiscard]] constexpr const std::vector<index_t> &removed() const noexcept {
    return removed_;
  }
  [[nodiscard]] constexpr const std::vector<index_t> &pending() const noexcept {
    return pending_;
  }
  [[nodiscard]] constexpr std::size_t
  n_active(std::size_t total) const noexcept {
    return total - removed_.size();
  }

  constexpr void clear() noexcept {
    removed_.clear();
    pending_.clear();
  }

private:
  std::vector<index_t> removed_;
  std::vector<index_t> pending_;
};

} // namespace RMC
