#pragma once
#include <fullrmc/core/Types.hpp>
#include <vector>
#include <algorithm>

namespace fullrmc {

// Tracks atoms temporarily removed from the system (RemoveGenerator).
// Constraints query this to skip removed atoms without reallocating arrays.
class AtomsCollector {
public:
    void stage_removal(std::span<const index_t> indices) {
        pending_.assign(indices.begin(), indices.end());
    }

    void commit_removal() {
        for (auto i : pending_)
            removed_.push_back(i);
        pending_.clear();
        std::sort(removed_.begin(), removed_.end());
    }

    void rollback_removal() noexcept { pending_.clear(); }

    [[nodiscard]] bool is_removed(index_t i) const noexcept {
        return std::binary_search(removed_.begin(), removed_.end(), i);
    }

    [[nodiscard]] bool is_pending(index_t i) const noexcept {
        return std::ranges::contains(pending_, i);
    }

    [[nodiscard]] const std::vector<index_t>& removed()  const noexcept { return removed_; }
    [[nodiscard]] const std::vector<index_t>& pending()  const noexcept { return pending_; }
    [[nodiscard]] std::size_t n_active(std::size_t total) const noexcept {
        return total - removed_.size();
    }

    void clear() noexcept { removed_.clear(); pending_.clear(); }

private:
    std::vector<index_t> removed_;
    std::vector<index_t> pending_;
};

} // namespace fullrmc
