#pragma once
#include <fullrmc/core/Types.hpp>
#include <memory>

namespace fullrmc {

// Type-erased group selector interface.
struct IGroupSelector {
    virtual ~IGroupSelector() = default;
    // Returns index in [0, n_groups) of the next group to move.
    virtual std::size_t select(std::size_t n_groups) = 0;
    // Called after a move is accepted/rejected so adaptive selectors can update.
    virtual void feedback(std::size_t /*group_idx*/, bool /*accepted*/) {}
    virtual std::unique_ptr<IGroupSelector> clone() const = 0;
};

} // namespace fullrmc
