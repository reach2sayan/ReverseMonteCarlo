#pragma once
#include <fullrmc/selectors/GroupSelector.hpp>

namespace fullrmc {

// Cycles through groups 0, 1, 2, … , N-1, 0, 1, … in order.
struct OrderedSelector : IGroupSelector {
    std::size_t current{0};

    std::size_t select(std::size_t n_groups) override {
        std::size_t idx = current % n_groups;
        ++current;
        return idx;
    }

    std::unique_ptr<IGroupSelector> clone() const override {
        return std::make_unique<OrderedSelector>(*this);
    }
};

} // namespace fullrmc
