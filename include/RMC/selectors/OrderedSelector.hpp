#pragma once
#include <RMC/selectors/GroupSelector.hpp>

namespace RMC {

// Cycles through groups 0, 1, 2, …, N-1, 0, 1, … in order.
struct OrderedSelector : SelectorBase<OrderedSelector> {
  std::size_t current{0};
  constexpr FORCE_INLINE std::size_t select(IGroupSelector::Token,
                                            std::size_t n_groups) {
    std::size_t idx = current % n_groups;
    ++current;
    return idx;
  }
  constexpr void feedback(IGroupSelector::Token, std::size_t /*group_idx*/,
                          bool /*accepted*/) noexcept {}
};

} // namespace RMC
