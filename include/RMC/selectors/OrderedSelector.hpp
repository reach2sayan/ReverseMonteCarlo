#pragma once
#include <cstddef>

namespace RMC {

// Cycles through groups 0, 1, 2, …, N-1, 0, 1, … in order.
struct OrderedSelector {
  std::size_t current{0};
  std::size_t select(std::size_t n_groups) { return current++ % n_groups; }
};

} // namespace RMC
