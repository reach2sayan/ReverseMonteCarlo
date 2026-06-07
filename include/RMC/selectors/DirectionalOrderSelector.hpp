#pragma once
#include <RMC/core/Types.hpp>
#include <RMC/selectors/GroupSelector.hpp>
#include <vector>

namespace RMC {

// Cycles through groups in order of increasing or decreasing distance from a
// fixed reference point. Order is computed once at construction from the
// supplied group centroids and then cycled indefinitely.
//
// Useful for directional refinement (e.g. nearest-to-surface first).
//
// Usage:
//   std::vector<vec3_t> cents;
//   for (const auto& g : groups)
//       cents.push_back(structure.coordinates(g.indices, Eigen::all)
//                           .colwise().mean().transpose());
//   engine.set_selector(DirectionalOrderSelector{origin, cents});
struct DirectionalOrderSelector : SelectorBase<DirectionalOrderSelector> {
  bool nearest_first = true;

  DirectionalOrderSelector() = default;
  DirectionalOrderSelector(vec3_t ref, std::vector<vec3_t> centroids,
                           bool nf = true);

  std::size_t select(GroupSelector::Token, std::size_t n_groups) {
    if (order_.size() != n_groups) {
      return current_++ % n_groups;
    }
    const std::size_t idx = order_[current_ % n_groups];
    ++current_;
    return idx;
  }

  constexpr void feedback(GroupSelector::Token, std::size_t, bool) noexcept {}

private:
  std::vector<std::size_t> order_;
  std::size_t current_{0};
};

} // namespace RMC
