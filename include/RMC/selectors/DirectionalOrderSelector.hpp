#pragma once
#include <RMC/core/Types.hpp>
#include <RMC/selectors/GroupSelector.hpp>
#include <algorithm>
#include <numeric>
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
                           bool nf = true)
      : nearest_first(nf) {
    const std::size_t n = centroids.size();
    order_.resize(n);
    std::iota(order_.begin(), order_.end(), std::size_t{0});
    std::ranges::sort(order_, [&](std::size_t a, std::size_t b) {
      const double da = (centroids[a] - ref).squaredNorm();
      const double db = (centroids[b] - ref).squaredNorm();
      return nearest_first ? da < db : da > db;
    });
  }

  std::size_t select(GroupSelector::Token, std::size_t n_groups) {
    if (order_.size() != n_groups) {
      return current_++ % n_groups;
    }
    const std::size_t idx = order_[current_ % n_groups];
    ++current_;
    return idx;
  }

  constexpr void feedback(GroupSelector::Token, std::size_t,
                          bool) noexcept {}

private:
  std::vector<std::size_t> order_;
  std::size_t current_{0};
};

} // namespace RMC
