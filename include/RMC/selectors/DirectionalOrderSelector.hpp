#pragma once
#include <RMC/core/Types.hpp>
#include <algorithm>
#include <ranges>
#include <vector>

namespace RMC {

// Cycles through groups in order of increasing (nearest_first) or decreasing
// distance of their centroids from a fixed reference point.
//
// Useful for directional refinement (e.g. nearest-to-surface first).
//
// Usage:
//   std::vector<vec3_t> cents;
//   for (const auto& g : groups)
//       cents.push_back(structure.coordinates(g.indices, Eigen::all)
//                           .colwise().mean().transpose());
//   engine.set_selector(DirectionalOrderSelector{origin, cents});
struct DirectionalOrderSelector {
  bool nearest_first = true;

  DirectionalOrderSelector() = default;
  DirectionalOrderSelector(const vec3_t &ref,
                           const std::vector<vec3_t> &centroids, bool nf = true)
      : nearest_first{nf},
        order_{std::views::iota(std::size_t{0}, centroids.size()) |
               std::ranges::to<std::vector>()} {
    // Descending order sorts on the negated distance.
    std::ranges::sort(order_, {}, [&](std::size_t i) {
      const double d = (centroids[i] - ref).squaredNorm();
      return nearest_first ? d : -d;
    });
  }

  std::size_t select(std::size_t n_groups) {
    const std::size_t k = current_++ % n_groups;
    return order_.size() == n_groups ? order_[k] : k;
  }

private:
  std::vector<std::size_t> order_;
  std::size_t current_{0};
};

} // namespace RMC
