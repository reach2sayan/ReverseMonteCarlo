#include <RMC/selectors/DirectionalOrderSelector.hpp>

#include <algorithm>
#include <numeric>

namespace RMC {

DirectionalOrderSelector::DirectionalOrderSelector(vec3_t ref,
                                                   std::vector<vec3_t> centroids,
                                                   bool nf)
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

} // namespace RMC
