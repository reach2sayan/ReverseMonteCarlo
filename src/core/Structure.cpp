#include <fullrmc/core/Structure.hpp>
#include <stdexcept>

namespace fullrmc {

void AtomicStructure::save_snapshot(std::span<const std::size_t> indices) {
  snapshot_indices_.assign(indices.begin(), indices.end());
  Eigen::Map<const Eigen::Array<std::size_t, Eigen::Dynamic, 1>> idx(
      indices.data(), static_cast<Eigen::Index>(indices.size()));

  snapshot_coords_ = coordinates(idx, Eigen::all);
}

void AtomicStructure::restore_snapshot(std::span<const std::size_t>) {
  Eigen::Map<const Eigen::Array<std::size_t, Eigen::Dynamic, 1>> idx(
      snapshot_indices_.data(),
      static_cast<Eigen::Index>(snapshot_indices_.size()));

  coordinates(idx, Eigen::all) = snapshot_coords_;
}

} // namespace fullrmc
