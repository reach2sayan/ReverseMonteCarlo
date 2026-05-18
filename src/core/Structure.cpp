#include <RMC/core/Structure.hpp>
#include <stdexcept>

namespace RMC {

void AtomicStructure::save_snapshot(std::span<const std::size_t> indices) {
  snapshot_indices_.assign(indices.begin(), indices.end());
  snapshot_coords_.resize(static_cast<Eigen::Index>(indices.size()), 3);
  for (Eigen::Index k = 0; k < static_cast<Eigen::Index>(indices.size()); ++k)
    snapshot_coords_.row(k) =
        coordinates.row(static_cast<Eigen::Index>(indices[k]));
}

void AtomicStructure::restore_snapshot(std::span<const std::size_t>) {
  for (Eigen::Index k = 0;
       k < static_cast<Eigen::Index>(snapshot_indices_.size()); ++k)
    coordinates.row(static_cast<Eigen::Index>(snapshot_indices_[k])) =
        snapshot_coords_.row(k);
}

} // namespace RMC
