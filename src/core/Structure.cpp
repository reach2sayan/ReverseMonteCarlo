#include <fullrmc/core/Structure.hpp>
#include <stdexcept>

namespace fullrmc {

void AtomicStructure::save_snapshot(std::span<const index_t> indices) {
    snapshot_indices_.assign(indices.begin(), indices.end());
    snapshot_coords_.resize(static_cast<Eigen::Index>(indices.size()), 3);
    for (Eigen::Index k = 0; k < static_cast<Eigen::Index>(indices.size()); ++k)
        snapshot_coords_.row(k) = coordinates.row(indices[static_cast<std::size_t>(k)]);
}

void AtomicStructure::restore_snapshot(std::span<const index_t> /*indices*/) {
    for (Eigen::Index k = 0; k < static_cast<Eigen::Index>(snapshot_indices_.size()); ++k)
        coordinates.row(snapshot_indices_[static_cast<std::size_t>(k)]) =
            snapshot_coords_.row(k);
}

} // namespace fullrmc
