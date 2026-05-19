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

void AtomicStructure::save_species_snapshot() {
  snapshot_elements_ = elements;
  snapshot_atomic_numbers_.resize(
      static_cast<std::size_t>(atomic_numbers.size()));
  for (Eigen::Index k = 0; k < atomic_numbers.size(); ++k)
    snapshot_atomic_numbers_[static_cast<std::size_t>(k)] = atomic_numbers[k];
  has_species_snapshot_ = true;
}

void AtomicStructure::restore_species_snapshot() {
  if (!has_species_snapshot_)
    return;
  elements = snapshot_elements_;
  for (Eigen::Index k = 0; k < atomic_numbers.size(); ++k)
    atomic_numbers[k] = snapshot_atomic_numbers_[static_cast<std::size_t>(k)];
}

} // namespace RMC
