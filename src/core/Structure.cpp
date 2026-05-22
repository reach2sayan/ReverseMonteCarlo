#include <RMC/core/Structure.hpp>
#include <stdexcept>

namespace RMC {

void AtomicStructure::save_snapshot(std::span<const std::size_t> indices) {
  snapshot_indices_.assign(indices.begin(), indices.end());
  snapshot_coords_.resize(static_cast<Eigen::Index>(indices.size()), 3);
  for (Eigen::Index k = 0; k < snapshot_coords_.rows(); ++k) {
    snapshot_coords_.row(k) = coordinates.row(indices[k]);
  }
}

void AtomicStructure::restore_snapshot(std::span<const std::size_t>) {
  for (Eigen::Index k = 0; k < snapshot_coords_.rows(); ++k) {
    coordinates.row(snapshot_indices_[k]) = snapshot_coords_.row(k);
  }
}

void AtomicStructure::save_species_snapshot() {
  snapshot_elements_ = elements;
  snapshot_atomic_numbers_.resize(
      static_cast<std::size_t>(atomic_numbers.size()));
  std::copy(atomic_numbers.data(),
            atomic_numbers.data() + atomic_numbers.size(),
            snapshot_atomic_numbers_.begin());
  has_species_snapshot_ = true;
}

void AtomicStructure::restore_species_snapshot() {
  if (!has_species_snapshot_) {
    return;
  }
  elements = snapshot_elements_;
  std::copy(snapshot_atomic_numbers_.begin(), snapshot_atomic_numbers_.end(),
            atomic_numbers.data());
}

} // namespace RMC
