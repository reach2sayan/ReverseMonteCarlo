#include <RMC/core/Structure.hpp>
#include <ranges>
#include <stdexcept>

namespace RMC {

void AtomicStructure::save_snapshot(std::span<const std::size_t> indices) {
  // snapshot_coords_ only grows; snapshot_indices_.size() is the row count
  // (snapshot_coords_ may be larger from a previous, bigger group).
  snapshot_indices_.assign(indices.begin(), indices.end());
  const auto n = static_cast<Eigen::Index>(indices.size());
  if (snapshot_coords_.rows() < n) {
    snapshot_coords_.resize(n, 3);
  }
  snapshot_coords_ = coordinates(indices, Eigen::all);
}

void AtomicStructure::restore_snapshot(std::span<const std::size_t>) {
  for (const auto [k, idx] : snapshot_indices_ | std::views::enumerate) {
    coordinates.row(idx) = snapshot_coords_.row(k);
  }
}

void AtomicStructure::assign_mutable_state(const AtomicStructure &other) {
  coordinates = other.coordinates;
  atomic_numbers = other.atomic_numbers;
  elements = other.elements;
}

void AtomicStructure::save_species_snapshot() {
  const auto n = static_cast<std::size_t>(atomic_numbers.size());
  snapshot_atomic_numbers_.resize(n);
  std::copy(atomic_numbers.data(),
            atomic_numbers.data() + atomic_numbers.size(),
            snapshot_atomic_numbers_.begin());
  // Build the code→symbol map once; species moves only permute existing pairs.
  if (code_to_symbol_.empty() && !elements.empty()) {
    // zip stops at the shorter, giving the min(n, elements) bound.
    for (const auto &[code, sym] :
         std::views::zip(snapshot_atomic_numbers_, elements)) {
      code_to_symbol_.try_emplace(code, sym);
    }
  }
  has_species_snapshot_ = true;
}

void AtomicStructure::restore_species_snapshot() {
  if (!has_species_snapshot_) {
    return;
  }
  // Fix only sites whose code changed: restore code, rebuild symbol from map.
  for (Eigen::Index k = 0; k < atomic_numbers.size(); ++k) {
    const int old_code = snapshot_atomic_numbers_[static_cast<std::size_t>(k)];
    if (atomic_numbers[k] == old_code) {
      continue;
    }
    atomic_numbers[k] = old_code;
    if (const auto it = code_to_symbol_.find(old_code);
        it != code_to_symbol_.end()) {
      elements[static_cast<std::size_t>(k)] = it->second;
    }
  }
}

} // namespace RMC
