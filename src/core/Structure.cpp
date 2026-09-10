#include <RMC/core/Structure.hpp>
#include <ranges>

namespace RMC {

void AtomicStructure::save_snapshot(std::span<const std::size_t> indices) {
  snapshot_indices_.assign(indices.begin(), indices.end());
  snapshot_coords_ = coordinates(snapshot_indices_, Eigen::all);
}

void AtomicStructure::restore_snapshot(std::span<const std::size_t>) {
  coordinates(snapshot_indices_, Eigen::all) = snapshot_coords_;
}

void AtomicStructure::assign_mutable_state(const AtomicStructure &other) {
  coordinates = other.coordinates;
  atomic_numbers = other.atomic_numbers;
  elements = other.elements;
}

void AtomicStructure::save_species_snapshot() {
  snapshot_atomic_numbers_.assign(atomic_numbers.begin(), atomic_numbers.end());
  // Build the code→symbol map once; species moves only permute existing pairs.
  if (code_to_symbol_.empty()) {
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
  const std::span codes(atomic_numbers.data(),
                        static_cast<std::size_t>(atomic_numbers.size()));
  for (auto &&[k, code, old] : std::views::zip(
           std::views::iota(0uz), codes, snapshot_atomic_numbers_)) {
    if (code == old) {
      continue;
    }
    code = old;
    if (const auto it = code_to_symbol_.find(old); it != code_to_symbol_.end()) {
      elements[k] = it->second;
    }
  }
}

} // namespace RMC
