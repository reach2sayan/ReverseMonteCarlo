#pragma once
#include <RMC/core/BoundaryConditions.hpp>
#include <RMC/core/Types.hpp>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace RMC {

struct AtomicStructure {
  // ---- Coordinate data ----
  coords_t coordinates;           // N×3, row-major: row i = (x,y,z) of atom i
  ivec_t atomic_numbers;          // N
  std::vector<std::string> names; // atom names (e.g. "CA", "OW")
  std::vector<std::string> elements; // element symbols (e.g. "C", "O")
  std::vector<std::string> residues;
  std::vector<std::size_t> molecule_ids; // groups atoms into molecules

  [[nodiscard]] constexpr std::size_t size() const noexcept {
    return static_cast<std::size_t>(coordinates.rows());
  }

  // ---- Snapshot support for move revert ----
  void save_snapshot(std::span<const std::size_t> indices);
  void restore_snapshot(std::span<const std::size_t> indices);

  // Species snapshot: saves the full elements/atomic_numbers arrays.
  // Call conditionally (only when a generator modifies species).
  // restore_species_snapshot() is a no-op if save was never called.
  void save_species_snapshot();
  void restore_species_snapshot();

  [[nodiscard]] FORCE_INLINE double
  distance(std::size_t i, std::size_t j,
           const BoundaryConditions &bc) const noexcept {
    return bc_min_image(bc, coordinates.row(j) - coordinates.row(i)).norm();
  }

private:
  coords_t snapshot_coords_;
  std::vector<std::size_t> snapshot_indices_;

  std::vector<std::string> snapshot_elements_;
  std::vector<int> snapshot_atomic_numbers_;
  bool has_species_snapshot_{false};
};

} // namespace RMC
