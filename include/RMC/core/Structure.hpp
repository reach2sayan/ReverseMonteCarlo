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

  // Species snapshot: saves only the (cheap, contiguous) atomic_numbers array
  // and reconstructs the few changed element symbols on restore from a
  // code→symbol map — species moves (SpeciesSwap) permute existing
  // (code, symbol) pairs, so the map built once stays valid. Avoids deep-copying
  // the whole std::vector<std::string> elements every step.
  // Call conditionally (only when a generator modifies species).
  // restore_species_snapshot() is a no-op if save was never called.
  void save_species_snapshot();
  void restore_species_snapshot();

  // Copy only the fields a running engine mutates — coordinates, atomic_numbers
  // and elements — from another structure with identical immutable metadata
  // (names, residues, molecule_ids). Used to broadcast the best replica in
  // cooperative ensembles without deep-copying the shared label vectors.
  // Keeps this object's vectors in place, so spans bound to them stay valid.
  void assign_mutable_state(const AtomicStructure &other);

  [[nodiscard]] FORCE_INLINE double
  distance(std::size_t i, std::size_t j,
           const BoundaryConditions &bc) const noexcept {
    return bc_min_image(bc, coordinates.row(j) - coordinates.row(i)).norm();
  }

private:
  coords_t snapshot_coords_;
  std::vector<std::size_t> snapshot_indices_;

  std::vector<int> snapshot_atomic_numbers_;
  // code (atomic number / occupation index) → element symbol, built once.
  std::unordered_map<int, std::string> code_to_symbol_;
  bool has_species_snapshot_{false};
};

} // namespace RMC
