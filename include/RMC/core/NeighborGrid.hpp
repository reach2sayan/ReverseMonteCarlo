#pragma once
#include <RMC/core/AtomsCollector.hpp>
#include <RMC/core/BoundaryConditions.hpp>
#include <RMC/core/Types.hpp>

#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace RMC {

// Linked-cell list over a (possibly triclinic) periodic box, used to answer
// "which atoms are within `cutoff` of atom i" in O(<n>) instead of an O(N)
// scan. It is kept in sync across Monte-Carlo moves (relocate/remove) and
// supports a small-set snapshot/restore so a rejected move can roll it back.
//
// Binning is done in FRACTIONAL coordinates (frac = inv_box * r), so it works
// for triclinic cells. The number of cells per axis is chosen so each cell is
// at least `cutoff` wide along its perpendicular direction; neighbours of an
// atom then lie in the atom's own cell plus the 26 surrounding cells (or, for a
// box too small to subdivide, in every cell along that axis — see the stencil
// note in neighbors_of). For an aperiodic / null box the grid degenerates to a
// single cell (brute force); the ADF use-case is periodic.
//
// IMPORTANT: neighbours of i are located from i's CURRENT coordinate row, not
// from its stored cell, so a query still works after i has been removed from
// the grid (the atom-removal path needs i's former neighbours even though i is
// absent).
class NeighborGrid {
public:
  // (Re)build from scratch. Absent atoms (collector) are not inserted. O(N).
  // Captures the box transform and per-axis cell counts for `cutoff`.
  void build(const coords_t &coords, const BoundaryConditions *bc,
             double cutoff, const AtomsCollector *collector);

  // Append the present neighbours of atom i (j != i, min-image distance <=
  // cutoff, j not absent) into `out`. `out` is cleared first.
  void neighbors_of(std::size_t i, const coords_t &coords,
                    const BoundaryConditions *bc, double cutoff,
                    const AtomsCollector *collector,
                    std::vector<std::uint32_t> &out) const;

  // Move atom i to the cell implied by its current coordinate row.
  void relocate(std::size_t i, const coords_t &coords);
  // Drop atom i from the grid (it became absent this move).
  void remove(std::size_t i) noexcept;

  // Snapshot the cell membership of a small index set (the moved atoms); the
  // matching restore undoes any relocate/remove applied to them since.
  void save_cells(std::span<const std::size_t> idx);
  void restore_cells() noexcept;
  // Discard a pending snapshot (the move was accepted, keep the new state).
  void clear_saved() noexcept { saved_.clear(); }
  [[nodiscard]] bool empty() const noexcept { return cell_of_.empty(); }

private:
  static constexpr std::int32_t kRemoved = -1;

  // Per-axis cell indices of a Cartesian position (column vector).
  [[nodiscard]] Eigen::Vector3i cell_coords(const vec3_t &r) const noexcept;
  [[nodiscard]] std::int32_t cell_index(const vec3_t &r) const noexcept;
  // Row-major strides {n_y·n_z, n_z, 1} so flat = cell_coords·strides().
  [[nodiscard]] Eigen::Vector3i strides() const noexcept {
    return Eigen::Vector3i{n_.y() * n_.z(), n_.z(), 1};
  }
  // Set atom i's cell to `target` (kRemoved drops it), fixing both arrays.
  void set_cell(std::size_t i, std::int32_t target);

  bool periodic_{false};
  mat3_t inv_box_{mat3_t::Identity()};   // fractional = inv_box_ * cartesian
  Eigen::Vector3i n_{1, 1, 1};           // cells per axis

  std::vector<std::int32_t> cell_of_;                  // flat cell per atom
  std::vector<std::vector<std::uint32_t>> cell_atoms_; // atoms per cell
  std::vector<std::pair<std::size_t, std::int32_t>>
      saved_; // (atom, prior cell)
};

} // namespace RMC
