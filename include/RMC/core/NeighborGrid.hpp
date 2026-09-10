#pragma once
#include <RMC/core/AtomsCollector.hpp>
#include <RMC/core/BoundaryConditions.hpp>
#include <RMC/core/Types.hpp>

#include <boost/container/small_vector.hpp>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace RMC {

// Linked-cell list over a (possibly triclinic) periodic box: "atoms within
// cutoff of i" in O(<n>). Binning in fractional coords (frac = inv_box * r).
// Cells sized >= cutoff per axis, so neighbours lie in i's cell + 26 around;
// null box degenerates to one cell (brute force).
// Neighbours of i are located from i's CURRENT coordinate row, not its stored
// cell, so a query still works after i has been removed from the grid.
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
  std::vector<boost::container::small_vector<std::uint32_t, 8>>
      cell_atoms_; // atoms per cell
  std::vector<std::pair<std::size_t, std::int32_t>>
      saved_; // (atom, prior cell)
};

} // namespace RMC
