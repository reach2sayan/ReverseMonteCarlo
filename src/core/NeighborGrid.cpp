#include <RMC/core/NeighborGrid.hpp>

#include <algorithm>
#include <iterator>

namespace RMC {

Eigen::Vector3i NeighborGrid::cell_coords(const vec3_t &r) const noexcept {
  if (!periodic_) {
    return Eigen::Vector3i::Zero();
  }
  Eigen::Array3d frac = (inv_box_ * r).array();
  frac -= frac.floor(); // wrap into [0, 1)
  const Eigen::Array3i c = (frac * n_.cast<double>().array()).floor().cast<int>();
  // Guard the frac == 1-ε case that rounds up to n_ back into [0, n_).
  return c.max(0).min(n_.array() - 1).matrix();
}

std::int32_t NeighborGrid::cell_index(const vec3_t &r) const noexcept {
  return cell_coords(r).dot(strides());
}

void NeighborGrid::build(const coords_t &coords, const BoundaryConditions *bc,
                         double cutoff, const AtomsCollector *collector) {
  const Eigen::Index N = coords.rows();
  const PeriodicBC *pbc = bc ? std::get_if<PeriodicBC>(bc) : nullptr;
  if (pbc && cutoff > 0.0) {
    periodic_ = true;
    inv_box_ = pbc->inv_box();
    // Perpendicular width along reciprocal direction a is 1/|b_a|, with b_a the
    // a-th reciprocal vector = row a of inv_box (since inv_box * box = I and
    // the lattice vectors are the columns of box). A cell must be >= cutoff
    // wide so that neighbours fall within the +/-1 cell stencil.
    const Eigen::Array3d widths =
        inv_box_.rowwise().norm().cwiseInverse().array();
    n_ = (widths / cutoff).floor().cast<int>().max(1).matrix();
  } else {
    periodic_ = false;
    inv_box_ = mat3_t::Identity();
    n_.setOnes();
  }

  const std::size_t ncells = static_cast<std::size_t>(n_.prod());
  cell_atoms_.assign(ncells, {});
  cell_of_.assign(static_cast<std::size_t>(N), kRemoved);
  saved_.clear();

  for (Eigen::Index i = 0; i < N; ++i) {
    const std::size_t ii = static_cast<std::size_t>(i);
    if (collector && collector->absent(ii)) {
      continue; // removed atom is not in the grid
    }
    const std::int32_t c = cell_index(coords.row(i).transpose());
    cell_of_[ii] = c;
    cell_atoms_[static_cast<std::size_t>(c)].push_back(
        static_cast<std::uint32_t>(i));
  }
}

void NeighborGrid::set_cell(std::size_t i, std::int32_t target) {
  const std::int32_t cur = cell_of_[i];
  if (cur == target) {
    return;
  }
  if (cur >= 0) {
    auto &v = cell_atoms_[static_cast<std::size_t>(cur)];
    const auto u = static_cast<std::uint32_t>(i);
    const auto it = std::ranges::find(v, u);
    if (it != v.end()) {
      *it = v.back();
      v.pop_back();
    }
  }
  if (target >= 0) {
    cell_atoms_[static_cast<std::size_t>(target)].push_back(
        static_cast<std::uint32_t>(i));
  }
  cell_of_[i] = target;
}

void NeighborGrid::relocate(std::size_t i, const coords_t &coords) {
  set_cell(i, cell_index(coords.row(static_cast<Eigen::Index>(i)).transpose()));
}

void NeighborGrid::remove(std::size_t i) noexcept { set_cell(i, kRemoved); }

void NeighborGrid::save_cells(std::span<const std::size_t> idx) {
  saved_.clear();
  saved_.reserve(idx.size());
  std::ranges::transform(idx, std::back_inserter(saved_), [&](std::size_t i) {
    return std::pair{i, cell_of_[i]};
  });
}

void NeighborGrid::restore_cells() noexcept {
  for (const auto &[i, prior] : saved_) {
    set_cell(i, prior);
  }
  saved_.clear();
}

void NeighborGrid::neighbors_of(std::size_t i, const coords_t &coords,
                                const BoundaryConditions *bc, double cutoff,
                                const AtomsCollector *collector,
                                std::vector<std::uint32_t> &out) const {
  out.clear();
  const double cut2 = cutoff * cutoff;
  const vec3_t ri = coords.row(static_cast<Eigen::Index>(i)).transpose();
  const Eigen::Vector3i ci = cell_coords(ri);
  const Eigen::Vector3i strd = strides();

  // Per-axis list of cell indices to visit. For >=3 cells the +/-1 stencil with
  // wrap gives three distinct cells; for 1 or 2 cells we scan every cell along
  // that axis (a +/-1 stencil would alias and double-visit the same cell).
  Eigen::Matrix3i visit;  // row a = axis a, columns 0..nvisit(a)-1
  Eigen::Vector3i nvisit;
  for (int a = 0; a < 3; ++a) {
    if (n_(a) >= 3) {
      visit(a, 0) = (ci(a) - 1 + n_(a)) % n_(a);
      visit(a, 1) = ci(a);
      visit(a, 2) = (ci(a) + 1) % n_(a);
      nvisit(a) = 3;
    } else {
      for (int c = 0; c < n_(a); ++c) {
        visit(a, c) = c;
      }
      nvisit(a) = n_(a);
    }
  }

  for (int ax = 0; ax < nvisit(0); ++ax) {
    for (int ay = 0; ay < nvisit(1); ++ay) {
      for (int az = 0; az < nvisit(2); ++az) {
        const Eigen::Vector3i cell{visit(0, ax), visit(1, ay), visit(2, az)};
        const std::size_t flat = static_cast<std::size_t>(cell.dot(strd));
        for (const std::uint32_t j : cell_atoms_[flat]) {
          if (static_cast<std::size_t>(j) == i) {
            continue;
          }
          if (collector && collector->absent(static_cast<std::size_t>(j))) {
            continue;
          }
          vec3_t d =
              (coords.row(static_cast<Eigen::Index>(j)).transpose() - ri);
          if (bc) {
            d = bc_min_image(*bc, d);
          }
          if (d.squaredNorm() <= cut2) {
            out.push_back(j);
          }
        }
      }
    }
  }
}

} // namespace RMC
