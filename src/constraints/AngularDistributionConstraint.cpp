#include <RMC/constraints/AngularDistributionConstraint.hpp>

#include <boost/assert.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <numbers>
#include <ranges>
#include <span>
#include <utility>
#include <vector>

namespace RMC {

void accumulate_angle_histogram(vec_t &hist, const coords_t &coords,
                                const BoundaryConditions *bc,
                                const std::vector<uint8_t> &elem_id, int n_types,
                                double max_dis, int n_bins,
                                const AtomsCollector *collector) {
  const Eigen::Index N = coords.rows();
  const int n_leg_pairs = adf_n_leg_pairs(n_types);
  const int n_cols = n_types * n_leg_pairs;
  hist.setZero();
  if (N < 3 || n_bins <= 0 || n_cols == 0) {
    return;
  }
  const double max2 = max_dis * max_dis;
  const double inv_bw = static_cast<double>(n_bins) / std::numbers::pi;

  // Neighbour adjacency within the cutoff (brute force, minimum image). O(N²);
  // see the header note on the planned cell-list / incremental optimisation.
  std::vector<std::vector<std::uint32_t>> adj(static_cast<std::size_t>(N));
  for (Eigen::Index i = 0; i < N; ++i) {
    if (collector && collector->absent(static_cast<std::size_t>(i))) {
      continue; // a removed atom joins no neighbour list, so forms no angle
    }
    for (Eigen::Index j = i + 1; j < N; ++j) {
      if (collector && collector->absent(static_cast<std::size_t>(j))) {
        continue;
      }
      vec3_t d = (coords.row(j) - coords.row(i)).transpose();
      if (bc) {
        d = bc_min_image(*bc, d);
      }
      if (d.squaredNorm() <= max2) {
        adj[static_cast<std::size_t>(i)].push_back(
            static_cast<std::uint32_t>(j));
        adj[static_cast<std::size_t>(j)].push_back(
            static_cast<std::uint32_t>(i));
      }
    }
  }

  // Every angle j–i–k for atom i's neighbour pairs (j < k in the adjacency).
  for (Eigen::Index i = 0; i < N; ++i) {
    const auto &nb = adj[static_cast<std::size_t>(i)];
    const int a = elem_id[static_cast<std::size_t>(i)];
    const std::size_t deg = nb.size();
    for (std::size_t aa = 0; aa + 1 < deg; ++aa) {
      const std::uint32_t j = nb[aa];
      vec3_t v1 = (coords.row(j) - coords.row(i)).transpose();
      if (bc) {
        v1 = bc_min_image(*bc, v1);
      }
      const double n1 = v1.norm();
      for (std::size_t bb = aa + 1; bb < deg; ++bb) {
        const std::uint32_t k = nb[bb];
        vec3_t v2 = (coords.row(k) - coords.row(i)).transpose();
        if (bc) {
          v2 = bc_min_image(*bc, v2);
        }
        const double cos_a = v1.dot(v2) / (n1 * v2.norm() + 1e-30);
        const double angle = std::acos(std::clamp(cos_a, -1.0, 1.0));
        int bin = static_cast<int>(angle * inv_bw);
        bin = std::clamp(bin, 0, n_bins - 1); // angle == π lands in the last bin
        int p = elem_id[j];
        int q = elem_id[k];
        if (p > q) {
          std::swap(p, q);
        }
        const int col = a * n_leg_pairs + adf_leg_pair_index(p, q, n_types);
        hist(static_cast<Eigen::Index>(bin) * n_cols + col) += 1.0;
      }
    }
  }
}

void accumulate_moved_angles(vec_t &hist, const coords_t &coords,
                             const BoundaryConditions *bc,
                             const std::vector<uint8_t> &elem_id, int n_types,
                             double max_dis, int n_bins,
                             std::span<const std::size_t> moved,
                             const NeighborGrid &grid,
                             const AtomsCollector *collector) {
  const Eigen::Index N = coords.rows();
  const int n_leg_pairs = adf_n_leg_pairs(n_types);
  const int n_cols = n_types * n_leg_pairs;
  // `hist` is zeroed by the caller (matches accumulate_moved_pairs).
  if (N < 1 || n_bins <= 0 || n_cols == 0 || moved.empty()) {
    return;
  }
  const double inv_bw = static_cast<double>(n_bins) / std::numbers::pi;

  // O(1) "is this atom in `moved`" test, sized to N and reset on touched slots
  // only (same trick as moved_pos in accumulate_moved_pairs). Used both to seed
  // the candidate-centre set and as the inclusion predicate.
  thread_local std::vector<char> is_moved, center_seen;
  if (is_moved.size() < static_cast<std::size_t>(N)) {
    is_moved.assign(static_cast<std::size_t>(N), 0);
    center_seen.assign(static_cast<std::size_t>(N), 0);
  }
  for (const std::size_t m : moved) {
    is_moved[m] = 1;
  }

  // Candidate centres C = moved ∪ neighbours(moved). Iterating by apex over C
  // visits every affected angle exactly once (the apex is its unique key).
  thread_local std::vector<std::size_t> centers;
  centers.clear();
  thread_local std::vector<std::uint32_t> nbr;
  const auto add_center = [&](std::size_t c) {
    if (!center_seen[c]) {
      center_seen[c] = 1;
      centers.push_back(c);
    }
  };
  for (const std::size_t m : moved) {
    add_center(m); // m as apex (skipped below if absent)
    // Seed from m's coordinate row even if m is absent — its former neighbours
    // must still be revisited so its angles get subtracted, not re-added.
    grid.neighbors_of(m, coords, bc, max_dis, collector, nbr);
    for (const std::uint32_t c : nbr) {
      add_center(c);
    }
  }

  // Enumerate angles j–i–k by apex i, mirroring accumulate_angle_histogram, but
  // counting a pair only when {i, j, k} ∩ moved ≠ ∅.
  for (const std::size_t i : centers) {
    if (collector && collector->absent(i)) {
      continue;
    }
    grid.neighbors_of(i, coords, bc, max_dis, collector, nbr);
    const int a = elem_id[i];
    const std::size_t deg = nbr.size();
    const bool apex_moved = is_moved[i] != 0;
    for (std::size_t aa = 0; aa + 1 < deg; ++aa) {
      const std::uint32_t j = nbr[aa];
      vec3_t v1 = (coords.row(j) - coords.row(static_cast<Eigen::Index>(i)))
                      .transpose();
      if (bc) {
        v1 = bc_min_image(*bc, v1);
      }
      const double n1 = v1.norm();
      const bool j_moved = is_moved[j] != 0;
      for (std::size_t bb = aa + 1; bb < deg; ++bb) {
        const std::uint32_t k = nbr[bb];
        if (!apex_moved && !j_moved && !is_moved[k]) {
          continue; // angle untouched by the move — excluded from the delta
        }
        vec3_t v2 = (coords.row(k) - coords.row(static_cast<Eigen::Index>(i)))
                        .transpose();
        if (bc) {
          v2 = bc_min_image(*bc, v2);
        }
        const double cos_a = v1.dot(v2) / (n1 * v2.norm() + 1e-30);
        const double angle = std::acos(std::clamp(cos_a, -1.0, 1.0));
        int bin = static_cast<int>(angle * inv_bw);
        bin = std::clamp(bin, 0, n_bins - 1);
        int p = elem_id[j];
        int q = elem_id[k];
        if (p > q) {
          std::swap(p, q);
        }
        const int col = a * n_leg_pairs + adf_leg_pair_index(p, q, n_types);
        hist(static_cast<Eigen::Index>(bin) * n_cols + col) += 1.0;
      }
    }
  }

  // Reset the touched mask slots, keeping the thread_local clear for next call.
  for (const std::size_t m : moved) {
    is_moved[m] = 0;
  }
  for (const std::size_t c : centers) {
    center_seen[c] = 0;
  }
}

void AngularDistributionConstraint::set_experimental_data(const mat_t &data) {
  BOOST_ASSERT_MSG(data.cols() >= 2,
                   "ADF: data needs an angle column and >= 1 value column");
  exp_theta_ = data.col(0);
  n_bins_ = static_cast<int>(data.rows());
  exp_values_ = data.rightCols(data.cols() - 1); // flattened in initialise()
}

void AngularDistributionConstraint::initialise() {
  if (initialised_) {
    return;
  }
  initialised_ = true;

  // Species ids by *sorted* symbol — canonical and independent of atom order, so
  // a target written by --adf-compute lines up column-for-column here.
  std::map<std::string, std::uint8_t> id_of;
  for (const auto &e : elements_) {
    id_of.emplace(e, 0);
  }
  std::uint8_t next = 0;
  for (auto &kv : id_of) {
    kv.second = next++;
  }
  n_types_ = static_cast<int>(id_of.size());
  n_leg_pairs_ = adf_n_leg_pairs(n_types_);
  n_cols_ = n_types_ * n_leg_pairs_;
  hist_len_ = n_bins_ * n_cols_;

  species_.assign(static_cast<std::size_t>(n_types_), {});
  for (const auto &[sym, id] : id_of) {
    species_[id] = sym;
  }

  elem_id_.resize(elements_.size());
  std::vector<int> count(static_cast<std::size_t>(n_types_), 0);
  for (const auto [i, e] : std::views::enumerate(elements_)) {
    const std::uint8_t id = id_of.at(e);
    elem_id_[static_cast<std::size_t>(i)] = id;
    ++count[id];
  }

  // Per-column 1/(N_a·N_p·N_q) weight (MAST's per-triplet divisor). A column with
  // an absent species gets 0 so it never contributes.
  col_inv_count_ = vec_t::Zero(n_cols_);
  for (int a = 0; a < n_types_; ++a) {
    for (int p = 0; p < n_types_; ++p) {
      for (int q = p; q < n_types_; ++q) {
        const int col = a * n_leg_pairs_ + adf_leg_pair_index(p, q, n_types_);
        const double denom = static_cast<double>(count[static_cast<std::size_t>(a)]) *
                             static_cast<double>(count[static_cast<std::size_t>(p)]) *
                             static_cast<double>(count[static_cast<std::size_t>(q)]);
        col_inv_count_(col) = denom > 0.0 ? 1.0 / denom : 0.0;
      }
    }
  }

  // Flatten + validate the experimental target.
  BOOST_ASSERT_MSG(
      static_cast<int>(exp_values_.cols()) == n_cols_,
      "ADF target column count != adf_n_cols(n_types); column order is "
      "central-id outer, leg-pair inner");
  // Row-major flatten of the (n_bins_ × n_cols_) target into the flat layout
  // the histogram uses (entry b·n_cols_+c ↦ exp_values_(b, c)).
  exp_data_.resize(hist_len_);
  Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                           Eigen::RowMajor>>(exp_data_.data(), n_bins_,
                                             n_cols_) = exp_values_;

  // Single-frame buffers (multi-frame ones are allocated in set_n_frames()).
  single_hist_ = vec_t::Zero(hist_len_);
  single_hist_current_ = false;
  saved_frame_hist_ = vec_t::Zero(hist_len_);
  saved_moved_delta_ = vec_t::Zero(hist_len_);
  scratch_delta_ = vec_t::Zero(hist_len_);
  incremental_ready_ = false;
  computed_.resize(hist_len_);
  smooth_scratch_.resize(hist_len_);
}

void AngularDistributionConstraint::set_n_frames(std::size_t n) {
  BOOST_ASSERT_MSG(hist_len_ > 0, "call initialise() before set_n_frames");
  n_frames_ = n;
  frame_hists_.assign(n, vec_t::Zero(hist_len_));
  frame_hist_current_.assign(n, false);
  frame_grids_.assign(n, NeighborGrid{});
  sum_hist_ = vec_t::Zero(hist_len_);
  saved_frame_hist_ = vec_t::Zero(hist_len_);
  incremental_ready_ = false;
}

void AngularDistributionConstraint::rollback_frame() noexcept {
  if (n_frames_ > 1) {
    sum_hist_ -= frame_hists_[active_frame_];
    frame_hists_[active_frame_] = saved_frame_hist_;
    sum_hist_ += saved_frame_hist_;
    frame_hist_current_[active_frame_] = true;
  } else {
    single_hist_ = saved_frame_hist_;
    single_hist_current_ = true;
  }
  incremental_ready_ = false;
  // Undo any cell relocation/removal applied by the (rejected) after-move.
  active_grid().restore_cells();
}

void AngularDistributionConstraint::commit_frame() noexcept {
  // The after-move grid mutation is now permanent; drop the rollback snapshot.
  active_grid().clear_saved();
  if (resync_every_ > 0 && ++accepts_since_resync_ >= resync_every_) {
    accepts_since_resync_ = 0;
    if (n_frames_ > 1) {
      frame_hist_current_[active_frame_] = false; // force a full rebuild
    } else {
      single_hist_current_ = false;
    }
  }
}

void AngularDistributionConstraint::normalise_and_smooth(
    const coords_t &coords) const {
  // Per-column triplet normalisation. Under the scale-invariant metric the
  // global `inc` cancels in the residual, so only the relative per-column weight
  // matters; in MAST-parity mode the full volume normalisation is applied.
  const double Nat = static_cast<double>(coords.rows());
  const double inc =
      scale_invariant_
          ? 1.0
          : (bc_ ? bc_volume(*bc_) : 1.0) * Nat / std::numbers::pi *
                static_cast<double>(n_bins_);
  // Per-column scaling: `computed_` is a row-major (bin × column) matrix in a
  // flat vector, so a column-tiled copy of the per-column weight aligns
  // element-for-element with it.
  computed_.array() *= inc * col_inv_count_.replicate(n_bins_, 1).array();

  // Per-column boxcar smoothing (shrinking window at the edges), 2 passes —
  // matching MAST's smooth(hist, 2, 2). Never bleeds across column boundaries.
  if (smooth_range_ > 0 && n_bins_ > 1) {
    using RowMajMat =
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    const int range = smooth_range_;
    for (int pass = 0; pass < 2; ++pass) {
      Eigen::Map<const RowMajMat> C(computed_.data(), n_bins_, n_cols_);
      Eigen::Map<RowMajMat> Cs(smooth_scratch_.data(), n_bins_, n_cols_);
      for (int c = 0; c < n_cols_; ++c) {
        for (int b = 0; b < n_bins_; ++b) {
          const int lo = std::max(0, b - range);
          const int hi = std::min(n_bins_ - 1, b + range);
          Cs(b, c) = C.col(c).segment(lo, hi - lo + 1).sum() /
                     static_cast<double>(hi - lo + 1);
        }
      }
      computed_.swap(smooth_scratch_);
    }
  }
}

double AngularDistributionConstraint::compute_error(
    const coords_t &coords, std::span<const std::size_t> moved) const {
  // The before-move call (incremental_ready_ == false) snapshots the histogram
  // and stages the moved-vertex angle delta at the OLD coords; the after-move
  // call (incremental_ready_ == true) recomputes that delta at the NEW coords
  // and patches new_hist = saved − D_old + D_new. A full-evaluation call (empty
  // `moved`) always takes the before path and rebuilds on demand. Mirrors
  // PairFunctionConstraint::compute_error.
  if (n_frames_ > 1) {
    NeighborGrid &grid = frame_grids_[active_frame_];
    if (incremental_ready_ && !moved.empty()) {
      // After-move: sync the grid to the new coords, then patch with the delta.
      for (const std::size_t m : moved) {
        if (collector_ && collector_->absent(m)) {
          grid.remove(m);
        } else {
          grid.relocate(m, coords);
        }
      }
      scratch_delta_.setZero();
      accumulate_moved_angles(scratch_delta_, coords, bc_, elem_id_, n_types_,
                              max_dis_, n_bins_, moved, grid, collector_);
      vec_t new_frame_hist =
          saved_frame_hist_ - saved_moved_delta_ + scratch_delta_;
      sum_hist_ += new_frame_hist - frame_hists_[active_frame_];
      frame_hists_[active_frame_] = std::move(new_frame_hist);
      incremental_ready_ = false;
    } else {
      // Before-move (or full evaluation).
      saved_frame_hist_ = frame_hists_[active_frame_];
      if (!frame_hist_current_[active_frame_]) {
        vec_t tmp = vec_t::Zero(hist_len_);
        accumulate_angle_histogram(tmp, coords, bc_, elem_id_, n_types_,
                                   max_dis_, n_bins_, collector_);
        grid.build(coords, bc_, max_dis_, collector_);
        sum_hist_ += tmp - saved_frame_hist_;
        frame_hists_[active_frame_] = tmp;
        saved_frame_hist_ = std::move(tmp);
        frame_hist_current_[active_frame_] = true;
      }
      if (!moved.empty()) {
        grid.save_cells(moved);
        saved_moved_delta_.setZero();
        accumulate_moved_angles(saved_moved_delta_, coords, bc_, elem_id_,
                                n_types_, max_dis_, n_bins_, moved, grid,
                                collector_);
        incremental_ready_ = true;
      }
    }
    computed_ = sum_hist_ / static_cast<double>(n_frames_);
  } else {
    if (incremental_ready_ && !moved.empty()) {
      for (const std::size_t m : moved) {
        if (collector_ && collector_->absent(m)) {
          grid_.remove(m);
        } else {
          grid_.relocate(m, coords);
        }
      }
      scratch_delta_.setZero();
      accumulate_moved_angles(scratch_delta_, coords, bc_, elem_id_, n_types_,
                              max_dis_, n_bins_, moved, grid_, collector_);
      single_hist_ = saved_frame_hist_ - saved_moved_delta_ + scratch_delta_;
      incremental_ready_ = false;
    } else {
      saved_frame_hist_ = single_hist_;
      if (!single_hist_current_) {
        single_hist_ = vec_t::Zero(hist_len_);
        accumulate_angle_histogram(single_hist_, coords, bc_, elem_id_,
                                   n_types_, max_dis_, n_bins_, collector_);
        grid_.build(coords, bc_, max_dis_, collector_);
        saved_frame_hist_ = single_hist_;
        single_hist_current_ = true;
      }
      if (!moved.empty()) {
        grid_.save_cells(moved);
        saved_moved_delta_.setZero();
        accumulate_moved_angles(saved_moved_delta_, coords, bc_, elem_id_,
                                n_types_, max_dis_, n_bins_, moved, grid_,
                                collector_);
        incremental_ready_ = true;
      }
    }
    computed_ = single_hist_;
  }

  normalise_and_smooth(coords);

  if (scale_invariant_) {
    const double denom = computed_.squaredNorm();
    const double scale =
        (denom > 1e-30) ? computed_.dot(exp_data_) / denom : 1.0;
    return (scale * computed_ - exp_data_).squaredNorm();
  }
  return (computed_ - exp_data_).squaredNorm();
}

} // namespace RMC
