#include <RMC/constraints/AngularDistributionConstraint.hpp>
#include <RMC/core/Parallel.hpp>
#include <RMC/core/SpeciesIndex.hpp>

#include <boost/assert.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <ranges>
#include <span>
#include <utility>
#include <vector>

namespace RMC {

namespace {

// Minimum-image displacement r_to − r_from.
FORCE_INLINE vec3_t displacement(const coords_t &c, const BoundaryConditions *bc,
                                 Eigen::Index from, Eigen::Index to) {
  vec3_t d = (c.row(to) - c.row(from)).transpose();
  if (bc) {
    d = bc->min_image(d);
  }
  return d;
}

// Adds 1 to `hist` for every angle j–i–k at apex i over the pairs (j before k)
// of its neighbour list `nb` that `keep(j, k)` accepts. The flat index is
// angle_bin·n_cols + (central=elem_id[i], legs=elem_id[j],elem_id[k]) column.
template <class Keep>
FORCE_INLINE void bin_apex_angles(vec_t &hist, const coords_t &coords,
                                  const BoundaryConditions *bc,
                                  const std::vector<uint8_t> &elem_id,
                                  int n_types, int n_bins, Eigen::Index i,
                                  std::span<const std::uint32_t> nb, Keep keep) {
  const int n_leg_pairs = adf_n_leg_pairs(n_types);
  const int n_cols = n_types * n_leg_pairs;
  const double inv_bw = static_cast<double>(n_bins) / std::numbers::pi;
  const int a = elem_id[static_cast<std::size_t>(i)];
  for (std::size_t aa = 0; aa + 1 < nb.size(); ++aa) {
    const std::uint32_t j = nb[aa];
    const vec3_t v1 = displacement(coords, bc, i, j);
    const double n1 = v1.norm();
    for (std::size_t bb = aa + 1; bb < nb.size(); ++bb) {
      const std::uint32_t k = nb[bb];
      if (!keep(j, k)) {
        continue;
      }
      const vec3_t v2 = displacement(coords, bc, i, k);
      const double cos_a = v1.dot(v2) / (n1 * v2.norm() + 1e-30);
      const double angle = std::acos(std::clamp(cos_a, -1.0, 1.0));
      // angle == π lands in the last bin.
      const int bin = std::clamp(static_cast<int>(angle * inv_bw), 0, n_bins - 1);
      // initializer_list overload: returns the pair by value (the two-argument
      // form would bind references to these temporaries).
      const auto [p, q] = std::minmax({int{elem_id[j]}, int{elem_id[k]}});
      const int col = a * n_leg_pairs + adf_leg_pair_index(p, q, n_types);
      hist(static_cast<Eigen::Index>(bin) * n_cols + col) += 1.0;
    }
  }
}

} // namespace

std::vector<AdfColumn> adf_columns(int n_types) {
  std::vector<AdfColumn> cols;
  cols.reserve(static_cast<std::size_t>(adf_n_cols(n_types)));
  const int n_leg_pairs = adf_n_leg_pairs(n_types);
  for (int a = 0; a < n_types; ++a) {
    for (int p = 0; p < n_types; ++p) {
      for (int q = p; q < n_types; ++q) {
        cols.push_back(
            {a, p, q, a * n_leg_pairs + adf_leg_pair_index(p, q, n_types)});
      }
    }
  }
  return cols;
}

void accumulate_angle_histogram(vec_t &hist, const coords_t &coords,
                                const BoundaryConditions *bc,
                                const std::vector<uint8_t> &elem_id,
                                int n_types, double max_dis, int n_bins,
                                const AtomsCollector *collector) {
  const Eigen::Index N = coords.rows();
  hist.setZero();
  if (N < 3 || n_bins <= 0 || adf_n_cols(n_types) == 0) {
    return;
  }
  const double max2 = max_dis * max_dis;
  const auto absent = [&](Eigen::Index a) {
    return collector && collector->absent(static_cast<std::size_t>(a));
  };
  const auto rows =
      std::views::iota(Eigen::Index{0}, N) | std::ranges::to<std::vector>();

  // Phase A: each row's neighbour list within the cutoff (brute force,
  // minimum image), written privately per row in ascending index order.
  std::vector<std::vector<std::uint32_t>> adj(static_cast<std::size_t>(N));
  parallel::for_each(rows.begin(), rows.end(), [&](Eigen::Index i) {
    if (absent(i)) {
      return; // a removed atom joins no neighbour list, so forms no angle
    }
    for (Eigen::Index j = 0; j < N; ++j) {
      if (j != i && !absent(j) &&
          displacement(coords, bc, i, j).squaredNorm() <= max2) {
        adj[static_cast<std::size_t>(i)].push_back(static_cast<std::uint32_t>(j));
      }
    }
  });

  // Phase B: every angle at apex i, binned into per-lane histograms.
  const vec_t zero = vec_t::Zero(hist.size());
  hist = parallel::parallel_sum(
      static_cast<std::size_t>(N), zero, [&](std::size_t i, vec_t &local) {
        bin_apex_angles(local, coords, bc, elem_id, n_types, n_bins,
                        static_cast<Eigen::Index>(i), adj[i],
                        [](std::uint32_t, std::uint32_t) { return true; });
      });
}

void accumulate_moved_angles(vec_t &hist, const coords_t &coords,
                             const BoundaryConditions *bc,
                             const std::vector<uint8_t> &elem_id, int n_types,
                             double max_dis, int n_bins,
                             std::span<const std::size_t> moved,
                             const NeighborGrid &grid, AngleScratch &s,
                             const AtomsCollector *collector) {
  const Eigen::Index N = coords.rows();
  // `hist` is zeroed by the caller (matches accumulate_moved_pairs).
  if (N < 1 || n_bins <= 0 || adf_n_cols(n_types) == 0 || moved.empty()) {
    return;
  }

  // O(1) membership masks sized to N; only touched slots are set and reset.
  if (s.is_moved.size() < static_cast<std::size_t>(N)) {
    s.is_moved.assign(static_cast<std::size_t>(N), 0);
    s.center_seen.assign(static_cast<std::size_t>(N), 0);
  }
  for (const std::size_t m : moved) {
    s.is_moved[m] = 1;
  }

  // Candidate apexes C = moved ∪ neighbours(moved): iterating by apex visits
  // every affected angle exactly once (the apex is its unique key).
  s.centers.clear();
  const auto add_center = [&](std::size_t c) {
    if (!s.center_seen[c]) {
      s.center_seen[c] = 1;
      s.centers.push_back(c);
    }
  };
  for (const std::size_t m : moved) {
    add_center(m); // m as apex (skipped below if absent)
    // Seed from m's coordinate row even if m is absent — its former neighbours
    // must still be revisited so its angles get subtracted, not re-added.
    grid.neighbors_of(m, coords, bc, max_dis, collector, s.nbr);
    std::ranges::for_each(s.nbr, add_center);
  }

  // Angles by apex, counted only when {i, j, k} ∩ moved ≠ ∅.
  for (const std::size_t i : s.centers) {
    if (collector && collector->absent(i)) {
      continue;
    }
    grid.neighbors_of(i, coords, bc, max_dis, collector, s.nbr);
    const bool apex_moved = s.is_moved[i] != 0;
    bin_apex_angles(hist, coords, bc, elem_id, n_types, n_bins,
                    static_cast<Eigen::Index>(i), s.nbr,
                    [&](std::uint32_t j, std::uint32_t k) {
                      return apex_moved || s.is_moved[j] || s.is_moved[k];
                    });
  }

  for (const std::size_t m : moved) {
    s.is_moved[m] = 0;
  }
  for (const std::size_t c : s.centers) {
    s.center_seen[c] = 0;
  }
}

void adf_smooth(vec_t &hist, vec_t &scratch, int n_bins, int n_cols,
                int range) {
  if (range <= 0 || n_bins <= 1) {
    return;
  }
  using RowMajMat =
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  scratch.resize(hist.size());
  for (int pass = 0; pass < 2; ++pass) {
    Eigen::Map<const RowMajMat> H(hist.data(), n_bins, n_cols);
    Eigen::Map<RowMajMat> Hs(scratch.data(), n_bins, n_cols);
    for (int b = 0; b < n_bins; ++b) {
      const int lo = std::max(0, b - range);
      const int hi = std::min(n_bins - 1, b + range);
      Hs.row(b) = H.middleRows(lo, hi - lo + 1).colwise().sum() /
                  static_cast<double>(hi - lo + 1);
    }
    hist.swap(scratch);
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

  // Species ids by *sorted* symbol — canonical and independent of atom order,
  // so a target written by --adf-compute lines up column-for-column here.
  const SpeciesIndex sp(elements_);
  species_ = sp.symbols;
  elem_id_ = sp.id;
  const auto &count = sp.count;
  n_types_ = static_cast<int>(sp.size());
  n_leg_pairs_ = adf_n_leg_pairs(n_types_);
  n_cols_ = n_types_ * n_leg_pairs_;
  hist_len_ = n_bins_ * n_cols_;

  // Per-column 1/(N_a·N_p·N_q) per-triplet divisor weight. A column with
  // an absent species gets 0 so it never contributes.
  col_inv_count_ = vec_t::Zero(n_cols_);
  for (const auto &[a, p, q, col] : adf_columns(n_types_)) {
    const double denom =
        static_cast<double>(count[static_cast<std::size_t>(a)]) *
        static_cast<double>(count[static_cast<std::size_t>(p)]) *
        static_cast<double>(count[static_cast<std::size_t>(q)]);
    col_inv_count_(col) = denom > 0.0 ? 1.0 / denom : 0.0;
  }

  // Flatten + validate the experimental target.
  BOOST_ASSERT_MSG(
      static_cast<int>(exp_values_.cols()) == n_cols_,
      "ADF target column count != adf_n_cols(n_types); column order is "
      "central-id outer, leg-pair inner");
  // Row-major flatten of the (n_bins_ × n_cols_) target into the flat layout
  // the histogram uses (entry b·n_cols_+c ↦ exp_values_(b, c)).
  exp_data_.resize(hist_len_);
  Eigen::Map<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
      exp_data_.data(), n_bins_, n_cols_) = exp_values_;

  // Pre-size the engine's reused per-step delta buffers and default to a single
  // frame so a constraint used standalone (outside the engine) is ready to
  // compute; the engine's later set_n_frames(N) overrides this.
  hist_.set_length(hist_len_);
  hist_.set_n_frames(1);
  frame_grids_.assign(1, NeighborGrid{});
  computed_.resize(hist_len_);
  smooth_scratch_.resize(hist_len_);
}

void AngularDistributionConstraint::set_n_frames(std::size_t n) {
  BOOST_ASSERT_MSG(hist_len_ > 0, "call initialise() before set_n_frames");
  hist_.set_n_frames(n);
  frame_grids_.assign(n, NeighborGrid{});
}

void AngularDistributionConstraint::rollback_frame() noexcept {
  // Undo any cell relocation/removal applied by the (rejected) after-move.
  hist_.rollback([this] { active_grid().restore_cells(); });
}

void AngularDistributionConstraint::commit_frame() noexcept {
  // The after-move grid mutation is now permanent; drop the rollback snapshot.
  active_grid().clear_saved();
  if (resync_every_ > 0 && ++accepts_since_resync_ >= resync_every_) {
    accepts_since_resync_ = 0;
    hist_.invalidate_active(); // force a full rebuild on next compute
  }
}

void AngularDistributionConstraint::normalise_and_smooth(
    const coords_t &coords) const {
  // Per-column triplet normalisation. Under the scale-invariant metric the
  // global `inc` cancels in the residual, so only the relative per-column
  // weight matters; in non-scale-invariant mode the full volume normalisation
  // is applied.
  const double Nat = static_cast<double>(coords.rows());
  const double inc = scale_invariant_
                         ? 1.0
                         : (bc_ ? bc_->volume() : 1.0) * Nat /
                               std::numbers::pi * static_cast<double>(n_bins_);
  // Per-column scaling: `computed_` is a row-major (bin × column) matrix in a
  // flat vector, so a column-tiled copy of the per-column weight aligns
  // element-for-element with it.
  computed_.array() *= inc * col_inv_count_.replicate(n_bins_, 1).array();
  adf_smooth(computed_, smooth_scratch_, n_bins_, n_cols_, smooth_range_);
}

double AngularDistributionConstraint::compute_error(
    const coords_t &coords, std::span<const std::size_t> moved) const {
  // The before-move call snapshots the histogram and stages the moved-vertex
  // angle delta at the OLD coords; the after-move call recomputes that delta at
  // the NEW coords and patches new = saved − D_old + D_new. The state machine
  // is shared (IncrementalHistogram); the hooks below carry the angle-specific
  // accumulation and the neighbour-grid relocation that the pair path lacks.
  hist_.update(
      computed_, moved,
      [&](vec_t &h) {
        accumulate_angle_histogram(h, coords, bc_, elem_id_, n_types_, max_dis_,
                                   n_bins_, collector_);
        active_grid().build(coords, bc_, max_dis_, collector_);
      },
      [&](vec_t &delta, std::span<const std::size_t> mv) {
        accumulate_moved_angles(delta, coords, bc_, elem_id_, n_types_,
                                max_dis_, n_bins_, mv, active_grid(), scratch_,
                                collector_);
      },
      [&](std::span<const std::size_t> mv) { active_grid().save_cells(mv); },
      [&](std::span<const std::size_t> mv) {
        // After-move: sync the grid to the new coords before recomputing.
        NeighborGrid &grid = active_grid();
        for (const std::size_t m : mv) {
          if (collector_ && collector_->absent(m)) {
            grid.remove(m);
          } else {
            grid.relocate(m, coords);
          }
        }
      });

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
