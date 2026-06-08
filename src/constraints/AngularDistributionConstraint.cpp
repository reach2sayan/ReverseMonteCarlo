#include <RMC/constraints/AngularDistributionConstraint.hpp>

#include <boost/assert.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <numbers>
#include <ranges>
#include <utility>
#include <vector>

namespace RMC {

void accumulate_angle_histogram(vec_t &hist, const coords_t &coords,
                                const BoundaryConditions *bc,
                                const std::vector<uint8_t> &elem_id, int n_types,
                                double max_dis, int n_bins) {
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
    for (Eigen::Index j = i + 1; j < N; ++j) {
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
  exp_data_ = vec_t::Zero(hist_len_);
  for (int b = 0; b < n_bins_; ++b) {
    for (int c = 0; c < n_cols_; ++c) {
      exp_data_(static_cast<Eigen::Index>(b) * n_cols_ + c) = exp_values_(b, c);
    }
  }

  // Single-frame buffers (multi-frame ones are allocated in set_n_frames()).
  single_hist_ = vec_t::Zero(hist_len_);
  saved_frame_hist_ = vec_t::Zero(hist_len_);
  computed_.resize(hist_len_);
  smooth_scratch_.resize(hist_len_);
}

void AngularDistributionConstraint::set_n_frames(Constraint::Token,
                                                 std::size_t n) {
  BOOST_ASSERT_MSG(hist_len_ > 0, "call initialise() before set_n_frames");
  n_frames_ = n;
  frame_hists_.assign(n, vec_t::Zero(hist_len_));
  sum_hist_ = vec_t::Zero(hist_len_);
  saved_frame_hist_ = vec_t::Zero(hist_len_);
}

void AngularDistributionConstraint::rollback_frame() noexcept {
  if (n_frames_ > 1) {
    sum_hist_ -= frame_hists_[active_frame_];
    frame_hists_[active_frame_] = saved_frame_hist_;
    sum_hist_ += saved_frame_hist_;
  } else {
    single_hist_ = saved_frame_hist_;
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
  for (int b = 0; b < n_bins_; ++b) {
    for (int c = 0; c < n_cols_; ++c) {
      computed_(static_cast<Eigen::Index>(b) * n_cols_ + c) *=
          inc * col_inv_count_(c);
    }
  }

  // Per-column boxcar smoothing (shrinking window at the edges), 2 passes —
  // matching MAST's smooth(hist, 2, 2). Never bleeds across column boundaries.
  if (smooth_range_ > 0 && n_bins_ > 1) {
    const int range = smooth_range_;
    for (int pass = 0; pass < 2; ++pass) {
      for (int c = 0; c < n_cols_; ++c) {
        for (int b = 0; b < n_bins_; ++b) {
          const int lo = std::max(0, b - range);
          const int hi = std::min(n_bins_ - 1, b + range);
          double sum = 0.0;
          for (int t = lo; t <= hi; ++t) {
            sum += computed_(static_cast<Eigen::Index>(t) * n_cols_ + c);
          }
          smooth_scratch_(static_cast<Eigen::Index>(b) * n_cols_ + c) =
              sum / static_cast<double>(hi - lo + 1);
        }
      }
      computed_.swap(smooth_scratch_);
    }
  }
}

double AngularDistributionConstraint::compute_error(
    const coords_t &coords, std::span<const std::size_t>) const {
  if (n_frames_ > 1) {
    saved_frame_hist_ = frame_hists_[active_frame_];
    vec_t tmp = vec_t::Zero(hist_len_);
    accumulate_angle_histogram(tmp, coords, bc_, elem_id_, n_types_, max_dis_,
                               n_bins_);
    sum_hist_ += tmp - frame_hists_[active_frame_];
    frame_hists_[active_frame_] = std::move(tmp);
    computed_ = sum_hist_ / static_cast<double>(n_frames_);
  } else {
    saved_frame_hist_ = single_hist_;
    accumulate_angle_histogram(single_hist_, coords, bc_, elem_id_, n_types_,
                               max_dis_, n_bins_);
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
