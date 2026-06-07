#include <RMC/constraints/PairHistogram.hpp>
#include <RMC/core/Parallel.hpp>
#include <cmath>
#include <numbers>
#include <ranges>
#include <unordered_map>
#include <variant>
#include <vector>

#if defined(RMC_USE_TBB)
#include <numeric>
#endif

namespace RMC {

namespace {

// Bin one distance `d` (weight `w`) into `hist`, reproducing boost::histogram's
// regular-axis mapping  bin = floor((d − r_min)/Δr)  but without the per-insert
// allocation/variant dispatch the profile showed (~10% of accumulate_*). A
// distance outside [r_min, r_max) fell into the under/overflow bins, which were
// never copied out — so we simply drop it.
FORCE_INLINE void bin_distance(vec_t &hist, double d, double w, double r_min,
                               double inv_dr, int n_bins) {
  const double f = (d - r_min) * inv_dr;
  if (f >= 0.0 && f < static_cast<double>(n_bins)) {
    hist(static_cast<int>(f)) += w;
  }
}

// Squared distances from atom `k` to every atom, written into `d2` (length N).
// coords is RowMajor AoS, so the columns X/Y/Z are passed in already copied to
// contiguous SoA arrays — that lets Eigen vectorise the whole column at once.
// Periodic boundaries apply the minimum image as 3×N matrix ops (Eigen blocks
// the small 3×3 · 3×N products); infinite/no-BC takes the cheap direct form.
FORCE_INLINE void squared_distances(Eigen::ArrayXd &d2, const Eigen::ArrayXd &X,
                                    const Eigen::ArrayXd &Y,
                                    const Eigen::ArrayXd &Z, Eigen::Index k,
                                    const PeriodicBC *pbc,
                                    Eigen::Matrix3Xd &delta_scratch,
                                    Eigen::Matrix3Xd &frac_scratch) {
  if (pbc == nullptr) {
    d2 = (X - X(k)).square() + (Y - Y(k)).square() + (Z - Z(k)).square();
    return;
  }
  const Eigen::Index N = X.size();
  delta_scratch.resize(3, N);
  delta_scratch.row(0) = (X - X(k)).matrix().transpose();
  delta_scratch.row(1) = (Y - Y(k)).matrix().transpose();
  delta_scratch.row(2) = (Z - Z(k)).matrix().transpose();
  frac_scratch.noalias() = pbc->inv_box() * delta_scratch;
  frac_scratch -= frac_scratch.array().round().matrix();
  delta_scratch.noalias() = pbc->box() * frac_scratch;
  d2 = delta_scratch.colwise().squaredNorm().transpose().array();
}

} // namespace

void accumulate_pair_histogram(vec_t &hist, const coords_t &coords,
                               const BoundaryConditions *bc,
                               const std::vector<uint8_t> &elem_id,
                               const PairWeightMatrix &weights,
                               double r_min, double r_max, int n_bins,
                               std::span<const std::size_t> molecule_ids,
                               bool exclude_intra) {
  const Eigen::Index N = coords.rows();
  const bool filter_intra = exclude_intra && !molecule_ids.empty();
  const double inv_dr = static_cast<double>(n_bins) / (r_max - r_min);
  const PeriodicBC *pbc = bc ? std::get_if<PeriodicBC>(bc) : nullptr;

  // SoA columns (contiguous) so the per-row distance fan-out vectorises.
  Eigen::ArrayXd X = coords.col(0), Y = coords.col(1), Z = coords.col(2);
  hist.setZero();

#if defined(RMC_USE_TBB)
  // TBB thread pool — no per-call spawn cost. Each row i bins into its own
  // private vector; they are summed at the end.
  std::vector<Eigen::Index> rows(static_cast<std::size_t>(N - 1));
  std::iota(rows.begin(), rows.end(), Eigen::Index{0});
  std::vector<vec_t> partial(static_cast<std::size_t>(N - 1),
                             vec_t::Zero(n_bins));
  parallel::for_each(rows.begin(), rows.end(), [&](Eigen::Index i) {
    vec_t &local = partial[static_cast<std::size_t>(i)];
    Eigen::ArrayXd d2;
    Eigen::Matrix3Xd delta_scratch, frac_scratch;
    squared_distances(d2, X, Y, Z, i, pbc, delta_scratch, frac_scratch);
    for (Eigen::Index j = i + 1; j < N; ++j) {
      if (filter_intra && molecule_ids[static_cast<std::size_t>(i)] ==
                              molecule_ids[static_cast<std::size_t>(j)]) {
        continue;
      }
      const double w = weights.weight_of(elem_id, static_cast<std::size_t>(i),
                                         static_cast<std::size_t>(j));
      bin_distance(local, std::sqrt(d2(j)), 2.0 * w, r_min, inv_dr, n_bins);
    }
  });
  for (const vec_t &p : partial) {
    hist += p;
  }

#else
  Eigen::ArrayXd d2;
  Eigen::Matrix3Xd delta_scratch, frac_scratch;
  for (Eigen::Index i = 0; i < N; ++i) {
    squared_distances(d2, X, Y, Z, i, pbc, delta_scratch, frac_scratch);
    for (Eigen::Index j = i + 1; j < N; ++j) {
      if (filter_intra && molecule_ids[static_cast<std::size_t>(i)] ==
                              molecule_ids[static_cast<std::size_t>(j)]) {
        continue;
      }
      const double w = weights.weight_of(elem_id, static_cast<std::size_t>(i),
                                         static_cast<std::size_t>(j));
      bin_distance(hist, std::sqrt(d2(j)), 2.0 * w, r_min, inv_dr, n_bins);
    }
  }
#endif
}

void accumulate_moved_pairs(
    vec_t &hist, const coords_t &coords, const BoundaryConditions *bc,
    const std::vector<uint8_t> &elem_id, const PairWeightMatrix &weights,
    double r_min, double r_max, int n_bins, std::span<const std::size_t> moved,
    std::span<const std::size_t> molecule_ids, bool exclude_intra) {
  const Eigen::Index N = coords.rows();
  if (N == 0 || moved.empty()) {
    return;
  }
  const bool filter_intra = exclude_intra && !molecule_ids.empty();
  // #2: inline binning — write straight into `hist`, no boost::histogram
  // allocation and no per-insert variant dispatch.
  const double inv_dr = static_cast<double>(n_bins) / (r_max - r_min);

  // O(1) "position of atom in moved" lookup, replacing the O(K) ranges::contains
  // scan that ran per (moved, j) pair. moved_pos_[atom] = its index in `moved`
  // (kSentinel ⇒ not moved). Sized to N, only the K moved slots are touched and
  // reset, so setup/teardown is O(K) not O(N).
  constexpr std::size_t kSentinel = static_cast<std::size_t>(-1);
  thread_local std::vector<std::size_t> moved_pos;
  if (moved_pos.size() < static_cast<std::size_t>(N)) {
    moved_pos.assign(static_cast<std::size_t>(N), kSentinel);
  }
  for (const auto [mk, k] : std::views::enumerate(moved)) {
    moved_pos[k] = static_cast<std::size_t>(mk);
  }

  // #3 + Eigen vectorisation: coords is RowMajor AoS, so column access is
  // strided. Copy the three columns into contiguous SoA arrays once; every
  // moved atom then gets its squared distances to all atoms in a single
  // vectorised Eigen expression (see squared_distances()).
  thread_local Eigen::ArrayXd X, Y, Z, d2;
  X = coords.col(0);
  Y = coords.col(1);
  Z = coords.col(2);
  thread_local Eigen::Matrix3Xd delta_scratch, frac_scratch;
  const PeriodicBC *pbc = bc ? std::get_if<PeriodicBC>(bc) : nullptr;

  for (const auto [mk, k] : std::views::enumerate(moved)) {
    const auto kk = static_cast<std::size_t>(k);
    squared_distances(d2, X, Y, Z, static_cast<Eigen::Index>(k), pbc,
                      delta_scratch, frac_scratch);
    for (Eigen::Index jj = 0; jj < N; ++jj) {
      const std::size_t j = static_cast<std::size_t>(jj);
      if (j == kk) {
        continue;
      }
      // Avoid double-counting pairs where both atoms are in `moved`:
      // count (k,j) only when k appears before j in the moved array, i.e. skip
      // when j is also moved and appears earlier (position < mk).
      const bool already_counted =
          moved_pos[j] != kSentinel &&
          moved_pos[j] < static_cast<std::size_t>(mk);
      if (already_counted ||
          (filter_intra && molecule_ids[kk] == molecule_ids[j])) {
        continue;
      }

      const double w = weights.weight_of(elem_id, kk, j);
      bin_distance(hist, std::sqrt(d2(jj)), 2.0 * w, r_min, inv_dr, n_bins);
    }
  }

  // Reset only the touched slots, keeping the thread_local clear for next call.
  for (const std::size_t k : moved) {
    moved_pos[k] = kSentinel;
  }
}

void PairConstraintBase::set_n_frames(std::size_t n) {
  BOOST_ASSERT_MSG(n_bins_ > 0,
                   "call set_experimental_data before set_n_frames");
  n_frames_ = n;
  frame_hists_.assign(n, vec_t::Zero(n_bins_));
  frame_hist_current_.assign(n, false);
  sum_hist_ = vec_t::Zero(n_bins_);
  saved_frame_hist_.resize(n_bins_);
  incremental_ready_ = false;
}

void PairConstraintBase::rollback_frame() noexcept {
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
}

void PairConstraintBase::set_experimental_data(const mat_t &data) {
  BOOST_ASSERT_MSG(data.cols() >= 2, "PairConstraint: need 2-column r/data");
  const Eigen::Index N = data.rows();
  exp_r_ = data.col(0);
  exp_data_ = data.col(1);
  r_min_ = exp_r_(0);
  r_max_ = exp_r_(N - 1);
  bin_width_ = (N > 1) ? (exp_r_(1) - exp_r_(0)) : 0.1;
  n_bins_ = static_cast<int>(N);
  computed_.resize(N);
  // Pre-size the reused per-step delta buffers so the hot path can setZero()
  // in place instead of allocating a fresh vec_t::Zero(n_bins_) each step.
  saved_moved_delta_.resize(N);
  scratch_delta_.resize(N);
}

void PairConstraintBase::initialise() {
  if (initialised_) {
    return;
  }
  initialised_ = true;
  const auto idx = Eigen::ArrayXd::LinSpaced(n_bins_, 0, n_bins_ - 1);
  const auto r_lo = r_min_ + idx * bin_width_;
  const auto r_hi = r_lo + bin_width_;
  shell_vols_ = (4.0 * std::numbers::pi / 3.0) * (r_hi.cube() - r_lo.cube());

  std::unordered_map<std::string_view, uint8_t> name_to_id;
  uint8_t next_id = 0;
  elem_id_.resize(elements_.size());
  for (const auto [i, elem] : std::views::enumerate(elements_)) {
    auto [it, ins] = name_to_id.try_emplace(elem, next_id);
    if (ins) {
      ++next_id;
    }
    elem_id_[i] = it->second;
  }
  // Build the dense n_types×n_types weight matrix. Default 1.0 so unset pairs
  // behave exactly like the old flat_map miss. `weighted` short-circuits the
  // whole lookup when no weights were set (the common unweighted case).
  const int n_types = static_cast<int>(next_id);
  weight_table_.n_types = n_types;
  weight_table_.weighted = !weights_.empty();
  weight_table_.w.assign(
      static_cast<std::size_t>(n_types) * static_cast<std::size_t>(n_types),
      1.0);
  for (const auto &[key, w] : weights_) {
    auto ia = name_to_id.find(key.a);
    auto ib = name_to_id.find(key.b);
    if (ia != name_to_id.end() && ib != name_to_id.end()) {
      weight_table_.w[static_cast<std::size_t>(ia->second) * n_types +
                      ib->second] = w;
      weight_table_.w[static_cast<std::size_t>(ib->second) * n_types +
                      ia->second] = w;
    }
  }
}

} // namespace RMC
