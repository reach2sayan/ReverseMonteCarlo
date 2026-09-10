#include <RMC/constraints/PairHistogram.hpp>
#include <RMC/core/Parallel.hpp>
#include <cmath>
#include <ranges>
#include <vector>

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
                                    const BoundaryConditions *pbc,
                                    Eigen::Matrix3Xd &delta,
                                    Eigen::Matrix3Xd &frac) {
  if (pbc == nullptr) {
    d2 = (X - X(k)).square() + (Y - Y(k)).square() + (Z - Z(k)).square();
    return;
  }
  const Eigen::Index N = X.size();
  delta.resize(3, N);
  delta.row(0) = (X - X(k)).matrix().transpose();
  delta.row(1) = (Y - Y(k)).matrix().transpose();
  delta.row(2) = (Z - Z(k)).matrix().transpose();
  frac.noalias() = pbc->inv_box() * delta;
  frac -= frac.array().round().matrix();
  delta.noalias() = pbc->box() * frac;
  d2 = delta.colwise().squaredNorm().transpose().array();
}

} // namespace

void accumulate_pair_histogram(vec_t &hist, const coords_t &coords,
                               const BoundaryConditions *bc,
                               const std::vector<uint8_t> &elem_id,
                               const PairWeightMatrix &weights,
                               double r_min, double r_max, int n_bins,
                               std::span<const std::size_t> molecule_ids,
                               bool exclude_intra,
                               const AtomsCollector *collector) {
  const Eigen::Index N = coords.rows();
  hist.setZero();
  if (N < 2) {
    return;
  }
  const bool filter_intra = exclude_intra && !molecule_ids.empty();
  const double inv_dr = static_cast<double>(n_bins) / (r_max - r_min);
  const BoundaryConditions *pbc = bc && bc->periodic() ? bc : nullptr;
  const auto absent = [&](Eigen::Index a) {
    return collector && collector->absent(static_cast<std::size_t>(a));
  };

  // SoA columns (contiguous) so the per-row distance fan-out vectorises.
  const Eigen::ArrayXd X = coords.col(0), Y = coords.col(1), Z = coords.col(2);
  // Rows i bin their pairs (i, j > i) into per-lane histograms.
  const vec_t zero = vec_t::Zero(n_bins);
  hist = parallel::parallel_sum(
      static_cast<std::size_t>(N - 1), zero, [&](std::size_t ii, vec_t &local) {
        const auto i = static_cast<Eigen::Index>(ii);
        if (absent(i)) {
          return; // every pair (i, ·) involves a removed atom
        }
        Eigen::ArrayXd d2;
        Eigen::Matrix3Xd delta, frac;
        squared_distances(d2, X, Y, Z, i, pbc, delta, frac);
        for (Eigen::Index j = i + 1; j < N; ++j) {
          const auto jj = static_cast<std::size_t>(j);
          if (absent(j) ||
              (filter_intra && molecule_ids[ii] == molecule_ids[jj])) {
            continue;
          }
          bin_distance(local, std::sqrt(d2(j)),
                       2.0 * weights.weight_of(elem_id, ii, jj), r_min, inv_dr,
                       n_bins);
        }
      });
}

void accumulate_moved_pairs(
    vec_t &hist, const coords_t &coords, const BoundaryConditions *bc,
    const std::vector<uint8_t> &elem_id, const PairWeightMatrix &weights,
    double r_min, double r_max, int n_bins, std::span<const std::size_t> moved,
    std::span<const std::size_t> molecule_ids, bool exclude_intra,
    const AtomsCollector *collector, PairScratch *scratch) {
  const Eigen::Index N = coords.rows();
  if (N == 0 || moved.empty()) {
    return;
  }
  PairScratch local;
  PairScratch &s = scratch ? *scratch : local;
  const bool filter_intra = exclude_intra && !molecule_ids.empty();
  const double inv_dr = static_cast<double>(n_bins) / (r_max - r_min);

  // O(1) "position of atom in moved" lookup: moved_pos[atom] = its index in
  // `moved` (kNotMoved otherwise). Sized to N, only the K moved slots are
  // touched and reset, so setup/teardown is O(K) not O(N).
  constexpr std::size_t kNotMoved = static_cast<std::size_t>(-1);
  if (s.moved_pos.size() < static_cast<std::size_t>(N)) {
    s.moved_pos.assign(static_cast<std::size_t>(N), kNotMoved);
  }
  for (const auto [mk, k] : moved | std::views::enumerate) {
    s.moved_pos[k] = static_cast<std::size_t>(mk);
  }

  // Contiguous SoA columns: each moved atom gets its squared distances to all
  // atoms in one vectorised Eigen expression (see squared_distances()).
  s.X = coords.col(0);
  s.Y = coords.col(1);
  s.Z = coords.col(2);
  const BoundaryConditions *pbc = bc && bc->periodic() ? bc : nullptr;

  for (const auto [mk, k] : moved | std::views::enumerate) {
    if (collector && collector->absent(k)) {
      continue; // moved atom is removed: every pair (k, ·) is gone
    }
    squared_distances(s.d2, s.X, s.Y, s.Z, static_cast<Eigen::Index>(k), pbc,
                      s.delta, s.frac);
    for (std::size_t j = 0; j < static_cast<std::size_t>(N); ++j) {
      // A pair of two moved atoms is counted once, when k precedes j in
      // `moved`; removed partners and (optionally) same-molecule pairs skip.
      const bool counted_earlier =
          s.moved_pos[j] != kNotMoved &&
          s.moved_pos[j] < static_cast<std::size_t>(mk);
      if (j == k || (collector && collector->absent(j)) || counted_earlier ||
          (filter_intra && molecule_ids[k] == molecule_ids[j])) {
        continue;
      }
      bin_distance(hist, std::sqrt(s.d2(static_cast<Eigen::Index>(j))),
                   2.0 * weights.weight_of(elem_id, k, j), r_min, inv_dr,
                   n_bins);
    }
  }

  for (const std::size_t k : moved) {
    s.moved_pos[k] = kNotMoved;
  }
}

void PairConstraintBase::set_n_frames(std::size_t n) {
  BOOST_ASSERT_MSG(n_bins_ > 0,
                   "call set_experimental_data before set_n_frames");
  hist_.set_n_frames(n);
}

void PairConstraintBase::rollback_frame() noexcept {
  hist_.rollback([] {}); // no auxiliary undo for the pair path
}

void PairConstraintBase::set_grid(double r_first, double dr, int n) {
  exp_r_ = (r_first + Eigen::ArrayXd::LinSpaced(n, 0, n - 1) * dr).matrix();
  exp_data_ = vec_t::Zero(n);
  r_min_ = r_first;
  r_max_ = exp_r_(n - 1);
  bin_width_ = dr;
  n_bins_ = n;
  computed_.resize(n);
  hist_.set_length(n);
  hist_.set_n_frames(1);
}

void PairConstraintBase::set_experimental_data(const mat_t &data) {
  BOOST_ASSERT_MSG(data.cols() >= 2, "PairConstraint: need 2-column r/data");
  const Eigen::Index N = data.rows();
  set_grid(data(0, 0), N > 1 ? data(1, 0) - data(0, 0) : 0.1,
           static_cast<int>(N));
  exp_r_ = data.col(0);
  exp_data_ = data.col(1);
  r_max_ = exp_r_(N - 1);
}

void PairConstraintBase::initialise() {
  if (initialised_) {
    return;
  }
  initialised_ = true;
  shell_vols_ = shell_volumes(r_min_, bin_width_, n_bins_);

  // Dense species×species weight table; unset pairs weigh 1.0, and `weighted`
  // skips the lookup entirely when no weights were set (the common case).
  const SpeciesIndex sp(elements_);
  elem_id_ = sp.id;
  const std::size_t n = sp.size();
  weight_table_ = {static_cast<int>(n), !weights_.empty(),
                   std::vector<double>(n * n, 1.0)};
  for (const auto &[key, w] : weights_) {
    if (const auto a = sp.id_of(key.a), b = sp.id_of(key.b); a && b) {
      weight_table_.w[*a * n + *b] = w;
      weight_table_.w[*b * n + *a] = w;
    }
  }
}

} // namespace RMC
