#pragma once
#include <RMC/constraints/Constraint.hpp>
#include <RMC/core/BoundaryConditions.hpp>
#include <RMC/core/Types.hpp>
#include <boost/assert.hpp>
#include <boost/container/flat_map.hpp>
#include <cmath>
#include <cstdint>
#include <functional>
#include <numbers>
#include <optional>
#include <span>
#include <string>
#include <vector>
#if defined(_OPENMP)
#include <omp.h>
#endif

namespace RMC {

// Canonical element-pair key by string (used at init time only).
struct PairElemKey {
  std::string a, b;
  PairElemKey(std::string x, std::string y)
      : a(x < y ? std::move(x) : std::move(y)),
        b(x < y ? std::move(y) : std::move(x)) {}
  auto operator<=>(const PairElemKey &) const = default;
};

// Composite uint8_t key — lo <= hi always (hot-path lookup table).
struct PairIdKey {
  uint8_t lo, hi;
  constexpr PairIdKey(uint8_t a, uint8_t b)
      : lo(a < b ? a : b), hi(a < b ? b : a) {}
  auto operator<=>(const PairIdKey &) const = default;
};

using PairWeightTable = boost::container::flat_map<PairIdKey, double>;

// Shape-function factory helpers for nanoparticle PDF corrections.
// Apply via PairConstraintBase::set_shape_function().

// Spherical envelope for a particle of given diameter:
//   f(r) = 1 - (3/2)(r/d) + (1/2)(r/d)³   for r < d, else 0
inline auto spherical_shape_fn(double diameter) {
  return [d = diameter](double r) -> double {
    if (r >= d) {
      return 0.0;
    }
    const double x = r / d;
    return 1.0 - 1.5 * x + 0.5 * x * x * x;
  };
}

// Gaussian damping envelope:  f(r) = exp(-r²/σ²)
inline auto gaussian_shape_fn(double sigma) {
  return
      [s = sigma](double r) -> double { return std::exp(-(r * r) / (s * s)); };
}

// Accumulate a raw pair-count histogram into `hist`.
// Each pair (i<j) contributes 2*w to hist[bin].
// If molecule_ids is non-empty and exclude_intra is true, same-molecule pairs
// are skipped (useful for modelling molecular liquids).
void accumulate_pair_histogram(vec_t &hist, const coords_t &coords,
                               const BoundaryConditions *bc,
                               const std::vector<uint8_t> &elem_id,
                               const PairWeightTable &weight_table,
                               double r_min, double r_max, int n_bins,
                               std::span<const std::size_t> molecule_ids = {},
                               bool exclude_intra = false);

// Accumulate only the pairs that involve at least one atom from `moved`.
// Used for O(K·N) incremental histogram updates in the multi-frame MC path.
// Adds into `hist` (caller should zero-initialise before calling).
// Double-counting of moved-moved pairs is avoided: pair (k,j) with both in
// moved is counted once, when k appears before j in the moved array.
void accumulate_moved_pairs(
    vec_t &hist, const coords_t &coords, const BoundaryConditions *bc,
    const std::vector<uint8_t> &elem_id, const PairWeightTable &weight_table,
    double r_min, double r_max, int n_bins, std::span<const std::size_t> moved,
    std::span<const std::size_t> molecule_ids = {}, bool exclude_intra = false);

class PairConstraintBase {
protected:
  vec_t exp_r_, exp_data_;
  vec_t shell_vols_;
  mutable vec_t computed_;
  double r_min_{0.0}, r_max_{10.0}, bin_width_{0.1};
  int n_bins_{100};
  double rho0_{0.1};

  boost::container::flat_map<PairElemKey, double> weights_;
  std::span<const std::string> elements_;
  std::span<const std::size_t> molecule_ids_;
  bool exclude_intra_{false};

  std::vector<uint8_t> elem_id_;
  PairWeightTable weight_table_;

public:
  void set_experimental_data(const mat_t &data);

  constexpr void set_weight(const std::string &el1, const std::string &el2,
                            double w) {
    weights_.insert_or_assign(PairElemKey{el1, el2}, w); // = w;
  }
  constexpr void set_elements(std::span<const std::string> e) noexcept {
    elements_ = e;
  }
  constexpr void set_number_density(double rho0) noexcept { rho0_ = rho0; }
  constexpr void set_molecule_ids(std::span<const std::size_t> ids) noexcept {
    molecule_ids_ = ids;
  }
  constexpr void set_exclude_intra(bool v) noexcept { exclude_intra_ = v; }

  // Shape function: f(r) multiplied into the computed G(r) before chi²
  // evaluation. Use spherical_shape_fn() or gaussian_shape_fn() as factories.
  void set_shape_function(std::function<double(double)> fn) {
    shape_fn_ = std::move(fn);
  }

  // Multi-frame: allocate per-frame histograms. Must be called after
  // set_experimental_data() so that n_bins_ is known.
  void set_n_frames(std::size_t n) {
    BOOST_ASSERT_MSG(n_bins_ > 0,
                     "call set_experimental_data before set_n_frames");
    n_frames_ = n;
    frame_hists_.assign(n, vec_t::Zero(n_bins_));
    frame_hist_current_.assign(n, false);
    sum_hist_ = vec_t::Zero(n_bins_);
    saved_frame_hist_.resize(n_bins_);
    incremental_ready_ = false;
  }
  void set_active_frame_idx(std::size_t k) noexcept {
    active_frame_ = k;
    incremental_ready_ = false; // frame switch invalidates any pending delta
  }

  // Roll back the active frame's histogram to the state saved during the last
  // compute_error() call (used by PairFunctionConstraint::reject()).
  void rollback_frame() noexcept {
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

  [[nodiscard]] constexpr const vec_t &computed_G() const noexcept {
    return computed_;
  }
  [[nodiscard]] constexpr const vec_t &experimental_data() const noexcept {
    return exp_data_;
  }

  // Call after set_experimental_data / set_elements / set_weight.
  void initialise();

protected:
  std::optional<std::function<double(double)>> shape_fn_;

  // Single-frame incremental state
  mutable vec_t single_hist_;
  mutable bool single_hist_current_{
      false}; // false → next call does full rebuild

  // Multi-frame state (mutable: modified inside const compute_error())
  std::size_t n_frames_{1};
  std::size_t active_frame_{0};
  mutable std::vector<vec_t> frame_hists_;
  mutable std::vector<bool>
      frame_hist_current_; // true after first full build per frame
  mutable vec_t sum_hist_;
  mutable vec_t saved_frame_hist_;
  mutable vec_t
      saved_moved_delta_; // pair contributions from moved atoms (old pos)
  mutable bool incremental_ready_{
      false}; // set by before-move, cleared by after-move/reject
};

// ---------------------------------------------------------------------------
// PairNorm selects the normalization formula applied after histogramming.
//   PDF: G(r) = 4π·r·ρ₀·(g(r) − 1)   — radial prefactor
//   PCF: F(r) =         g(r) − 1       — no radial prefactor
// ---------------------------------------------------------------------------
enum class PairNorm { PDF, PCF };

template <PairNorm Mode>
class PairFunctionConstraint
    : public SingularConstraintBase<PairFunctionConstraint<Mode>>,
      public PairConstraintBase {
public:
  using SingularConstraintBase<PairFunctionConstraint<Mode>>::bc_;

  using PairConstraintBase::initialise;
  using PairConstraintBase::set_elements;
  using PairConstraintBase::set_exclude_intra;
  using PairConstraintBase::set_experimental_data;
  using PairConstraintBase::set_molecule_ids;
  using PairConstraintBase::set_number_density;
  using PairConstraintBase::set_weight;

  // set_boundary_conditions must be forwarded from SingularConstraintBase.
  using SingularConstraintBase<
      PairFunctionConstraint<Mode>>::set_boundary_conditions;

  [[nodiscard]] static constexpr std::string_view name() noexcept {
    if constexpr (Mode == PairNorm::PDF) {
      return "PairDistributionConstraint";
    } else {
      return "PairCorrelationConstraint";
    }
  }

  [[nodiscard]] static constexpr double
  computation_cost(Constraint::Token) noexcept {
    return 1e6;
  }

  [[nodiscard]] double compute_error(const coords_t &coords,
                                     std::span<const std::size_t> moved) const;

  // Token-gated wrappers for CConstraint compliance (multi-frame).
  void set_n_frames(Constraint::Token, std::size_t n) {
    PairConstraintBase::set_n_frames(n);
  }
  void set_active_frame(Constraint::Token, std::size_t k) noexcept {
    PairConstraintBase::set_active_frame_idx(k);
  }

  // Restore the active frame's histogram on rejection.
  void reject(Constraint::Token tok) noexcept {
    SingularConstraintBase<PairFunctionConstraint<Mode>>::reject(tok);
    PairConstraintBase::rollback_frame();
  }
};

template <PairNorm Mode>
double PairFunctionConstraint<Mode>::compute_error(
    const coords_t &coords, std::span<const std::size_t> moved) const {
  const Eigen::Index N = coords.rows();

  if (n_frames_ > 1) {
    if (incremental_ready_ && !moved.empty()) {
      // After-move: O(K·N) incremental update — only recompute pairs
      // involving the moved atoms; skip the full O(N²) rebuild.
      vec_t new_delta = vec_t::Zero(n_bins_);
      accumulate_moved_pairs(new_delta, coords, bc_, elem_id_, weight_table_,
                             r_min_, r_max_, n_bins_, moved, molecule_ids_,
                             exclude_intra_);
      const vec_t new_frame_hist =
          saved_frame_hist_ - saved_moved_delta_ + new_delta;
      sum_hist_ += new_frame_hist - frame_hists_[active_frame_];
      frame_hists_[active_frame_] = new_frame_hist;
      incremental_ready_ = false;
    } else {
      // Before-move path.
      saved_frame_hist_ = frame_hists_[active_frame_];
      if (!frame_hist_current_[active_frame_]) {
        // First call for this frame (e.g. during initialise()): full rebuild.
        vec_t tmp = vec_t::Zero(n_bins_);
        accumulate_pair_histogram(tmp, coords, bc_, elem_id_, weight_table_,
                                  r_min_, r_max_, n_bins_, molecule_ids_,
                                  exclude_intra_);
        sum_hist_ += tmp - saved_frame_hist_;
        frame_hists_[active_frame_] = tmp;
        saved_frame_hist_ = tmp;
        frame_hist_current_[active_frame_] = true;
      }
      // Record moved-atom contributions for the upcoming after-move call.
      if (!moved.empty()) {
        saved_moved_delta_ = vec_t::Zero(n_bins_);
        accumulate_moved_pairs(saved_moved_delta_, coords, bc_, elem_id_,
                               weight_table_, r_min_, r_max_, n_bins_, moved,
                               molecule_ids_, exclude_intra_);
        incremental_ready_ = true;
      }
    }

    // Normalise averaged histogram to G(r) / PCF.
    computed_ = sum_hist_ / static_cast<double>(n_frames_);
    if constexpr (Mode == PairNorm::PDF) {
      const auto idx = Eigen::ArrayXd::LinSpaced(n_bins_, 0, n_bins_ - 1);
      const auto r = r_min_ + (idx + 0.5) * bin_width_;
      computed_.array() =
          4.0 * std::numbers::pi * r * rho0_ *
          (computed_.array() / (shell_vols_.array() * rho0_ * N) - 1.0);
    } else {
      computed_.array() /= shell_vols_.array() * rho0_ * N;
      computed_.array() -= 1.0;
    }
  } else {
    // Single-frame incremental path: O(K·N) per step after the first call.
    if (incremental_ready_ && !moved.empty()) {
      // After-move: patch the running histogram with the moved-atom delta.
      vec_t new_delta = vec_t::Zero(n_bins_);
      accumulate_moved_pairs(new_delta, coords, bc_, elem_id_, weight_table_,
                             r_min_, r_max_, n_bins_, moved, molecule_ids_,
                             exclude_intra_);
      single_hist_ = saved_frame_hist_ - saved_moved_delta_ + new_delta;
      incremental_ready_ = false;
    } else {
      // Before-move (or first call after initialise / set_experimental_data).
      saved_frame_hist_ = single_hist_;
      if (!single_hist_current_) {
        single_hist_ = vec_t::Zero(n_bins_);
        accumulate_pair_histogram(single_hist_, coords, bc_, elem_id_,
                                  weight_table_, r_min_, r_max_, n_bins_,
                                  molecule_ids_, exclude_intra_);
        saved_frame_hist_ = single_hist_;
        single_hist_current_ = true;
      }
      if (!moved.empty()) {
        saved_moved_delta_ = vec_t::Zero(n_bins_);
        accumulate_moved_pairs(saved_moved_delta_, coords, bc_, elem_id_,
                               weight_table_, r_min_, r_max_, n_bins_, moved,
                               molecule_ids_, exclude_intra_);
        incremental_ready_ = true;
      }
    }
    computed_ = single_hist_;
    if constexpr (Mode == PairNorm::PDF) {
      const auto idx = Eigen::ArrayXd::LinSpaced(n_bins_, 0, n_bins_ - 1);
      const auto r = r_min_ + (idx + 0.5) * bin_width_;
      computed_.array() =
          4.0 * std::numbers::pi * r * rho0_ *
          (computed_.array() / (shell_vols_.array() * rho0_ * N) - 1.0);
    } else {
      computed_.array() /= shell_vols_.array() * rho0_ * N;
      computed_.array() -= 1.0;
    }
  }

  // Apply shape function (nanoparticle envelope), if set.
  if (shape_fn_) {
    const auto &fn = *shape_fn_;
    for (int i = 0; i < n_bins_; ++i) {
      computed_[i] *= std::invoke(fn, r_min_ + (i + 0.5) * bin_width_);
    }
  }

  const double denom = computed_.squaredNorm();
  const double scale = (denom > 1e-30) ? computed_.dot(exp_data_) / denom : 1.0;
  return (scale * computed_ - exp_data_).squaredNorm();
}

} // namespace RMC
