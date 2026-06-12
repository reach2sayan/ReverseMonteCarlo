#pragma once
#include <RMC/constraints/Constraint.hpp>
#include <RMC/constraints/IncrementalHistogram.hpp>
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

namespace RMC {

// Canonical element-pair key by string (used at init time only).
struct PairElemKey {
  std::string a, b;
  PairElemKey(std::string x, std::string y)
      : a(x < y ? std::move(x) : std::move(y)),
        b(x < y ? std::move(y) : std::move(x)) {}
  auto operator<=>(const PairElemKey &) const = default;
};

// Dense element-pair weight table for O(1) hot-path lookup.
struct PairWeightMatrix {
  int n_types{0};
  bool weighted{false};
  std::vector<double> w; // n_types*n_types, row-major, symmetric
  [[nodiscard]] constexpr double at(uint8_t a, uint8_t b) const noexcept {
    return w[static_cast<std::size_t>(a) * static_cast<std::size_t>(n_types) +
             static_cast<std::size_t>(b)];
  }

  // Hot-path weight for the pair (elem_id[i], elem_id[j]).
  [[nodiscard]] constexpr double weight_of(const std::vector<uint8_t> &elem_id,
                                           std::size_t i,
                                           std::size_t j) const noexcept {
    return weighted ? at(elem_id[i], elem_id[j]) : 1.0;
  }
};

// Shape-function factory helpers for nanoparticle PDF corrections.
// Apply via PairConstraintBase::set_shape_function().

// Spherical envelope for a particle of given diameter:
//   f(r) = 1 - (3/2)(r/d) + (1/2)(r/d)³   for r < d, else 0
FORCE_INLINE auto spherical_shape_fn(double diameter) {
  return [d = diameter](double r) -> double {
    if (r >= d) {
      return 0.0;
    }
    const double x = r / d;
    return 1.0 - 1.5 * x + 0.5 * x * x * x;
  };
}

// Gaussian damping envelope:  f(r) = exp(-r²/σ²)
FORCE_INLINE auto gaussian_shape_fn(double sigma) {
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
                               const PairWeightMatrix &weights, double r_min,
                               double r_max, int n_bins,
                               std::span<const std::size_t> molecule_ids = {},
                               bool exclude_intra = false,
                               const AtomsCollector *collector = nullptr);

// Accumulate only the pairs that involve at least one atom from `moved`.
// Used for O(K·N) incremental histogram updates in the multi-frame MC path.
// Adds into `hist` (caller should zero-initialize before calling).
// Double-counting of moved-moved pairs is avoided: pair (k,j) with both in
// moved is counted once, when k appears before j in the moved array.
void accumulate_moved_pairs(
    vec_t &hist, const coords_t &coords, const BoundaryConditions *bc,
    const std::vector<uint8_t> &elem_id, const PairWeightMatrix &weights,
    double r_min, double r_max, int n_bins, std::span<const std::size_t> moved,
    std::span<const std::size_t> molecule_ids = {}, bool exclude_intra = false,
    const AtomsCollector *collector = nullptr);

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
  PairWeightMatrix weight_table_;

public:
  void set_experimental_data(const mat_t &data);

  void set_weight(const std::string &el1, const std::string &el2, double w) {
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

  // Shape function: f(r) multiplied into the computed G(r) before chi² eval
  void set_shape_function(std::function<double(double)> fn) {
    shape_fn_ = std::move(fn);
  }

  void set_n_frames(std::size_t n);
  void set_active_frame_idx(std::size_t k) noexcept { hist_.set_active_frame(k); }

  // Roll back the active frame's histogram to the state saved during the last
  // compute_error() call (used by PairFunctionConstraint::reject()).
  void rollback_frame() noexcept;

  [[nodiscard]] constexpr const vec_t &computed_G() const noexcept {
    return computed_;
  }
  [[nodiscard]] constexpr const vec_t &experimental_data() const noexcept {
    return exp_data_;
  }

  // Call after set_experimental_data / set_elements / set_weight. Idempotent:
  // safe to call more than once (e.g. client call + engine safety-net sweep).
  void initialise();

protected:
  bool initialised_{false}; // guards initialise() against repeat work
  std::optional<std::function<double(double)>> shape_fn_;

  // Single-/multi-frame incremental histogram engine (shared with the angular
  // constraint). Holds all running counts and the moved-atom delta scratch.
  mutable IncrementalHistogram hist_;
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
  // Token-gated initialise — routes the engine's safety-net sweep to the real
  // (idempotent) per-constraint setup instead of ConstraintBase's no-op
  // default.
  void initialise(Constraint::Token) { PairConstraintBase::initialise(); }

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

  // Build/patch the raw pair-count histogram into computed_. The incremental
  // moved-atom machinery lives in IncrementalHistogram; pairs are recomputed
  // directly from coords, so the before/after-move hooks are no-ops.
  hist_.update(
      computed_, moved,
      [&](vec_t &h) {
        accumulate_pair_histogram(h, coords, bc_, elem_id_, weight_table_,
                                  r_min_, r_max_, n_bins_, molecule_ids_,
                                  exclude_intra_, this->collector_);
      },
      [&](vec_t &delta, std::span<const std::size_t> mv) {
        accumulate_moved_pairs(delta, coords, bc_, elem_id_, weight_table_,
                               r_min_, r_max_, n_bins_, mv, molecule_ids_,
                               exclude_intra_, this->collector_);
      },
      [](std::span<const std::size_t>) {}, [](std::span<const std::size_t>) {});

  // Normalise raw counts to G(r) / PCF.
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

  // Apply shape function (nanoparticle envelope), if set.
  if (shape_fn_) {
    const auto &fn = *shape_fn_;
    assert(computed_.size() == n_bins_);
    for (auto&& [i, computed_val] : computed_ | std::views::enumerate) {//int i = 0; i < n_bins_; ++i) {
      computed_val *= std::invoke(fn, r_min_ + (i + 0.5) * bin_width_);
    }
  }

  const double denom = computed_.squaredNorm();
  const double scale = (denom > 1e-30) ? computed_.dot(exp_data_) / denom : 1.0;
  return (scale * computed_ - exp_data_).squaredNorm();
}

} // namespace RMC
