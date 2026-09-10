#pragma once
#include <RMC/constraints/Constraint.hpp>
#include <RMC/constraints/IncrementalHistogram.hpp>
#include <RMC/core/BoundaryConditions.hpp>
#include <RMC/core/SpeciesIndex.hpp>
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

// Spherical envelope for a particle of given diameter:
// f(r) = 1 - (3/2)(r/d) + (1/2)(r/d)³   for r < d, else 0
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

// Per-bin ideal-gas shell volumes (4π/3)(r_hi³ − r_lo³) of a uniform radial
// grid starting at r_min.
[[nodiscard]] inline vec_t shell_volumes(double r_min, double bin_width,
                                         int n_bins) {
  const auto r_lo =
      r_min + Eigen::ArrayXd::LinSpaced(n_bins, 0, n_bins - 1) * bin_width;
  return (4.0 * std::numbers::pi / 3.0) *
         ((r_lo + bin_width).cube() - r_lo.cube());
}

// Accumulate a raw pair-count histogram into `hist`; each pair (i<j) adds 2*w to
// hist[bin]. If molecule_ids non-empty and exclude_intra, same-molecule pairs skipped.
void accumulate_pair_histogram(vec_t &hist, const coords_t &coords,
                               const BoundaryConditions *bc,
                               const std::vector<uint8_t> &elem_id,
                               const PairWeightMatrix &weights, double r_min,
                               double r_max, int n_bins,
                               std::span<const std::size_t> molecule_ids = {},
                               bool exclude_intra = false,
                               const AtomsCollector *collector = nullptr);

// Accumulate only pairs involving at least one atom from `moved` (O(K·N)
// incremental update); adds into `hist` (caller zeroes). A moved-moved pair (k,j)
// is counted once, when k precedes j in the moved array.
// Reused buffers of accumulate_moved_pairs, one per constraint. A null
// `scratch` makes the call use temporary buffers.
struct PairScratch {
  std::vector<std::size_t> moved_pos; // atom → its index in `moved` (or npos)
  Eigen::ArrayXd X, Y, Z, d2;         // SoA coordinates, squared distances
  Eigen::Matrix3Xd delta, frac;       // minimum-image workspace
};
void accumulate_moved_pairs(
    vec_t &hist, const coords_t &coords, const BoundaryConditions *bc,
    const std::vector<uint8_t> &elem_id, const PairWeightMatrix &weights,
    double r_min, double r_max, int n_bins, std::span<const std::size_t> moved,
    std::span<const std::size_t> molecule_ids = {}, bool exclude_intra = false,
    const AtomsCollector *collector = nullptr, PairScratch *scratch = nullptr);

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
  // Uniform grid of n bin centres from r_first in steps of dr, with a zero
  // target (for a PDF used only as a G(r) source, e.g. by S(Q)/F(Q)).
  void set_grid(double r_first, double dr, int n);
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
    shape_factors_.resize(0);
  }

  void set_n_frames(std::size_t n);
  void set_active_frame_idx(std::size_t k) noexcept {
    hist_.set_active_frame(k);
  }

  void rollback_frame() noexcept;

  [[nodiscard]] constexpr const vec_t &computed_G() const noexcept {
    return computed_;
  }
  [[nodiscard]] constexpr const vec_t &experimental_data() const noexcept {
    return exp_data_;
  }

  // Call after set_experimental_data / set_elements / set_weight. Idempotent.
  void initialise();

protected:
  bool initialised_{false}; // guards initialise() against repeat work
  std::function<double(double)> shape_fn_;
  mutable vec_t shape_factors_; // shape_fn_ at the bin centres, built on first use
  mutable PairScratch scratch_;

  // Single-/multi-frame incremental histogram engine (shared with angular constraint).
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
  using PairConstraintBase::set_grid;
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
  computation_cost() noexcept {
    return 1e6;
  }

  [[nodiscard]] double compute_error(const coords_t &coords,
                                     std::span<const std::size_t> moved) const;

  // Multi-frame hooks of the constraint interface.
  void set_n_frames(std::size_t n) {
    PairConstraintBase::set_n_frames(n);
  }
  void set_active_frame(std::size_t k) noexcept {
    PairConstraintBase::set_active_frame_idx(k);
  }
  // Routes the engine's safety-net initialise sweep to the real setup.
  void initialise() { PairConstraintBase::initialise(); }

  // Restore the active frame's histogram on rejection.
  void reject() noexcept {
    SingularConstraintBase<PairFunctionConstraint<Mode>>::reject();
    PairConstraintBase::rollback_frame();
  }
};

template <PairNorm Mode>
double PairFunctionConstraint<Mode>::compute_error(
    const coords_t &coords, std::span<const std::size_t> moved) const {
  const Eigen::Index N = coords.rows();

  // Build/patch the raw pair-count histogram into computed_; before/after-move
  // hooks are no-ops (pairs recomputed directly from coords).
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
                               exclude_intra_, this->collector_, &scratch_);
      });

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

  // Apply the shape function (nanoparticle envelope), cached per bin centre.
  if (shape_fn_) {
    if (shape_factors_.size() != n_bins_) {
      shape_factors_ = vec_t::NullaryExpr(n_bins_, [&](Eigen::Index i) {
        return shape_fn_(r_min_ + (static_cast<double>(i) + 0.5) * bin_width_);
      });
    }
    computed_.array() *= shape_factors_.array();
  }

  const double denom = computed_.squaredNorm();
  const double scale = (denom > 1e-30) ? computed_.dot(exp_data_) / denom : 1.0;
  return (scale * computed_ - exp_data_).squaredNorm();
}

} // namespace RMC
