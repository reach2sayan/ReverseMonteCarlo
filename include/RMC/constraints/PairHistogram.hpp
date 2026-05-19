#pragma once
#include <RMC/constraints/Constraint.hpp>
#include <RMC/core/BoundaryConditions.hpp>
#include <RMC/core/Types.hpp>
#include <boost/assert.hpp>
#include <boost/container/flat_map.hpp>
#include <cmath>
#include <cstdint>
#include <numbers>
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

// Composite uint8_t key — lo <= hi always (hot-path lookup table).
struct PairIdKey {
  uint8_t lo, hi;
  constexpr PairIdKey(uint8_t a, uint8_t b)
      : lo(a < b ? a : b), hi(a < b ? b : a) {}
  auto operator<=>(const PairIdKey &) const = default;
};

using PairWeightTable = boost::container::flat_map<PairIdKey, double>;

// Accumulate a raw pair-count histogram into `hist`.
// Each pair (i<j) contributes 2*w to hist[bin].
// If molecule_ids is non-empty and exclude_intra is true, same-molecule pairs
// are skipped (useful for modelling molecular liquids).
inline void accumulate_pair_histogram(
    vec_t &hist, const coords_t &coords, const BoundaryConditions *bc,
    const std::vector<uint8_t> &elem_id, const PairWeightTable &weight_table,
    double r_min, double r_max, double bin_width, int n_bins,
    std::span<const std::size_t> molecule_ids = {},
    bool exclude_intra = false) {
  const Eigen::Index N = coords.rows();
  const bool weighted = !weight_table.empty();
  const bool filter_intra = exclude_intra && !molecule_ids.empty();

  for (auto [i, j] : upper_triangle_pairs(N)) {
    if (filter_intra && molecule_ids[static_cast<std::size_t>(i)] ==
                            molecule_ids[static_cast<std::size_t>(j)]) {
      continue;
    }

    vec3_t delta = coords.row(j).transpose() - coords.row(i).transpose();
    if (bc) {
      delta = bc_min_image(*bc, delta);
    }
    const double d = delta.norm();
    if (d < r_min || d >= r_max) {
      continue;
    }
    const int bin = static_cast<int>((d - r_min) / bin_width);
    if (bin < 0 || bin >= n_bins) {
      continue;
    }

    double w = 1.0;
    if (weighted) {
      PairIdKey key{elem_id[static_cast<std::size_t>(i)],
                    elem_id[static_cast<std::size_t>(j)]};
      if (auto it = weight_table.find(key); it != weight_table.end()) {
        w = it->second;
      }
    }
    hist(bin) += 2.0 * w;
  }
}

// ---------------------------------------------------------------------------
// Shared base: r-grid, shell volumes, element-ID table, weight table.
// Derived classes (PairFunctionConstraint) supply the normalization formula.
// ---------------------------------------------------------------------------
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
  void set_experimental_data(const mat_t &data) {
    BOOST_ASSERT_MSG(data.cols() >= 2, "PairConstraint: need 2-column r/data");
    const Eigen::Index N = data.rows();
    exp_r_ = data.col(0);
    exp_data_ = data.col(1);
    r_min_ = exp_r_(0);
    r_max_ = exp_r_(N - 1);
    bin_width_ = (N > 1) ? (exp_r_(1) - exp_r_(0)) : 0.1;
    n_bins_ = static_cast<int>(N);
    computed_.resize(N);
  }

  void set_weight(const std::string &el1, const std::string &el2, double w) {
    weights_[PairElemKey{el1, el2}] = w;
  }
  constexpr void set_elements(std::span<const std::string> e) noexcept {
    elements_ = e;
  }
  constexpr void set_number_density(double rho0) noexcept { rho0_ = rho0; }
  constexpr void set_molecule_ids(std::span<const std::size_t> ids) noexcept {
    molecule_ids_ = ids;
  }
  constexpr void set_exclude_intra(bool v) noexcept { exclude_intra_ = v; }
  [[nodiscard]] constexpr const vec_t &computed_G() const noexcept {
    return computed_;
  }
  [[nodiscard]] constexpr const vec_t &experimental_data() const noexcept {
    return exp_data_;
  }

  // Call after set_experimental_data / set_elements / set_weight.
  void initialise();
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
  computation_cost(IConstraint::Token) noexcept {
    return 1e6;
  }

  [[nodiscard]] double
  compute_error(const coords_t &coords,
                std::span<const std::size_t> /*moved*/) const {
    computed_.setZero();
    accumulate_pair_histogram(computed_, coords, bc_, elem_id_, weight_table_,
                              r_min_, r_max_, bin_width_, n_bins_,
                              molecule_ids_, exclude_intra_);

    const Eigen::Index N = coords.rows();

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

    const double denom = computed_.squaredNorm();
    const double scale =
        (denom > 1e-30) ? computed_.dot(exp_data_) / denom : 1.0;
    return (scale * computed_ - exp_data_).squaredNorm();
  }
};

} // namespace RMC
