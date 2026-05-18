#pragma once
#include <Eigen/Core>
#include <RMC/constraints/Constraint.hpp>
#include <cmath>
#include <numbers>
#include <string>
#include <unordered_map>
#include <vector>

namespace RMC {

// Pair distribution function G(r) = 4πr·ρ₀·(g(r) − 1) constraint.
// Compares computed G(r) to loaded experimental data via squared residuals.
//
// Weighting: uniform (unweighted) or by element-pair weights (e.g. neutron
// coherent scattering lengths).
class PairDistributionConstraint
    : public ConstraintBase<PairDistributionConstraint> {
public:
  // Load experimental data: two-column matrix (r, G(r)).
  void set_experimental_data(const mat_t &data);

  // Per element-pair weight (e.g. b_i * b_j / <b>^2 for neutron).
  constexpr void set_weight(const std::string &el1, const std::string &el2,
                            double w) {
    std::string key = (el1 < el2) ? (el1 + "_" + el2) : (el2 + "_" + el1);
    weights_[key] = w;
  }
  constexpr void set_elements(std::span<const std::string> elements) noexcept {
    elements_ = elements;
  }
  constexpr void set_number_density(double rho0) noexcept { rho0_ = rho0; }
  void initialise();

  [[nodiscard]] static constexpr std::string_view name() noexcept {
    return "PairDistributionConstraint";
  }

  [[nodiscard]] double compute_error(const coords_t &coords,
                                     std::span<const std::size_t> moved) const;

  [[nodiscard]] const vec_t &computed_G() const noexcept { return computed_G_; }
  [[nodiscard]] const vec_t &experimental_G() const noexcept { return exp_G_; }

private:
  [[nodiscard]] double weight_for(const std::string &a,
                                  const std::string &b) const noexcept;

  vec_t exp_r_;
  vec_t exp_G_;
  vec_t shell_vols_; // (4π/3)(r_hi³ - r_lo³) per bin
  mutable vec_t computed_G_;

  double r_min_{0.0};
  double r_max_{10.0};
  double bin_width_{0.1};
  int n_bins_{100};
  double rho0_{0.1}; // atoms/Å³

  std::unordered_map<std::string, double> weights_;
  std::span<const std::string> elements_;
};

} // namespace RMC
