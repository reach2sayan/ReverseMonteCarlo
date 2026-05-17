#pragma once
#include <RMC/constraints/Constraint.hpp>
#include <cmath>
#include <numbers>
#include <string>
#include <unordered_map>
#include <vector>

namespace RMC {

// Total-scattering pair correlation F(r):
//   F(r) = Σ_{α,β} w_{αβ} · [g_{αβ}(r) − 1]
// compared to experimental F(r) via squared residuals.
class PairCorrelationConstraint
    : public ConstraintBase<PairCorrelationConstraint> {
public:
  void set_experimental_data(const mat_t &data);
  void set_weight(const std::string &el1, const std::string &el2, double w);
  constexpr void set_elements(std::span<const std::string> elements) noexcept {
    elements_ = elements;
  }
  constexpr void set_number_density(double rho0) noexcept { rho0_ = rho0; }
  void initialise();

  [[nodiscard]] std::string name() const { return "PairCorrelationConstraint"; }

  [[nodiscard]] double compute_error(const coords_t &coords,
                                     std::span<const std::size_t> moved) const;

private:
  [[nodiscard]] double weight_for(const std::string &a,
                                  const std::string &b) const noexcept;

  vec_t exp_r_, exp_F_;
  vec_t shell_vols_;
  mutable vec_t computed_F_;
  double r_min_{0.0}, r_max_{10.0}, bin_width_{0.1};
  int n_bins_{100};
  double rho0_{0.1};

  std::unordered_map<std::string, double> weights_;
  std::span<const std::string> elements_;
};

} // namespace RMC
