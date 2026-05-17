#pragma once
#include <Eigen/Core>
#include <fullrmc/constraints/Constraint.hpp>
#include <fullrmc/constraints/PairDistributionConstraint.hpp>
#include <string>
#include <vector>

namespace fullrmc {

// S(Q) constraint: S(Q) = 1 + (1/Q) ∫ G(r) sin(Qr) dr
// The G(r)→S(Q) transform is precomputed as a dense matrix:
//   Gr2Sq[q_i, r_j] = dr · sin(q_i · r_j) / q_i
// so S(Q) = 1 + Gr2Sq * G(r) as a single matrix–vector product.
class StructureFactorConstraint
    : public ConstraintBase<StructureFactorConstraint> {
public:
  void set_experimental_data(const mat_t &data); // columns: Q, S(Q)
  void set_elements(const std::vector<std::string> *elements) noexcept {
    elements_ = elements;
  }
  void set_number_density(double rho0) noexcept { rho0_ = rho0; }
  void set_weight(const std::string &el1, const std::string &el2, double w);
  void initialise(); // builds Gr2Sq matrix

  [[nodiscard]] std::string name() const override {
    return "StructureFactorConstraint";
  }

  [[nodiscard]] double compute_error(const coords_t &coords,
                                     std::span<const std::size_t> moved) const;

  [[nodiscard]] const vec_t &computed_S() const noexcept { return computed_S_; }
  [[nodiscard]] const vec_t &experimental_S() const noexcept { return exp_S_; }

private:
  vec_t exp_Q_, exp_S_;
  mat_t Gr2Sq_; // (n_Q × n_r) Fourier matrix
  mutable vec_t computed_S_;

  double r_min_{0.05}, r_max_{20.0}, r_bin_{0.05};
  double rho0_{0.1};
  int n_r_bins_{400};

  // Shared PDF calculation for G(r) intermediate.
  std::unique_ptr<PairDistributionConstraint> pdf_;
  std::unordered_map<std::string, double> weights_;
  const std::vector<std::string> *elements_ = nullptr;
};

} // namespace fullrmc
