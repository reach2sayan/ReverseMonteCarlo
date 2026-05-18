#pragma once
#include <Eigen/Core>
#include <RMC/constraints/Constraint.hpp>
#include <RMC/constraints/PairDistributionConstraint.hpp>
#include <string>
#include <vector>

namespace RMC {

// S(Q) constraint: S(Q) = 1 + (1/Q) ∫ G(r) sin(Qr) dr
// The G(r)→S(Q) transform is precomputed as a dense matrix:
//   Gr2Sq[q_i, r_j] = dr · sin(q_i · r_j) / q_i
// so S(Q) = 1 + Gr2Sq * G(r) as a single matrix–vector product.
class StructureFactorConstraint
    : public ConstraintBase<StructureFactorConstraint> {
public:
  void set_experimental_data(const mat_t &data); // columns: Q, S(Q)

  // Propagate bc to the embedded PDF constraint immediately.
  constexpr void
  set_boundary_conditions(IConstraint::Token tok,
                          const BoundaryConditions &bc) noexcept {
    ConstraintBase::set_boundary_conditions(tok, bc);
    pdf_.set_boundary_conditions(bc);
  }
  constexpr void
  set_boundary_conditions(const BoundaryConditions &bc) noexcept {
    ConstraintBase::set_boundary_conditions(bc);
    pdf_.set_boundary_conditions(bc);
  }

  constexpr void set_elements(std::span<const std::string> elements) noexcept {
    elements_ = elements;
  }
  constexpr void set_number_density(double rho0) noexcept { rho0_ = rho0; }
  constexpr void set_weight(const std::string &el1, const std::string &el2,
                            double w) {
    weights_[(el1 < el2) ? (el1 + "_" + el2) : (el2 + "_" + el1)] = w;
  }
  void initialise(); // builds Gr2Sq matrix

  [[nodiscard]] static constexpr std::string_view name() noexcept {
    return "StructureFactorConstraint";
  }
  [[nodiscard]] static constexpr double
  computation_cost(IConstraint::Token) noexcept {
    return 2e6;
  }

  [[nodiscard]] double compute_error(const coords_t &coords,
                                     std::span<const std::size_t> moved) const;

  [[nodiscard]] constexpr const vec_t &computed_S() const noexcept {
    return computed_S_;
  }
  [[nodiscard]] constexpr const vec_t &experimental_S() const noexcept {
    return exp_S_;
  }

private:
  vec_t exp_Q_, exp_S_;
  mat_t Gr2Sq_; // (n_Q × n_r) Fourier matrix
  mutable vec_t computed_S_;

  double r_min_{0.05}, r_max_{20.0}, r_bin_{0.05};
  double rho0_{0.1};
  int n_r_bins_{400};

  // Embedded PDF constraint used to compute the G(r) intermediate.
  PairDistributionConstraint pdf_;
  std::unordered_map<std::string, double> weights_;
  std::span<const std::string> elements_;
};

} // namespace RMC
