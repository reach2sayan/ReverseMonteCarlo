#pragma once
#include <RMC/constraints/Constraint.hpp>
#include <RMC/constraints/PairDistributionConstraint.hpp>
#include <functional>
#include <string>

namespace RMC {

// F(Q) = Q·(S(Q) − 1) = ∫ G(r)·sin(Qr) dr
// The G(r)→F(Q) transform is precomputed as a dense matrix:
//   Fr2FQ[q_i, r_j] = dr · sin(q_i · r_j)
// so F(Q) = Fr2FQ * G(r) as a single matrix–vector product (no 1/Q factor).
class ReducedStructureFactorConstraint
    : public SingularConstraintBase<ReducedStructureFactorConstraint> {
public:
  void set_experimental_data(const mat_t &data); // columns: Q, F(Q)

  constexpr void
  set_boundary_conditions(Constraint::Token tok,
                          const BoundaryConditions &bc) noexcept {
    SingularConstraintBase::set_boundary_conditions(tok, bc);
    pdf_.set_boundary_conditions(bc);
  }
  constexpr void
  set_boundary_conditions(const BoundaryConditions &bc) noexcept {
    SingularConstraintBase::set_boundary_conditions(bc);
    pdf_.set_boundary_conditions(bc);
  }

  // Forward the collector to the internal PDF (which does the atom iteration);
  // also re-applied in initialise(), where pdf_ is reconstructed.
  constexpr void set_collector(Constraint::Token tok,
                               const AtomsCollector *c) noexcept {
    SingularConstraintBase::set_collector(tok, c);
    pdf_.set_collector(c);
  }
  constexpr void set_collector(const AtomsCollector *c) noexcept {
    SingularConstraintBase::set_collector(c);
    pdf_.set_collector(c);
  }

  constexpr void set_elements(std::span<const std::string> elements) noexcept {
    elements_ = elements;
  }
  constexpr void set_number_density(double rho0) noexcept { rho0_ = rho0; }
  void set_weight(const std::string &el1, const std::string &el2, double w) {
    weights_[PairElemKey{el1, el2}] = w;
  }
  void set_shape_function(std::function<double(double)> fn) {
    pdf_.set_shape_function(std::move(fn));
  }
  void set_n_frames(Constraint::Token, std::size_t n) {
    pdf_.PairConstraintBase::set_n_frames(n);
  }
  void set_active_frame(Constraint::Token, std::size_t k) noexcept {
    pdf_.set_active_frame_idx(k);
  }
  void reject(Constraint::Token tok) noexcept {
    SingularConstraintBase<ReducedStructureFactorConstraint>::reject(tok);
    pdf_.rollback_frame();
  }
  void initialise(); // builds Fr2FQ matrix
  // Token-gated overload required by the CConstraint concept; forwards to the
  // public initialise() above (which would otherwise hide the base version).
  void initialise(Constraint::Token) { initialise(); }

  [[nodiscard]] static constexpr std::string_view name() noexcept {
    return "ReducedStructureFactorConstraint";
  }
  [[nodiscard]] static constexpr double
  computation_cost(Constraint::Token) noexcept {
    return 2e6;
  }

  [[nodiscard]] double compute_error(const coords_t &coords,
                                     std::span<const std::size_t> moved) const;

  [[nodiscard]] constexpr const vec_t &computed_F() const noexcept {
    return computed_F_;
  }
  [[nodiscard]] constexpr const vec_t &experimental_F() const noexcept {
    return exp_F_;
  }

private:
  vec_t exp_Q_, exp_F_;
  mat_t Fr2FQ_; // (n_Q × n_r) Fourier matrix (no 1/Q normalisation)
  mutable vec_t computed_F_;

  double r_min_{0.05}, r_max_{20.0}, r_bin_{0.05};
  double rho0_{0.1};
  int n_r_bins_{400};

  PairDistributionConstraint pdf_;
  boost::container::flat_map<PairElemKey, double> weights_;
  std::span<const std::string> elements_;
};

static_assert(CConstraint<ReducedStructureFactorConstraint>,
              "ReducedStructureFactorConstraint must satisfy the CConstraint "
              "concept");

} // namespace RMC
