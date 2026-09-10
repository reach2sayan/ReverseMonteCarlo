#pragma once
#include <RMC/constraints/Constraint.hpp>
#include <RMC/constraints/PairDistributionConstraint.hpp>
#include <boost/assert.hpp>
#include <boost/math/special_functions/sinc.hpp>
#include <functional>
#include <span>
#include <string>
#include <string_view>

namespace RMC {

// The Fourier-space quantity fitted against the G(r) of an internal PDF:
//   S: S(Q) = 1 + ∫ G(r) sin(Qr)/Q dr
//   F: F(Q) = Q·(S(Q) − 1) = ∫ G(r) sin(Qr) dr
enum class FourierNorm { S, F };

// Structure-factor constraint. The G(r)→S(Q) / F(Q) transform is precomputed
// as a dense (n_Q × n_r) matrix with rows Δr·r·sinc(Q·r) (S) or Δr·sin(Q·r)
// (F), so each evaluation is one matrix–vector product. Singular (one per
// collection); pair setup is forwarded to the internal PDF.
template <FourierNorm Norm>
class FourierConstraint
    : public SingularConstraintBase<FourierConstraint<Norm>> {
  using Base = SingularConstraintBase<FourierConstraint<Norm>>;

public:
  // Columns: Q, target S(Q) or F(Q).
  void set_experimental_data(const mat_t &data) {
    BOOST_ASSERT_MSG(data.cols() >= 2,
                     "FourierConstraint: need 2-column Q/target data");
    exp_q_ = data.col(0);
    exp_ = data.col(1);
  }

  constexpr void set_boundary_conditions(const BoundaryConditions &bc) noexcept {
    Base::set_boundary_conditions(bc);
    pdf_.set_boundary_conditions(bc);
  }
  constexpr void set_collector(const AtomsCollector *c) noexcept {
    Base::set_collector(c);
    pdf_.set_collector(c);
  }
  void set_elements(std::span<const std::string> elements) {
    pdf_.set_elements(elements);
  }
  void set_number_density(double rho0) { pdf_.set_number_density(rho0); }
  void set_weight(const std::string &el1, const std::string &el2, double w) {
    pdf_.set_weight(el1, el2, w);
  }
  void set_shape_function(std::function<double(double)> fn) {
    pdf_.set_shape_function(std::move(fn));
  }
  void set_n_frames(std::size_t n) { pdf_.set_n_frames(n); }
  void set_active_frame(std::size_t k) noexcept { pdf_.set_active_frame(k); }
  void reject() noexcept {
    Base::reject();
    pdf_.rollback_frame();
  }

  // Sets up the internal PDF on a fixed r grid and builds the Fourier matrix.
  // The PDF is configured in place, so the boundary conditions and collector it
  // was given survive. Idempotent.
  void initialise() {
    if (initialised_) {
      return;
    }
    initialised_ = true;
    const int n_r = static_cast<int>((kRMax - kRMin) / kDr);
    const double r_first = kRMin + 0.5 * kDr; // bin centres
    pdf_.set_grid(r_first, kDr, n_r);
    pdf_.initialise();

    const Eigen::ArrayXd r =
        r_first + Eigen::ArrayXd::LinSpaced(n_r, 0, n_r - 1) * kDr;
    kernel_.resize(exp_q_.size(), n_r);
    for (Eigen::Index qi = 0; qi < exp_q_.size(); ++qi) {
      const Eigen::ArrayXd qr = exp_q_(qi) * r;
      if constexpr (Norm == FourierNorm::S) {
        kernel_.row(qi) =
            (kDr * r *
             qr.unaryExpr([](double x) { return boost::math::sinc_pi(x); }))
                .matrix()
                .transpose();
      } else {
        kernel_.row(qi) = (kDr * qr.sin()).matrix().transpose();
      }
    }
  }

  [[nodiscard]] static constexpr std::string_view name() noexcept {
    return Norm == FourierNorm::S ? "StructureFactorConstraint"
                                  : "ReducedStructureFactorConstraint";
  }
  [[nodiscard]] static constexpr double computation_cost() noexcept {
    return 2e6;
  }

  [[nodiscard]] double compute_error(const coords_t &coords,
                                     std::span<const std::size_t> moved) const {
    (void)pdf_.compute_error(coords, moved);
    computed_ = kernel_ * pdf_.computed_G();
    if constexpr (Norm == FourierNorm::S) {
      computed_.array() += 1.0;
    }
    const double denom = computed_.squaredNorm();
    const double scale = denom > 1e-30 ? computed_.dot(exp_) / denom : 1.0;
    return (scale * computed_ - exp_).squaredNorm();
  }

  [[nodiscard]] const vec_t &computed() const noexcept { return computed_; }
  [[nodiscard]] const vec_t &experimental() const noexcept { return exp_; }

private:
  static constexpr double kRMin = 0.05, kRMax = 20.0, kDr = 0.05;
  vec_t exp_q_, exp_;
  mat_t kernel_; // (n_Q × n_r) Fourier matrix
  mutable vec_t computed_;
  bool initialised_{false};
  PairDistributionConstraint pdf_;
};

using StructureFactorConstraint = FourierConstraint<FourierNorm::S>;
using ReducedStructureFactorConstraint = FourierConstraint<FourierNorm::F>;

} // namespace RMC
