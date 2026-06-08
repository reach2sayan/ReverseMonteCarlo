#include <RMC/constraints/StructureFactorConstraint.hpp>
#include <boost/assert.hpp>
#include <cmath>

namespace RMC {

void StructureFactorConstraint::set_experimental_data(const mat_t &data) {
  BOOST_ASSERT_MSG(data.cols() >= 2,
                   "StructureFactorConstraint: need 2-column Q/S(Q) data");
  exp_Q_ = data.col(0);
  exp_S_ = data.col(1);
}

void StructureFactorConstraint::initialise() {
  pdf_ = PairDistributionConstraint{};
  pdf_.set_number_density(rho0_);
  n_r_bins_ = static_cast<int>((r_max_ - r_min_) / r_bin_);
  mat_t r_data(n_r_bins_, 2);
  r_data.col(0) =
      Eigen::VectorXd::LinSpaced(n_r_bins_, 0, n_r_bins_ - 1).array() * r_bin_ +
      (r_min_ + 0.5 * r_bin_);

  r_data.col(1).setZero();
  pdf_.set_experimental_data(r_data);
  for (const auto &[key, w] : weights_) {
    pdf_.set_weight(key.a, key.b, w);
  }
  if (!elements_.empty()) {
    pdf_.set_elements(elements_);
  }
  pdf_.set_collector(collector_); // pdf_ was just reconstructed; re-apply
  pdf_.initialise();

  const Eigen::Index nQ = exp_Q_.size();
  Gr2Sq_.resize(nQ, n_r_bins_);
  // Bin-centre radii r_i = r_min + (i + ½)·Δr.
  const Eigen::ArrayXd r =
      Eigen::ArrayXd::LinSpaced(n_r_bins_, 0, n_r_bins_ - 1) * r_bin_ +
      (r_min_ + 0.5 * r_bin_);
  // Row qi is the Δr·sin(Q·r)/Q kernel for Q = exp_Q_(qi).
  for (Eigen::Index qi = 0; qi < nQ; ++qi) {
    const double q = exp_Q_(qi);
    if (q > 1e-10)
      Gr2Sq_.row(qi) = (r_bin_ / q * (q * r).sin()).matrix().transpose();
    else
      Gr2Sq_.row(qi).setZero();
  }
  computed_S_.resize(nQ);
}

double StructureFactorConstraint::compute_error(
    const coords_t &coords, std::span<const std::size_t> moved) const {
  (void)pdf_.compute_error(coords, moved);
  const vec_t &G_r = pdf_.computed_G();

  computed_S_ = vec_t::Ones(exp_Q_.size()) + Gr2Sq_ * G_r;
  double denom = computed_S_.squaredNorm();
  double scale = (denom > 1e-30) ? computed_S_.dot(exp_S_) / denom : 1.0;
  return (scale * computed_S_ - exp_S_).squaredNorm();
}

} // namespace RMC
