#include <cmath>
#include <fullrmc/constraints/StructureFactorConstraint.hpp>
#include <numbers>
#include <stdexcept>

namespace fullrmc {

void StructureFactorConstraint::set_experimental_data(const mat_t &data) {
  if (data.cols() < 2)
    throw std::invalid_argument(
        "StructureFactorConstraint: need 2-column Q/S(Q) data");
  exp_Q_ = data.col(0);
  exp_S_ = data.col(1);
}

void StructureFactorConstraint::set_weight(const std::string &el1,
                                           const std::string &el2, double w) {
  weights_[(el1 < el2) ? (el1 + "_" + el2) : (el2 + "_" + el1)] = w;
}

void StructureFactorConstraint::initialise() {
  // Build the internal PDF constraint used to compute G(r).
  pdf_ = std::make_unique<PairDistributionConstraint>();
  pdf_->set_number_density(rho0_);

  // Build a synthetic r-axis covering [r_min_, r_max_] with step r_bin_.
  n_r_bins_ = static_cast<int>((r_max_ - r_min_) / r_bin_);
  mat_t r_data(n_r_bins_, 2);
  for (int i = 0; i < n_r_bins_; ++i) {
    double r = r_min_ + (i + 0.5) * r_bin_;
    r_data(i, 0) = r;
    r_data(i, 1) = 0.0;
  }
  pdf_->set_experimental_data(r_data);
  for (auto &[k, w] : weights_) {
    // k is "el1_el2" – split on '_'
    auto pos = k.find('_');
    if (pos != std::string::npos)
      pdf_->set_weight(k.substr(0, pos), k.substr(pos + 1), w);
  }
  if (elements_)
    pdf_->set_elements(elements_);
  pdf_->initialise();

  // Pre-compute Gr2Sq: (n_Q × n_r) matrix.
  // Gr2Sq[q_i, r_j] = dr * sin(q_i * r_j) / q_i
  const Eigen::Index nQ = exp_Q_.size();
  Gr2Sq_.resize(nQ, n_r_bins_);
  for (Eigen::Index qi = 0; qi < nQ; ++qi) {
    double q = exp_Q_(qi);
    for (int ri = 0; ri < n_r_bins_; ++ri) {
      double r = r_min_ + (ri + 0.5) * r_bin_;
      Gr2Sq_(qi, ri) = (q > 1e-10) ? r_bin_ * std::sin(q * r) / q : 0.0;
    }
  }
  computed_S_.resize(nQ);
}

double StructureFactorConstraint::compute_error(
    const coords_t &coords, std::span<const std::size_t> moved) const {
  // Compute G(r) via the embedded PDF constraint.
  pdf_->compute_after_move(coords, moved);
  const vec_t &G_r = pdf_->computed_G();

  // S(Q) = 1 + Gr2Sq * G(r) (matrix-vector product).
  computed_S_ = vec_t::Ones(exp_Q_.size()) + Gr2Sq_ * G_r;

  double denom = computed_S_.squaredNorm();
  double scale = (denom > 1e-30) ? computed_S_.dot(exp_S_) / denom : 1.0;
  return (scale * computed_S_ - exp_S_).squaredNorm();
}

} // namespace fullrmc
