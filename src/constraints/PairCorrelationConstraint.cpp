#include <RMC/constraints/PairCorrelationConstraint.hpp>
#include <boost/assert.hpp>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace RMC {

void PairCorrelationConstraint::set_experimental_data(const mat_t &data) {
  BOOST_ASSERT_MSG(data.cols() >= 2,
                   "PairCorrelationConstraint: need 2-column r/F(r) data");
  const Eigen::Index N = data.rows();
  exp_r_ = data.col(0);
  exp_F_ = data.col(1);
  r_min_ = exp_r_(0);
  r_max_ = exp_r_(N - 1);
  bin_width_ = (N > 1) ? (exp_r_(1) - exp_r_(0)) : 0.1;
  n_bins_ = static_cast<int>(N);
  computed_F_.resize(N);
}

void PairCorrelationConstraint::set_weight(const std::string &el1,
                                           const std::string &el2, double w) {
  std::string key = (el1 < el2) ? (el1 + "_" + el2) : (el2 + "_" + el1);
  weights_[key] = w;
}

double
PairCorrelationConstraint::weight_for(const std::string &a,
                                      const std::string &b) const noexcept {
  std::string key = (a < b) ? (a + "_" + b) : (b + "_" + a);
  auto it = weights_.find(key);
  return (it != weights_.end()) ? it->second : 1.0;
}

void PairCorrelationConstraint::initialise() {
  const auto idx = Eigen::ArrayXd::LinSpaced(n_bins_, 0, n_bins_ - 1);
  const auto r_lo = r_min_ + idx * bin_width_;
  const auto r_hi = r_lo + bin_width_;
  shell_vols_ = (4.0 * std::numbers::pi / 3.0) * (r_hi.cube() - r_lo.cube());
}

double
PairCorrelationConstraint::compute_error(const coords_t &coords,
                                         std::span<const std::size_t>) const {
  computed_F_.setZero();
  const Eigen::Index N = coords.rows();

  for (Eigen::Index i = 0; i < N; ++i) {
    for (Eigen::Index j = i + 1; j < N; ++j) {
      vec3_t delta = coords.row(j).transpose() - coords.row(i).transpose();
      if (bc_) {
        delta = bc_min_image(*bc_, delta);
      }
      double d = delta.norm();
      if (d < r_min_ || d >= r_max_) {
        continue;
      }
      int bin = static_cast<int>((d - r_min_) / bin_width_);
      if (bin < 0 || bin >= n_bins_) {
        continue;
      }

      double w = 1.0;
      if (!elements_.empty()) {
        const auto &ei = elements_[static_cast<std::size_t>(i)];
        const auto &ej = elements_[static_cast<std::size_t>(j)];
        w = weight_for(ei, ej);
      }
      computed_F_(bin) += 2.0 * w;
    }
  }

  // Normalise g(r) then F(r) = Σ w_{αβ} (g_{αβ}(r) - 1)
  for (int k = 0; k < n_bins_; ++k) {
    double g_r = computed_F_(k) / (shell_vols_(k) * rho0_ * N);
    computed_F_(k) = g_r - 1.0;
  }

  double denom = computed_F_.squaredNorm();
  double scale = (denom > 1e-30) ? computed_F_.dot(exp_F_) / denom : 1.0;
  return (scale * computed_F_ - exp_F_).squaredNorm();
}

} // namespace RMC
