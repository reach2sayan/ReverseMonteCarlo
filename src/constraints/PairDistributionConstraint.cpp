#include <RMC/constraints/PairDistributionConstraint.hpp>
#include <boost/assert.hpp>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace RMC {

void PairDistributionConstraint::set_experimental_data(const mat_t &data) {
  BOOST_ASSERT_MSG(data.cols() >= 2,
                   "PairDistributionConstraint: data must have >= 2 columns");
  const Eigen::Index N = data.rows();
  exp_r_ = data.col(0);
  exp_G_ = data.col(1);

  r_min_ = exp_r_(0);
  r_max_ = exp_r_(N - 1);
  bin_width_ = (N > 1) ? (exp_r_(1) - exp_r_(0)) : 0.1;
  n_bins_ = static_cast<int>(N);
  computed_G_.resize(N);
}

double
PairDistributionConstraint::weight_for(const std::string &a,
                                       const std::string &b) const noexcept {
  std::string key = (a < b) ? (a + "_" + b) : (b + "_" + a);
  auto it = weights_.find(key);
  return (it != weights_.end()) ? it->second : 1.0;
}

void PairDistributionConstraint::initialise() {
  const auto idx = Eigen::ArrayXd::LinSpaced(n_bins_, 0, n_bins_ - 1);
  const auto r_lo = r_min_ + idx * bin_width_;
  const auto r_hi = r_lo + bin_width_;
  shell_vols_ = (4.0 * std::numbers::pi / 3.0) * (r_hi.cube() - r_lo.cube());
}

double
PairDistributionConstraint::compute_error(const coords_t &coords,
                                          std::span<const std::size_t>) const {
  computed_G_.setZero();
  const Eigen::Index N = coords.rows();

  vec_t local_G = vec_t::Zero(n_bins_);
  for (Eigen::Index i = 0; i < N; ++i) {
    for (Eigen::Index j = i + 1; j < N; ++j) {
      vec3_t delta = coords.row(j).transpose() - coords.row(i).transpose();
      if (bc_) {
        delta = bc_min_image(*bc_, delta);
      }
      const double d = delta.norm();
      if (d < r_min_ || d >= r_max_) {
        continue;
      }
      const int bin = static_cast<int>((d - r_min_) / bin_width_);
      if (bin < 0 || bin >= n_bins_) {
        continue;
      }
      double w = 1.0;
      if (!elements_.empty()) {
        const auto &ei = elements_[static_cast<std::size_t>(i)];
        const auto &ej = elements_[static_cast<std::size_t>(j)];
        w = weight_for(ei, ej);
      }
      local_G(bin) += 2.0 * w; // pair i–j and j–i
    }
  }
  {
    computed_G_ += local_G;
  }

  // Normalise to g(r), then compute G(r) = 4πr·ρ₀·(g(r) − 1).
  const auto idx = Eigen::ArrayXd::LinSpaced(n_bins_, 0, n_bins_ - 1);
  const auto r = r_min_ + (idx + 0.5) * bin_width_;
  computed_G_.array() =
      4.0 * std::numbers::pi * r * rho0_ *
      (computed_G_.array() / (shell_vols_.array() * rho0_ * N) - 1.0);

  // Optional scale-factor optimisation: minimise ||scale*G_c - G_e||².
  double denom = computed_G_.squaredNorm();
  double scale = (denom > 1e-30) ? computed_G_.dot(exp_G_) / denom : 1.0;
  return (scale * computed_G_ - exp_G_).squaredNorm();
}

} // namespace RMC
