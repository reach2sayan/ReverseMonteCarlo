#include <fullrmc/constraints/PairDistributionConstraint.hpp>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace fullrmc {

void PairDistributionConstraint::set_experimental_data(const mat_t& data) {
    if (data.cols() < 2)
        throw std::invalid_argument("PairDistributionConstraint: data must have >= 2 columns");
    const Eigen::Index N = data.rows();
    exp_r_ = data.col(0);
    exp_G_ = data.col(1);

    r_min_     = exp_r_(0);
    r_max_     = exp_r_(N - 1);
    bin_width_ = (N > 1) ? (exp_r_(1) - exp_r_(0)) : 0.1;
    n_bins_    = static_cast<int>(N);
    computed_G_.resize(N);
}

void PairDistributionConstraint::set_weight(const std::string& el1,
                                              const std::string& el2,
                                              real_t w) {
    std::string key = (el1 < el2) ? (el1 + "_" + el2) : (el2 + "_" + el1);
    weights_[key] = w;
}

real_t PairDistributionConstraint::weight_for(const std::string& a,
                                                const std::string& b) const noexcept {
    std::string key = (a < b) ? (a + "_" + b) : (b + "_" + a);
    auto it = weights_.find(key);
    return (it != weights_.end()) ? it->second : 1.0;
}

void PairDistributionConstraint::initialise() {
    // Pre-compute shell volumes: V_shell[i] = (4π/3)(r_hi³ - r_lo³)
    shell_vols_.resize(n_bins_);
    for (int i = 0; i < n_bins_; ++i) {
        real_t r_lo = r_min_ + i * bin_width_;
        real_t r_hi = r_lo + bin_width_;
        shell_vols_(i) = (4.0 * std::numbers::pi / 3.0) * (r_hi*r_hi*r_hi - r_lo*r_lo*r_lo);
    }
}

real_t PairDistributionConstraint::compute_error(
    const coords_t& coords,
    std::span<const index_t> /*moved*/) const
{
    computed_G_.setZero();
    const Eigen::Index N = coords.rows();

    // Accumulate weighted pair counts per bin.
    for (Eigen::Index i = 0; i < N; ++i) {
        for (Eigen::Index j = i + 1; j < N; ++j) {
            vec3_t delta = coords.row(j).transpose() - coords.row(i).transpose();
            if (bc_) delta = bc_min_image(*bc_, delta);
            real_t d = delta.norm();
            if (d < r_min_ || d >= r_max_) continue;
            int bin = static_cast<int>((d - r_min_) / bin_width_);
            if (bin < 0 || bin >= n_bins_) continue;

            real_t w = 1.0;
            if (elements_) {
                const auto& ei = (*elements_)[static_cast<std::size_t>(i)];
                const auto& ej = (*elements_)[static_cast<std::size_t>(j)];
                w = weight_for(ei, ej);
            }
            computed_G_(bin) += 2.0 * w;  // pair i–j and j–i
        }
    }

    // Normalise to g(r), then compute G(r) = 4πr·ρ₀·(g(r) − 1).
    for (int k = 0; k < n_bins_; ++k) {
        real_t r      = r_min_ + (k + 0.5) * bin_width_;
        real_t g_r    = computed_G_(k) / (shell_vols_(k) * rho0_ * N);
        computed_G_(k) = 4.0 * std::numbers::pi * r * rho0_ * (g_r - 1.0);
    }

    // Optional scale-factor optimisation: minimise ||scale*G_c - G_e||².
    real_t denom = computed_G_.squaredNorm();
    real_t scale = (denom > 1e-30) ? computed_G_.dot(exp_G_) / denom : 1.0;
    return (scale * computed_G_ - exp_G_).squaredNorm();
}

} // namespace fullrmc
