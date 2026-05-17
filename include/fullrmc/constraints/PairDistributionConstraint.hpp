#pragma once
#include <fullrmc/constraints/Constraint.hpp>
#include <Eigen/Core>
#include <boost/histogram.hpp>
#include <vector>
#include <string>
#include <unordered_map>
#include <cmath>
#include <numbers>

namespace fullrmc {

// Pair distribution function G(r) = 4πr·ρ₀·(g(r) − 1) constraint.
// Compares computed G(r) to loaded experimental data via squared residuals.
//
// Weighting: uniform (unweighted) or by element-pair weights (e.g. neutron
// coherent scattering lengths).
class PairDistributionConstraint
    : public ConstraintBase<PairDistributionConstraint>
{
public:
    // Load experimental data: two-column matrix (r, G(r)).
    void set_experimental_data(const mat_t& data);

    // Per element-pair weight (e.g. b_i * b_j / <b>^2 for neutron).
    void set_weight(const std::string& el1, const std::string& el2, real_t w);

    void set_elements(const std::vector<std::string>* elements) noexcept {
        elements_ = elements;
    }
    void set_number_density(real_t rho0) noexcept { rho0_ = rho0; }

    // Build precomputed shell-volume array and histogram axis. Call once after
    // set_experimental_data().
    void initialise();

    [[nodiscard]] std::string name() const override {
        return "PairDistributionConstraint";
    }

    [[nodiscard]] real_t compute_error(const coords_t& coords,
                                        std::span<const index_t> moved) const;

    [[nodiscard]] const vec_t& computed_G() const noexcept { return computed_G_; }
    [[nodiscard]] const vec_t& experimental_G() const noexcept { return exp_G_; }

private:
    [[nodiscard]] real_t weight_for(const std::string& a,
                                     const std::string& b) const noexcept;
    void fill_histogram(const coords_t& coords,
                         boost::histogram::histogram<
                             std::tuple<boost::histogram::axis::regular<>>>& h) const;

    vec_t exp_r_;
    vec_t exp_G_;
    vec_t shell_vols_;        // (4π/3)(r_hi³ - r_lo³) per bin
    mutable vec_t computed_G_;

    real_t r_min_    {0.0};
    real_t r_max_    {10.0};
    real_t bin_width_{0.1};
    int    n_bins_   {100};
    real_t rho0_     {0.1};   // atoms/Å³

    std::unordered_map<std::string, real_t> weights_;
    const std::vector<std::string>*         elements_ = nullptr;
};

} // namespace fullrmc
