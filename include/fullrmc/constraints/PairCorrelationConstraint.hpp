#pragma once
#include <fullrmc/constraints/Constraint.hpp>
#include <boost/histogram.hpp>
#include <vector>
#include <string>
#include <unordered_map>
#include <cmath>
#include <numbers>

namespace fullrmc {

// Total-scattering pair correlation F(r):
//   F(r) = Σ_{α,β} w_{αβ} · [g_{αβ}(r) − 1]
// compared to experimental F(r) via squared residuals.
class PairCorrelationConstraint
    : public ConstraintBase<PairCorrelationConstraint>
{
public:
    void set_experimental_data(const mat_t& data);
    void set_weight(const std::string& el1, const std::string& el2, real_t w);
    void set_elements(const std::vector<std::string>* elements) noexcept {
        elements_ = elements;
    }
    void set_number_density(real_t rho0) noexcept { rho0_ = rho0; }
    void initialise();

    [[nodiscard]] std::string name() const override {
        return "PairCorrelationConstraint";
    }

    [[nodiscard]] real_t compute_error(const coords_t& coords,
                                        std::span<const index_t> moved) const;

private:
    [[nodiscard]] real_t weight_for(const std::string& a,
                                     const std::string& b) const noexcept;

    vec_t exp_r_, exp_F_;
    vec_t shell_vols_;
    mutable vec_t computed_F_;
    real_t r_min_{0.0}, r_max_{10.0}, bin_width_{0.1};
    int    n_bins_{100};
    real_t rho0_{0.1};

    std::unordered_map<std::string, real_t> weights_;
    const std::vector<std::string>*         elements_ = nullptr;
};

} // namespace fullrmc
