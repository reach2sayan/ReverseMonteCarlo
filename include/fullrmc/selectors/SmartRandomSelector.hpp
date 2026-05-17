#pragma once
#include <fullrmc/selectors/RandomSelector.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/count.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/random/discrete_distribution.hpp>
#include <Eigen/Core>

namespace fullrmc {

// Adaptive weighted selector: groups that produced accepted moves recently
// are given higher selection probability (machine-learning-like bias).
//
// After each move: w[i] *= bias_factor  if accepted
//                  w[i] /= bias_factor  if rejected
// Weights are renormalised to sum to 1 after each update.
struct SmartRandomSelector : IGroupSelector {
    real_t bias_factor{1.1};
    mutable boost::random::mt19937 rng;

private:
    Eigen::VectorXd weights_;
    std::size_t     last_selected_{0};

public:
    explicit SmartRandomSelector(real_t bf = 1.1, std::uint32_t seed = 42)
        : bias_factor(bf), rng(seed) {}

    void initialise(std::size_t n_groups) {
        weights_ = Eigen::VectorXd::Constant(n_groups, 1.0 / n_groups);
    }

    std::size_t select(std::size_t n_groups) override {
        if (static_cast<std::size_t>(weights_.size()) != n_groups)
            initialise(n_groups);

        // Sample from discrete distribution built from current weights.
        std::vector<real_t> w(weights_.data(),
                               weights_.data() + weights_.size());
        boost::random::discrete_distribution<std::size_t> dist(w.begin(), w.end());
        last_selected_ = dist(rng);
        return last_selected_;
    }

    void feedback(std::size_t group_idx, bool accepted) override {
        if (weights_.size() == 0) return;
        if (accepted)
            weights_[static_cast<Eigen::Index>(group_idx)] *= bias_factor;
        else
            weights_[static_cast<Eigen::Index>(group_idx)] /= bias_factor;

        // Renormalise; guard against numerical underflow.
        real_t total = weights_.sum();
        if (total < 1e-300)
            weights_.setConstant(1.0 / weights_.size());
        else
            weights_ /= total;
    }

    [[nodiscard]] Eigen::VectorXd weights() const { return weights_; }

    std::unique_ptr<IGroupSelector> clone() const override {
        return std::make_unique<SmartRandomSelector>(*this);
    }
};

} // namespace fullrmc
