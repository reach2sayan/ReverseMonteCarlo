#pragma once
#include <Eigen/Core>
#include <RMC/core/RngGenerator.hpp>
#include <cstddef>

namespace RMC {

// Adaptive weighted selector: groups that produced accepted moves recently
// are given higher selection probability.
//
// After each move: w[i] *= bias_factor  if accepted
//                  w[i] /= bias_factor  if rejected
//
// Sampling uses a linear cumulative scan (O(N)) with O(1) weight updates,
// avoiding the O(N) discrete_distribution rebuild that would otherwise occur
// every step.
struct SmartRandomSelector {
  double bias_factor{1.1};
  Rng rng;

  explicit SmartRandomSelector(double bf = 1.1, std::uint32_t seed = 42)
      : bias_factor(bf), rng(seed) {}

  void initialise(std::size_t n_groups) {
    weights_ = Eigen::VectorXd::Constant(static_cast<Eigen::Index>(n_groups),
                                         1.0 / static_cast<double>(n_groups));
    weight_sum_ = 1.0;
    steps_since_renorm_ = 0;
  }

  std::size_t select(std::size_t n_groups) {
    if (static_cast<std::size_t>(weights_.size()) != n_groups) {
      initialise(n_groups);
    }
    const double target = rng.uniform(0.0, weight_sum_);
    double cumsum = 0.0;
    for (Eigen::Index i = 0; i < weights_.size(); ++i) {
      cumsum += weights_[i];
      if (cumsum >= target) {
        return static_cast<std::size_t>(i);
      }
    }
    return static_cast<std::size_t>(weights_.size() - 1);
  }

  void feedback(std::size_t group_idx, bool accepted) {
    if (weights_.size() == 0) {
      return;
    }
    double &w = weights_[static_cast<Eigen::Index>(group_idx)];
    const double old_w = w;
    w = accepted ? w * bias_factor : w / bias_factor;
    weight_sum_ += w - old_w;

    // Periodic renormalisation to prevent floating-point drift.
    if (++steps_since_renorm_ >= kRenormInterval) {
      weight_sum_ = weights_.sum();
      if (weight_sum_ < 1e-300) {
        weights_.setConstant(1.0 / static_cast<double>(weights_.size()));
      } else {
        weights_ /= weight_sum_;
      }
      weight_sum_ = 1.0;
      steps_since_renorm_ = 0;
    }
  }

  [[nodiscard]] Eigen::VectorXd weights() const {
    return (weight_sum_ > 0.0 && weight_sum_ != 1.0) ? weights_ / weight_sum_
                                                     : weights_;
  }

private:
  Eigen::VectorXd weights_;
  double weight_sum_{0.0};
  std::size_t steps_since_renorm_{0};
  static constexpr std::size_t kRenormInterval = 1024;
};

} // namespace RMC
