#pragma once
#include <Eigen/Core>
#include <RMC/selectors/GroupSelector.hpp>
#include <random>

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
struct SmartRandomSelector : SelectorBase<SmartRandomSelector> {
  const double bias_factor{1.1};
  mutable std::mt19937 rng;

private:
  Eigen::VectorXd weights_;
  mutable double weight_sum_{0.0};
  mutable std::size_t steps_since_renorm_{0};
  static constexpr std::size_t kRenormInterval = 1024;

public:
  explicit SmartRandomSelector(double bf = 1.1, std::uint32_t seed = 42)
      : bias_factor(bf), rng(seed) {}

  void initialise(std::size_t n_groups) {
    weights_ = Eigen::VectorXd::Constant(static_cast<Eigen::Index>(n_groups),
                                         1.0 / static_cast<double>(n_groups));
    weight_sum_ = 1.0;
    steps_since_renorm_ = 0;
  }

  std::size_t select(IGroupSelector::Token, std::size_t n_groups) {
    if (static_cast<std::size_t>(weights_.size()) != n_groups) {
      initialise(n_groups);
    }

    const double target =
        std::uniform_real_distribution<double>(0.0, weight_sum_)(rng);
    double cumsum = 0.0;
    for (Eigen::Index i = 0; i < weights_.size(); ++i) {
      cumsum += weights_[i];
      if (cumsum >= target) {
        return static_cast<std::size_t>(i);
      }
    }
    return static_cast<std::size_t>(weights_.size() - 1);
  }

  void feedback(IGroupSelector::Token, std::size_t group_idx, bool accepted) {
    if (weights_.size() == 0) {
      return;
    }
    const auto idx = static_cast<Eigen::Index>(group_idx);
    const double old_w = weights_[idx];
    const double new_w = accepted ? old_w * bias_factor : old_w / bias_factor;
    weights_[idx] = new_w;
    weight_sum_ += new_w - old_w;

    // Periodic renormalisation to prevent floating-point drift.
    if (++steps_since_renorm_ >= kRenormInterval) {
      weight_sum_ = weights_.sum();
      if (weight_sum_ < 1e-300) {
        weights_.setConstant(1.0 / static_cast<double>(weights_.size()));
        weight_sum_ = 1.0;
      } else {
        weights_ /= weight_sum_;
        weight_sum_ = 1.0;
      }
      steps_since_renorm_ = 0;
    }
  }

  [[nodiscard]] FORCE_INLINE Eigen::VectorXd weights() const {
    return (weight_sum_ > 0.0 && weight_sum_ != 1.0) ? weights_ / weight_sum_
                                                     : weights_;
  }
};

} // namespace RMC
