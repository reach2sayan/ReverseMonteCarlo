#pragma once
#include <Eigen/Core>
#include <RMC/selectors/GroupSelector.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/count.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <random>

namespace RMC {

// Adaptive weighted selector: groups that produced accepted moves recently
// are given higher selection probability.
//
// After each move: w[i] *= bias_factor  if accepted
//                  w[i] /= bias_factor  if rejected
// Weights are renormalised to sum to 1 after each update.
struct SmartRandomSelector : SelectorBase<SmartRandomSelector> {
  const double bias_factor{1.1};
  mutable std::mt19937 rng;

private:
  Eigen::VectorXd weights_;

public:
  explicit SmartRandomSelector(double bf = 1.1, std::uint32_t seed = 42)
      : bias_factor(bf), rng(seed) {}

  FORCE_INLINE void initialise(std::size_t n_groups) {
    weights_ = Eigen::VectorXd::Constant(static_cast<Eigen::Index>(n_groups),
                                         1.0 / static_cast<double>(n_groups));
  }

  std::size_t select(IGroupSelector::Token, std::size_t n_groups) {
    if (static_cast<std::size_t>(weights_.size()) != n_groups) {
      initialise(n_groups);
    }
    std::vector w(weights_.data(), weights_.data() + weights_.size());
    std::discrete_distribution<std::size_t> dist(w.begin(), w.end());
    return dist(rng);
  }

  void feedback(IGroupSelector::Token, std::size_t group_idx, bool accepted) {
    if (weights_.size() == 0) {
      return;
    } else if (accepted) {
      weights_[static_cast<Eigen::Index>(group_idx)] *= bias_factor;
    } else {
      weights_[static_cast<Eigen::Index>(group_idx)] /= bias_factor;
    }

    double total = weights_.sum();
    if (total < 1e-300) {
      weights_.setConstant(1.0 / static_cast<double>(weights_.size()));
    } else {
      weights_ /= total;
    }
  }

  [[nodiscard]] Eigen::VectorXd weights() const { return weights_; }
};

} // namespace RMC
