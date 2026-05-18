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
// Weights are renormalised to sum to 1 after each update.
struct SmartRandomSelector : SelectorBase<SmartRandomSelector> {
  const double bias_factor{1.1};
  mutable std::mt19937 rng;

private:
  Eigen::VectorXd weights_;

public:
  explicit SmartRandomSelector(double bf = 1.1, std::uint32_t seed = 42)
      : bias_factor(bf), rng(seed) {}

  void initialise(std::size_t n_groups) {
    weights_ = Eigen::VectorXd::Constant(static_cast<Eigen::Index>(n_groups),
                                         1.0 / static_cast<double>(n_groups));
    rebuild_cache();
  }

  std::size_t select(IGroupSelector::Token, std::size_t n_groups) {
    if (static_cast<std::size_t>(weights_.size()) != n_groups) {
      initialise(n_groups);
    }
    if (dist_dirty_) {
      rebuild_cache();
    }
    return dist_cache_(rng);
  }

  void feedback(IGroupSelector::Token, std::size_t group_idx, bool accepted) {
    if (weights_.size() == 0)
      return;
    if (accepted) {
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
    dist_dirty_ = true;
  }

  [[nodiscard]] Eigen::VectorXd weights() const { return weights_; }

private:
  mutable std::discrete_distribution<std::size_t> dist_cache_;
  mutable bool dist_dirty_{true};
  void rebuild_cache() {
    dist_cache_ = std::discrete_distribution<std::size_t>(
        weights_.data(), weights_.data() + weights_.size());
    dist_dirty_ = false;
  }
};

} // namespace RMC
