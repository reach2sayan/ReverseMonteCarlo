#pragma once
#include <RMC/core/RngGenerator.hpp>
#include <RMC/core/Types.hpp>
#include <boost/random/discrete_distribution.hpp>
#include <cstddef>
#include <span>
#include <vector>

namespace RMC {

// Uniformly random group.
struct RandomSelector {
  Rng rng;
  explicit RandomSelector(std::uint32_t seed = 42) : rng(seed) {}
  std::size_t select(std::size_t n_groups) { return rng.index(n_groups); }
};

// Random group with probability proportional to `weights` (uniform while the
// weight count does not match the group count).
struct WeightedRandomSelector {
  Rng rng;
  std::vector<double> weights;

  WeightedRandomSelector() = default;
  explicit WeightedRandomSelector(std::vector<double> w, std::uint32_t seed = 42)
      : rng(seed) {
    set_weights(w);
  }

  void set_weights(std::span<const real_t> w) {
    weights.assign(w.begin(), w.end());
    dist_ = boost::random::discrete_distribution<std::size_t>(weights);
  }

  std::size_t select(std::size_t n_groups) {
    return weights.size() == n_groups ? dist_(rng.engine())
                                      : rng.index(n_groups);
  }

private:
  boost::random::discrete_distribution<std::size_t> dist_;
};

} // namespace RMC
