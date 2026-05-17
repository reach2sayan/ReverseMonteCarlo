#pragma once
#include <RMC/selectors/GroupSelector.hpp>
#include <random>
#include <vector>

namespace RMC {

struct RandomSelector : SelectorBase<RandomSelector> {
  mutable std::mt19937 rng;
  explicit RandomSelector(std::uint32_t seed = 42) : rng(seed) {}
  std::size_t select(IGroupSelector::Token, std::size_t n_groups) {
    std::uniform_int_distribution<std::size_t> dist(0, n_groups - 1);
    return dist(rng);
  }
  constexpr void feedback(IGroupSelector::Token, std::size_t /*group_idx*/,
                          bool /*accepted*/) noexcept {}
};

struct WeightedRandomSelector : SelectorBase<WeightedRandomSelector> {
  mutable std::mt19937 rng;
  std::vector<double> weights;

  WeightedRandomSelector() = default;
  explicit WeightedRandomSelector(std::vector<double> w,
                                  std::uint32_t seed = 42)
      : rng(seed), weights(std::move(w)) {}

  constexpr void set_weights(std::span<const real_t> w) {
    weights.assign(w.begin(), w.end());
  }

  std::size_t select(IGroupSelector::Token, std::size_t n_groups) {
    if (weights.size() != n_groups) {
      std::uniform_int_distribution<std::size_t> u(0, n_groups - 1);
      return u(rng);
    }
    std::discrete_distribution<std::size_t> dist(weights.begin(),
                                                 weights.end());
    return dist(rng);
  }
  constexpr void feedback(IGroupSelector::Token, std::size_t /*group_idx*/,
                          bool /*accepted*/) noexcept {}
};

} // namespace RMC
