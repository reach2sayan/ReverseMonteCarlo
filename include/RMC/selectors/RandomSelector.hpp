#pragma once
#include <RMC/core/RngGenerator.hpp>
#include <RMC/selectors/GroupSelector.hpp>
#include <optional>
#include <vector>

namespace RMC {

struct RandomSelector : SelectorBase<RandomSelector> {
  mutable RngBuffer<> rng;
  explicit RandomSelector(std::uint32_t seed = 42) : rng(seed) {}
  std::size_t select(IGroupSelector::Token, std::size_t n_groups) {
    return std::uniform_int_distribution<std::size_t>{0, n_groups -
                                                             1}(rng.engine());
  }
  constexpr void feedback(IGroupSelector::Token, std::size_t /*group_idx*/,
                          bool /*accepted*/) noexcept {}
};

struct WeightedRandomSelector : SelectorBase<WeightedRandomSelector> {
  mutable RngBuffer<> rng;
  std::vector<double> weights;

  WeightedRandomSelector() = default;
  explicit WeightedRandomSelector(std::vector<double> w,
                                  std::uint32_t seed = 42)
      : rng(seed), weights(std::move(w)) {
    rebuild_cache();
  }

  void set_weights(std::span<const real_t> w) {
    weights.assign(w.begin(), w.end());
    rebuild_cache();
  }

  std::size_t select(IGroupSelector::Token, std::size_t n_groups) {
    if (weights.size() != n_groups) {
      return std::uniform_int_distribution<std::size_t>{
          0, n_groups - 1}(rng.engine());
    }
    return std::invoke(*dist_cache_, rng.engine());
  }
  constexpr void feedback(IGroupSelector::Token, std::size_t /*group_idx*/,
                          bool /*accepted*/) noexcept {}

private:
  mutable std::optional<std::discrete_distribution<std::size_t>>
      dist_cache_;
  constexpr void rebuild_cache() {
    if (!weights.empty()) {
      dist_cache_.emplace(weights.begin(), weights.end());
    }
  }
};

} // namespace RMC
