#pragma once
#include <RMC/core/RngGenerator.hpp>
#include <RMC/generators/MoveGenerator.hpp>
#include <algorithm>
#include <tuple>
#include <vector>

namespace RMC {

// Applies a sequence of generators to the same group in order.
template <CMoveGenerator... TMoveGenerators>
class CombinedMoveGenerator
    : MoveGeneratorBase<CombinedMoveGenerator<TMoveGenerators...>> {
  std::tuple<TMoveGenerators...> generators_;

public:
  constexpr CombinedMoveGenerator() = default;
  template <typename... Gs>
    requires(sizeof...(Gs) > 0) && (CMoveGenerator<std::decay_t<Gs>> && ...)
  constexpr explicit CombinedMoveGenerator(Gs &&...gens)
      : generators_(std::forward<Gs>(gens)...) {}
  constexpr void generate(MoveGenerator::Token tok, coords_t &coords,
                          std::span<const std::size_t> indices) {
    std::apply([&](auto &...g) { (g.generate(tok, coords, indices), ...); },
               generators_);
  }
};

template <CMoveGenerator... Gs>
CombinedMoveGenerator(Gs &&...) -> CombinedMoveGenerator<std::decay_t<Gs>...>;

// Randomly selects ONE generator from a runtime collection and applies it.
// Unlike CombinedMoveGenerator (which applies all), this picks one per step.
class MoveGeneratorCollector : MoveGeneratorBase<MoveGeneratorCollector> {
  std::vector<MoveGenerator> generators_;
  std::vector<double> cumulative_weights_;
  mutable RngBuffer<> rng_;

public:
  MoveGeneratorCollector() = default;
  explicit MoveGeneratorCollector(std::uint32_t seed) : rng_(seed) {}
  void add(MoveGenerator gen, double weight = 1.0) {
    if (weight <= 0.0) {
      weight = 1.0;
    }
    generators_.push_back(std::move(gen));
    double prev =
        cumulative_weights_.empty() ? 0.0 : cumulative_weights_.back();
    cumulative_weights_.push_back(prev + weight);
  }

  void generate(MoveGenerator::Token, coords_t &coords,
                std::span<const std::size_t> indices) {
    if (generators_.empty()) {
      return;
    }
    double r = rng_.uniform(0.0, cumulative_weights_.back());
    auto it = std::ranges::lower_bound(cumulative_weights_, r);
    std::size_t idx =
        static_cast<std::size_t>(it - cumulative_weights_.begin());
    if (idx >= generators_.size()) {
      idx = generators_.size() - 1;
    }
    generators_[idx].generate(coords, indices);
  }
};

} // namespace RMC
