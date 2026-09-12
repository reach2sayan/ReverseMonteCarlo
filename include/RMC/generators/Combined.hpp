#pragma once
#include <RMC/generators/MoveGenerator.hpp>
#include <boost/random/discrete_distribution.hpp>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace RMC {

// Applies a sequence of generators to the same group in order.
template <CMoveGenerator... TMoveGenerators>
class CombinedMoveGenerator
    : public MoveGeneratorBase<CombinedMoveGenerator<TMoveGenerators...>> {
  std::tuple<TMoveGenerators...> generators_;

public:
  constexpr CombinedMoveGenerator() = default;
  template <typename... Gs>
    requires(sizeof...(Gs) > 0) && (CMoveGenerator<std::decay_t<Gs>> && ...)
  constexpr explicit CombinedMoveGenerator(Gs &&...gens)
      : generators_(std::forward<Gs>(gens)...) {}
  void generate(coords_t &coords,
                std::span<const std::size_t> indices) {
    std::apply([&](auto &...g) { (g.generate(coords, indices), ...); },
               generators_);
  }
};

template <CMoveGenerator... Gs>
CombinedMoveGenerator(Gs &&...) -> CombinedMoveGenerator<std::decay_t<Gs>...>;

// Applies ONE generator per step, picked by weight from a runtime collection
// (CombinedMoveGenerator applies all of them).
class MoveGeneratorCollector : public MoveGeneratorBase<MoveGeneratorCollector> {
  std::vector<MoveGenerator> generators_;
  std::vector<double> weights_;
  boost::random::discrete_distribution<std::size_t> pick_;
  Rng rng_;

public:
  MoveGeneratorCollector() = default;
  explicit MoveGeneratorCollector(std::uint32_t seed) : rng_(seed) {}
  void add(MoveGenerator gen, double weight = 1.0) {
    generators_.push_back(std::move(gen));
    weights_.push_back(weight > 0.0 ? weight : 1.0);
    pick_ = boost::random::discrete_distribution<std::size_t>(weights_);
  }

  void generate(coords_t &coords,
                std::span<const std::size_t> indices) {
    if (!generators_.empty()) {
      generators_[pick_(rng_.engine())].generate(coords, indices);
    }
  }
};

} // namespace RMC
