#pragma once
#include <RMC/generators/MoveGenerator.hpp>
#include <tuple>

namespace RMC {

// Applies a sequence of generators to the same group in order.
// Generator types are fixed at compile time in a std::tuple — zero overhead,
// no type erasure, direct dispatch.
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

  constexpr void generate(IMoveGenerator::Token tok, coords_t &coords,
                          std::span<const std::size_t> indices) {
    std::apply([&](auto &...g) { (g.generate(tok, coords, indices), ...); },
               generators_);
  }
};

template <CMoveGenerator... Gs>
CombinedMoveGenerator(Gs &&...) -> CombinedMoveGenerator<std::decay_t<Gs>...>;

} // namespace RMC
