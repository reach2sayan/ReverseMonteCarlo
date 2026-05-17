#pragma once
#include <fullrmc/generators/MoveGenerator.hpp>
#include <vector>

namespace fullrmc {

// Applies a sequence of generators to the same group in order.
// Useful for combined translation + rotation moves.
class CombinedMoveGenerator : MoveGeneratorBase<CombinedMoveGenerator> {
  std::vector<IMoveGenerator> generators;

public:
  constexpr CombinedMoveGenerator() = default;
  constexpr explicit CombinedMoveGenerator(std::vector<IMoveGenerator> gens)
      : generators(std::move(gens)) {}
  constexpr void generate(IMoveGenerator::Token, coords_t &coords,
                          std::span<const std::size_t> indices) {
    std::ranges::for_each(
        generators, [&](IMoveGenerator &g) { g.generate(coords, indices); });
  }
};

} // namespace fullrmc
