#pragma once
#include <fullrmc/generators/MoveGenerator.hpp>
#include <vector>

namespace fullrmc {

// Applies a sequence of generators to the same group in order.
// Useful for combined translation + rotation moves.
struct CombinedMoveGenerator : MoveGeneratorBase<CombinedMoveGenerator> {
    std::vector<std::shared_ptr<IMoveGenerator>> generators;

    CombinedMoveGenerator() = default;
    explicit CombinedMoveGenerator(
        std::vector<std::shared_ptr<IMoveGenerator>> gens)
        : generators(std::move(gens)) {}

    void generate_impl(coords_t& coords,
                       std::span<const std::size_t> indices) {
        for (auto& g : generators)
            g->generate(coords, indices);
    }
};

} // namespace fullrmc
