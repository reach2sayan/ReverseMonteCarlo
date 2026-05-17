#pragma once
#include <fullrmc/generators/MoveGenerator.hpp>
#include <fullrmc/core/AtomsCollector.hpp>
#include <memory>

namespace fullrmc {

// Stages the removal of the group's atoms via the shared AtomsCollector.
// The Engine commits or rolls back the removal after constraint evaluation.
// Coordinates are NOT modified; constraints must skip removed atoms.
struct RemoveGenerator : MoveGeneratorBase<RemoveGenerator> {
    std::shared_ptr<AtomsCollector> collector;

    RemoveGenerator() = default;
    explicit RemoveGenerator(std::shared_ptr<AtomsCollector> c)
        : collector(std::move(c)) {}

    void generate_impl(coords_t& /*coords*/,
                       std::span<const std::size_t> indices) {
        if (collector)
            collector->stage_removal(indices);
    }
};

} // namespace fullrmc
