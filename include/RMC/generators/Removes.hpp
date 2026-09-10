#pragma once
#include <RMC/core/AtomsCollector.hpp>
#include <RMC/generators/MoveGenerator.hpp>

namespace RMC {

// Stages the removal of the group's atoms via the engine's AtomsCollector.
// The Engine commits or rolls back the removal after constraint evaluation.
// Coordinates are NOT modified; constraints must skip removed atoms.
struct RemoveGenerator : MoveGeneratorBase<RemoveGenerator> {
  AtomsCollector *collector{nullptr}; // non-owning
  RemoveGenerator() = default;
  explicit RemoveGenerator(AtomsCollector *c) : collector(c) {}

  void generate(coords_t &,
                std::span<const std::size_t> indices) {
    if (collector) {
      collector->stage_removal(indices);
    }
  }
};

} // namespace RMC
