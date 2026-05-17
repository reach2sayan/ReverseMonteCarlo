#pragma once
#include <RMC/core/AtomsCollector.hpp>
#include <RMC/generators/MoveGenerator.hpp>
#include <memory>

namespace RMC {

// Stages the removal of the group's atoms via the shared AtomsCollector.
// The Engine commits or rolls back the removal after constraint evaluation.
// Coordinates are NOT modified; constraints must skip removed atoms.
struct RemoveGenerator : MoveGeneratorBase<RemoveGenerator> {
  std::shared_ptr<AtomsCollector> collector;
  constexpr RemoveGenerator() = default;
  constexpr explicit RemoveGenerator(std::shared_ptr<AtomsCollector> c)
      : collector(std::move(c)) {}

  constexpr void generate(IMoveGenerator::Token, coords_t &,
                          std::span<const std::size_t> indices) {
    if (collector) {
      collector->stage_removal(indices);
    }
  }
};

} // namespace RMC
