#pragma once
#include <boost/assert.hpp>
#include <fullrmc/generators/MoveGenerator.hpp>
#include <random>
#include <ranges>
#include <stdexcept>
#include <vector>

namespace fullrmc {

// Swaps the coordinates of this group with a randomly chosen group
// from a provided pool of candidate groups (same size).
// Typically used for identity swaps of solvent molecules.
struct SwapGenerator : MoveGeneratorBase<SwapGenerator> {
  std::vector<std::vector<std::size_t>> candidates;
  mutable std::mt19937 rng;

  SwapGenerator() = default;
  explicit SwapGenerator(std::vector<std::vector<std::size_t>> cands,
                         std::uint32_t seed = 42)
      : candidates(std::move(cands)), rng(seed) {}

  void generate(IMoveGenerator::Token, coords_t &coords,
                std::span<const std::size_t> indices) {
    if (candidates.empty()) {
      return;
    }
    std::uniform_int_distribution<std::size_t> pick(0, candidates.size() - 1);
    const auto &other = candidates[pick(rng)];
    BOOST_ASSERT_MSG(other.size() == indices.size(),
                     "SwapGenerator: group size mismatch");
    for (std::size_t k = 0; k < indices.size(); ++k) {
      coords.row(static_cast<Eigen::Index>(indices[k]))
          .swap(coords.row(static_cast<Eigen::Index>(other[k])));
    }
  }
};

} // namespace fullrmc
