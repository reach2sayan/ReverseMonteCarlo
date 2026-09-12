#pragma once
#include <RMC/generators/MoveGenerator.hpp>
#include <boost/assert.hpp>
#include <ranges>
#include <utility>
#include <vector>

namespace RMC {

// Swaps the coordinates of this group with a randomly chosen candidate group
// of the same size (e.g. identity swaps of solvent molecules).
struct SwapGenerator : MoveGeneratorBase<SwapGenerator> {
  std::vector<std::vector<std::size_t>> candidates;
  Rng rng;

  SwapGenerator() = default;
  explicit SwapGenerator(std::vector<std::vector<std::size_t>> cands,
                         std::uint32_t seed = 42)
      : candidates(std::move(cands)), rng(seed) {}

  void generate(coords_t &coords,
                std::span<const std::size_t> indices) {
    if (candidates.empty()) {
      return;
    }
    const auto &other = candidates[rng.index(candidates.size())];
    BOOST_ASSERT_MSG(other.size() == indices.size(),
                     "SwapGenerator: group size mismatch");
    for (const auto [a, b] : std::views::zip(indices, other)) {
      coords.row(a).swap(coords.row(b));
    }
  }
};

// Moves the group so its centroid coincides with that of a random candidate.
struct SwapCentersGenerator : MoveGeneratorBase<SwapCentersGenerator> {
  std::vector<std::vector<std::size_t>> candidates;
  Rng rng;

  SwapCentersGenerator() = default;
  explicit SwapCentersGenerator(std::vector<std::vector<std::size_t>> cands,
                                std::uint32_t seed = 42)
      : candidates(std::move(cands)), rng(seed) {}

  void generate(coords_t &coords,
                std::span<const std::size_t> indices) {
    if (!candidates.empty()) {
      const auto &other = candidates[rng.index(candidates.size())];
      translate(coords, indices,
                centroid(coords, other) - centroid(coords, indices));
    }
  }
};

} // namespace RMC
