#pragma once
#include <RMC/core/RngGenerator.hpp>
#include <RMC/generators/MoveGenerator.hpp>
#include <boost/assert.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <ranges>
#include <stdexcept>
#include <vector>

namespace RMC {

// Swaps the coordinates of this group with a randomly chosen group
// from a provided pool of candidate groups (same size).
// Typically used for identity swaps of solvent molecules.
struct SwapGenerator : MoveGeneratorBase<SwapGenerator> {
  std::vector<std::vector<std::size_t>> candidates;
  mutable RngBuffer<> rng;

  SwapGenerator() = default;
  explicit SwapGenerator(std::vector<std::vector<std::size_t>> cands,
                         std::uint32_t seed = 42)
      : candidates(std::move(cands)), rng(seed) {}

  void generate(IMoveGenerator::Token, coords_t &coords,
                std::span<const std::size_t> indices) {
    if (candidates.empty()) {
      return;
    }
    const auto &other =
        candidates[boost::random::uniform_int_distribution<std::size_t>{
            0, candidates.size() - 1}(rng.engine())];
    BOOST_ASSERT_MSG(other.size() == indices.size(),
                     "SwapGenerator: group size mismatch");
    for (auto [iindex, otherindex] : std::views::zip(indices, other)) {
      coords.row(static_cast<Eigen::Index>(iindex))
          .swap(coords.row(static_cast<Eigen::Index>(otherindex)));
    }
  }
};

// Translates the entire group so its centroid coincides with the centroid of a
// randomly chosen candidate group.
struct SwapCentersGenerator : MoveGeneratorBase<SwapCentersGenerator> {
  std::vector<std::vector<std::size_t>> candidates;
  mutable RngBuffer<> rng;

  SwapCentersGenerator() = default;
  explicit SwapCentersGenerator(std::vector<std::vector<std::size_t>> cands,
                                std::uint32_t seed = 42)
      : candidates(std::move(cands)), rng(seed) {}

  void generate(IMoveGenerator::Token, coords_t &coords,
                std::span<const std::size_t> indices) {
    if (candidates.empty())
      return;
    const auto &other =
        candidates[boost::random::uniform_int_distribution<std::size_t>{
            0, candidates.size() - 1}(rng.engine())];
    vec3_t my_center = centroid(coords, indices);
    vec3_t their_center = centroid(coords, other);
    vec3_t delta = their_center - my_center;
    coords(indices, Eigen::all).rowwise() += delta.transpose();
  }
};

} // namespace RMC
