#pragma once
#include <fullrmc/generators/MoveGenerator.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <vector>
#include <stdexcept>

namespace fullrmc {

// Swaps the coordinates of this group with a randomly chosen group
// from a provided pool of candidate groups (same size).
// Typically used for identity swaps of solvent molecules.
struct SwapGenerator : MoveGeneratorBase<SwapGenerator> {
    // Each entry is a list of atom indices for one candidate group.
    std::vector<std::vector<index_t>> candidates;
    mutable boost::random::mt19937 rng;

    SwapGenerator() = default;
    explicit SwapGenerator(std::vector<std::vector<index_t>> cands,
                            std::uint32_t seed = 42)
        : candidates(std::move(cands)), rng(seed) {}

    void generate_impl(coords_t& coords,
                       std::span<const index_t> indices) {
        if (candidates.empty()) return;

        boost::random::uniform_int_distribution<std::size_t>
            pick(0, candidates.size() - 1);
        const auto& other = candidates[pick(rng)];

        if (other.size() != indices.size())
            throw std::runtime_error("SwapGenerator: group size mismatch");

        for (std::size_t k = 0; k < indices.size(); ++k) {
            coords_t::RowXpr ra = coords.row(indices[k]);
            coords_t::RowXpr rb = coords.row(other[k]);
            Eigen::RowVector3d tmp = ra;
            ra = rb;
            rb = tmp;
        }
    }
};

} // namespace fullrmc
