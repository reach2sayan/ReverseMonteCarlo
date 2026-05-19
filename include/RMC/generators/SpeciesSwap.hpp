#pragma once
#include <RMC/core/RngGenerator.hpp>
#include <RMC/core/Structure.hpp>
#include <RMC/generators/MoveGenerator.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <span>
#include <vector>

namespace RMC {

// Swaps the species (elements + atomic_numbers) of a selected site with a
// randomly chosen site from the same sublattice that carries a different
// element.  Coordinates are never modified — lattice sites are fixed, only
// occupancy changes (SQS / alloy MC).
//
// Usage:
//   SpeciesSwapGenerator gen{engine.structure(), sublattices};
//   Group g{"site_0", {0}, IMoveGenerator{gen}};
//   engine.add_group(g);
//
// sublattices: outer index = sublattice id; inner = site indices in that
// sublattice.  Every site index must appear in exactly one sublattice.
struct SpeciesSwapGenerator : MoveGeneratorBase<SpeciesSwapGenerator> {
  AtomicStructure *structure;                        // non-owning
  std::vector<std::vector<std::size_t>> sublattices; // grouped site indices
  mutable RngBuffer<> rng;

private:
  std::vector<int> site_to_sublattice_; // reverse map: site → sublattice id

public:
  SpeciesSwapGenerator() = default;
  explicit SpeciesSwapGenerator(AtomicStructure &str,
                                std::vector<std::vector<std::size_t>> subs,
                                std::uint32_t seed = 42)
      : structure(&str), sublattices(std::move(subs)), rng(seed) {
    const std::size_t n = str.size();
    site_to_sublattice_.assign(n, -1);
    for (int sl = 0; sl < static_cast<int>(sublattices.size()); ++sl)
      for (std::size_t site : sublattices[static_cast<std::size_t>(sl)])
        site_to_sublattice_[site] = sl;
  }

  // Required by CMoveGenerator.
  // indices must contain exactly one element: the selected site i.
  // Picks a random site j from the same sublattice with a different element
  // and swaps their species labels.
  void generate(IMoveGenerator::Token, coords_t & /*coords*/,
                std::span<const std::size_t> indices) {
    if (indices.empty() || !structure)
      return;
    const std::size_t i = indices[0];
    const int sl = site_to_sublattice_[i];
    if (sl < 0)
      return;

    const auto &sl_sites = sublattices[static_cast<std::size_t>(sl)];

    // Collect candidate sites: same sublattice, different element.
    thread_local std::vector<std::size_t> candidates;
    candidates.clear();
    for (std::size_t j : sl_sites)
      if (j != i && structure->elements[j] != structure->elements[i])
        candidates.push_back(j);
    if (candidates.empty())
      return;

    const std::size_t j =
        candidates[boost::random::uniform_int_distribution<std::size_t>{
            0, candidates.size() - 1}(rng.engine())];

    std::swap(structure->elements[i], structure->elements[j]);
    std::swap(structure->atomic_numbers[i], structure->atomic_numbers[j]);
  }

  // Signals Engine to save/restore species snapshot around this move.
  [[nodiscard]] constexpr bool modifies_species() const noexcept {
    return true;
  }
};

} // namespace RMC
