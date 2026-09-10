#pragma once
#include <RMC/core/RngGenerator.hpp>
#include <RMC/core/Structure.hpp>
#include <RMC/generators/MoveGenerator.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <span>
#include <vector>

namespace RMC {

// Swaps the species (elements + atomic_numbers) of a site with a random site in
// the same sublattice carrying a different element. Coordinates unchanged (SQS /
// alloy MC). sublattices: outer = sublattice id, inner = its site indices; every
// site appears in exactly one sublattice.
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
  void generate(MoveGenerator::Token, coords_t & /*coords*/,
                std::span<const std::size_t> indices) {
    if (indices.empty() || !structure) {
      return;
    }
    const std::size_t i = indices[0];
    const int sl = site_to_sublattice_[i];
    if (sl < 0) {
      return;
    }

    const auto &sl_sites = sublattices[static_cast<std::size_t>(sl)];

    // Collect candidate sites: same sublattice, different element.
    thread_local std::vector<std::size_t> candidates;
    candidates.clear();
    candidates.reserve(sl_sites.size());
    std::ranges::copy_if(
        sl_sites, std::back_inserter(candidates), [&, i](std::size_t j) {
          // Compare via atomic_numbers (ints) to avoid per-candidate string cmp.
          return j != i &&
                 structure->atomic_numbers[static_cast<Eigen::Index>(j)] !=
                     structure->atomic_numbers[static_cast<Eigen::Index>(i)];
        });
    if (candidates.empty()) {
      return;
    }

    const std::size_t j =
        candidates[boost::random::uniform_int_distribution<std::size_t>{
            0, candidates.size() - 1}(rng.engine())];

    using std::swap;
    swap(structure->elements[i], structure->elements[j]);
    swap(structure->atomic_numbers[i], structure->atomic_numbers[j]);
  }

  // Signals Engine to save/restore species snapshot around this move.
  [[nodiscard]] constexpr bool modifies_species() const noexcept {
    return true;
  }
};

} // namespace RMC
