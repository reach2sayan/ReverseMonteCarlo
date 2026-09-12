#pragma once
#include <RMC/core/Structure.hpp>
#include <RMC/generators/MoveGenerator.hpp>
#include <algorithm>
#include <ranges>
#include <span>
#include <utility>
#include <vector>

namespace RMC {

// Swaps the species (elements + atomic_numbers) of a site with a random site in
// the same sublattice carrying a different element. Coordinates unchanged (SQS /
// alloy MC). sublattices: outer = sublattice id, inner = its site indices; every
// site appears in exactly one sublattice.
struct SpeciesSwapGenerator : MoveGeneratorBase<SpeciesSwapGenerator> {
  AtomicStructure *structure{nullptr};               // non-owning
  std::vector<std::vector<std::size_t>> sublattices; // grouped site indices
  Rng rng;

private:
  std::vector<int> site_to_sublattice_; // reverse map: site → sublattice id

public:
  SpeciesSwapGenerator() = default;
  explicit SpeciesSwapGenerator(AtomicStructure &str,
                                std::vector<std::vector<std::size_t>> subs,
                                std::uint32_t seed = 42)
      : structure(&str), sublattices(std::move(subs)), rng(seed),
        site_to_sublattice_(str.size(), -1) {
    for (const auto [sl, sites] : sublattices | std::views::enumerate) {
      for (const std::size_t site : sites) {
        site_to_sublattice_[site] = static_cast<int>(sl);
      }
    }
  }

  // indices = {i}: swap i's species with a random site of its sublattice whose
  // species differs (compared by atomic number, not string).
  void generate(coords_t & /*coords*/,
                std::span<const std::size_t> indices) {
    if (indices.empty() || !structure || site_to_sublattice_[indices[0]] < 0) {
      return;
    }
    const std::size_t i = indices[0];
    const auto &codes = structure->atomic_numbers;
    const auto differs = [&](std::size_t j) { return codes[j] != codes[i]; };
    const auto &sites =
        sublattices[static_cast<std::size_t>(site_to_sublattice_[i])];
    const auto n = std::ranges::count_if(sites, differs);
    if (n == 0) {
      return;
    }
    auto candidates = sites | std::views::filter(differs);
    const std::size_t j = *std::ranges::next(
        candidates.begin(),
        static_cast<std::ptrdiff_t>(rng.index(static_cast<std::size_t>(n))));
    std::swap(structure->elements[i], structure->elements[j]);
    std::swap(structure->atomic_numbers[i], structure->atomic_numbers[j]);
  }

  // Signals Engine to save/restore species snapshot around this move.
  [[nodiscard]] constexpr bool modifies_species() const noexcept {
    return true;
  }
};

} // namespace RMC
