#pragma once
#include <RMC/core/BoundaryConditions.hpp>
#include <RMC/core/Structure.hpp>
#include <RMC/core/Types.hpp>
#include <cstdint>
#include <span>
#include <string>

namespace RMC {

// Random amorphous structure generation for the Special Glass Structure
// workflow (`--gen random`).
// Cite: S. Zhu et al., "Special glass structures for first-principles studies
//   of bulk metallic glasses," Acta Materialia 262, 119456 (2024).
//   doi:10.1016/j.actamat.2023.119456

// A generated random configuration plus its cubic periodic cell.
struct RandomStructure {
  AtomicStructure structure;
  mat3_t box{mat3_t::Zero()};
  [[nodiscard]] PeriodicBC periodic_bc() const { return PeriodicBC(box); }
};

// Build a random amorphous starting configuration. Atoms placed on a
// body-centred grid of n = ceil((N/2)^(1/3)) cells per side in a cubic cell of
// side n·spacing; exact per-element composition assigned to sites in random
// order. Cartesian coordinates.
// elements and counts: same non-empty length, positive counts. spacing: grid
// pitch (≈ nearest-neighbour distance, Å).
[[nodiscard]] Result<RandomStructure>
make_random_amorphous(std::span<const std::string> elements,
                      std::span<const std::size_t> counts, double spacing,
                      std::uint32_t seed);

} // namespace RMC
