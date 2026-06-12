#pragma once
#include <RMC/core/BoundaryConditions.hpp>
#include <RMC/core/Structure.hpp>
#include <RMC/core/Types.hpp>
#include <cstdint>
#include <span>
#include <string>

namespace RMC {

// Random amorphous structure generation for the Special Glass Structure
// workflow (exposed as `--gen random`).
//
// If you use this code in your research, please cite:
//   S. Zhu, J. Schroers, S. Curtarolo, H. Eckert, and A. van de Walle,
//   "Special glass structures for first-principles studies of bulk metallic
//   glasses," Acta Materialia 262, 119456 (2024).
//   doi:10.1016/j.actamat.2023.119456

// A generated random configuration plus its cubic periodic cell.
struct RandomStructure {
  AtomicStructure structure;
  mat3_t box{mat3_t::Zero()};
  [[nodiscard]] PeriodicBC periodic_bc() const { return PeriodicBC(box); }
};

// Build a random amorphous starting configuration — the entry point for the
// Special Glass Structure workflow (generate → RMC-refine against PDF/ADF →
// export). Atoms are placed on a body-centred grid of
// `n = ceil((N/2)^(1/3))` cells per side in a cubic cell of side `n·spacing`,
// and the exact per-element composition is assigned to those sites in random
// order (so the species arrangement is fully shuffled). Coordinates are stored
// as Cartesian values, matching the rest of the codebase.
//
// `elements` and `counts` must be the same non-empty length with positive
// counts. `spacing` is the grid pitch (≈ nearest-neighbour distance, Å).
[[nodiscard]] Result<RandomStructure>
make_random_amorphous(std::span<const std::string> elements,
                      std::span<const std::size_t> counts, double spacing,
                      std::uint32_t seed);

} // namespace RMC
