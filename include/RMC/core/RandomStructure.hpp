#pragma once
#include <RMC/core/BoundaryConditions.hpp>
#include <RMC/core/Structure.hpp>
#include <RMC/core/Types.hpp>
#include <cstdint>
#include <span>
#include <string>

namespace RMC {

// A generated random configuration plus its cubic periodic cell.
struct RandomStructure {
  AtomicStructure structure;
  mat3_t box{mat3_t::Zero()};

  [[nodiscard]] PeriodicBC periodic_bc() const { return PeriodicBC(box); }
};

// Build a random amorphous starting configuration — the entry point for the
// Special Glass Structure workflow (generate → RMC-refine against PDF/ADF →
// export). Ports MAST's `randstr`: atoms are placed on a body-centred grid of
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
