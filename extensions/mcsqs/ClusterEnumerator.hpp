#pragma once
// Symmetry-faithful enumeration of cluster instances on a supercell — the one
// piece ATAT cannot hand us (no binary emits per-supercell site tuples).
//
// Ports, into the axes-fractional frame:
//   apply_symmetry / equivalent_mod_cell / find_equivalent_clusters
//     (atat/src/calccorr.c++:186-208, xtalutil.c++:145-203, 691-707)
//   find_all_atom_in_supercell + LatticePointInCellIterator
//     (atat/src/xtalutil.c++:262-308)
//   which_atom (atat/src/xtalutil.c++:19-26)
//
// Output: a supercell AtomicStructure (random initial occupation honouring the
// rndstr.in composition) plus ClusterOrbits carrying site-index instances,
// per-point func/site_type, and the random-state target correlation.
#include "AtatFormats.hpp"

#include <RMC/constraints/ClusterCorrelationConstraint.hpp>
#include <RMC/core/Structure.hpp>
#include <RMC/core/Types.hpp>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace RMC::atat {

struct EnumeratedSqs {
  AtomicStructure structure;        // supercell with random initial occupation
  std::vector<ClusterOrbit> orbits; // instances + func/site_type + target
  std::unordered_map<std::string, int> occ_index; // element → occupation index
  CorrFuncTable table;              // trigonometric site basis
  // Kept for writing str.out (the corrdump validation oracle / bestsqs.out):
  mat3_t axes{mat3_t::Identity()};
  mat3_t supercell{mat3_t::Identity()}; // axes coords
  std::vector<vec3_t> frac_positions;   // axes coords, per atom
};

// supercell = lat.cell * sc_matrix (columns). Diagonal sc_matrix = the common
// nx×ny×nz case. `seed` drives the random initial occupation.
//
// Single-sublattice assumption: every active site shares one species set, so the
// global (alphabetical) occupation index equals the within-site index. Throws if
// a heterogeneous-sublattice lattice is detected (documented limitation).
EnumeratedSqs enumerate(const AtatLattice &lat, const std::vector<SymOp> &sym,
                        const std::vector<RawOrbit> &raw,
                        const Eigen::Matrix3i &sc_matrix, std::uint32_t seed);

} // namespace RMC::atat
