#pragma once
// Symmetry-faithful enumeration of cluster instances on a supercell (axes-
// fractional frame). Ports from ATAT:
//   apply_symmetry / equivalent_mod_cell / find_equivalent_clusters
//     (atat/src/calccorr.c++:186-208, xtalutil.c++:145-203, 691-707)
//   find_all_atom_in_supercell + LatticePointInCellIterator
//     (atat/src/xtalutil.c++:262-308)
//   which_atom (atat/src/xtalutil.c++:19-26)
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

// supercell = lat.cell * sc_matrix (columns); `seed` drives random occupation.
// Single-sublattice only: all active sites share one species set, so global
// (alphabetical) occupation index == within-site index. Throws otherwise.
EnumeratedSqs enumerate(const AtatLattice &lat, const std::vector<SymOp> &sym,
                        const std::vector<RawOrbit> &raw,
                        const Eigen::Matrix3i &sc_matrix, std::uint32_t seed);

} // namespace RMC::atat
