#pragma once
// SQS problem setup on seitz: the parent lattice and its symmetry-distinct
// cluster orbits (seitz::alloy), a supercell with a random occupation, and
// every orbit instance mapped onto supercell sites.
#include <boost/describe/class.hpp>
#include "AtatFormats.hpp"

#include <RMC/constraints/ClusterCorrelationConstraint.hpp>
#include <RMC/core/Structure.hpp>
#include <RMC/core/Types.hpp>

#include <boost/container/flat_map.hpp>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace RMC::atat {

// Largest cluster diameter (Å) per body order: {{2, d2}, {3, d3}}.
using Diameters = boost::container::flat_map<int, double>;

struct EnumeratedSqs {
  AtomicStructure structure;        // supercell with a random initial occupation
  std::vector<ClusterOrbit> orbits; // instances + funcs/site_types + target
  std::unordered_map<std::string, int> occ_index; // label → global rank
  CorrFuncTable table;              // one site-basis block per sublattice

  // For write_str_out (bestsqs.out, ATAT's str.out format):
  mat3_t axes{mat3_t::Identity()};
  mat3_t supercell{mat3_t::Identity()}; // axes coords
  std::vector<vec3_t> frac_positions;   // axes coords, per atom
};
BOOST_DESCRIBE_STRUCT(EnumeratedSqs, (),
                      (structure, orbits, occ_index, table, axes, supercell,
                       frac_positions))

// supercell = lat.cell · sc (columns); `seed` drives the random occupation.
// Sites with equal species sets form one sublattice (residue "SL<id>").
// Throws std::runtime_error when seitz rejects the lattice or the supercell.
EnumeratedSqs enumerate(const AtatLattice &lat, const Eigen::Matrix3i &sc,
                        const Diameters &diameters, std::uint32_t seed);

} // namespace RMC::atat
