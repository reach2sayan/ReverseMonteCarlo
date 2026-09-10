#pragma once
// Reader/writer for ATAT's lattice (rndstr.in / lat.in) and structure
// (str.out) text formats. Coordinates are in the lattice's "axes" frame
// (fractional w.r.t. the coordinate-system vectors); the Cartesian `axes`
// matrix is only for emitting physical (Å) coordinates.
#include <RMC/core/Types.hpp>

#include <filesystem>
#include <istream>
#include <string>
#include <utility>
#include <vector>

namespace RMC::atat {

// One lattice site: position + the species it may hold, with target
// occupations.
struct LatticeSite {
  vec3_t frac{vec3_t::Zero()};                     // axes coords
  std::vector<std::pair<std::string, double>> occ; // (species, occupation)
};

// A parsed rndstr.in / lat.in lattice.
struct AtatLattice {
  mat3_t axes{mat3_t::Identity()}; // cols: coordinate-system (Cartesian)
  mat3_t cell{mat3_t::Identity()}; // cols: cell vectors (axes coords)
  std::vector<LatticeSite> sites;
  std::vector<std::string> labels; // global species labels, alphabetical

  // Occupation index of a species (its position in `labels`); -1 if absent.
  [[nodiscard]] int occupation_index(const std::string &species) const;
};

// ---- lattice parser (throws std::runtime_error on malformed input) ----
AtatLattice parse_lattice(std::istream &in);
AtatLattice parse_lattice(const std::filesystem::path &path);

// ATAT str.out: axes rows, supercell rows (axes coords), then one
// "x y z species" line per atom (axes coords).
void write_str_out(const std::filesystem::path &path, const mat3_t &axes,
                   const mat3_t &supercell,
                   const std::vector<vec3_t> &positions,
                   const std::vector<std::string> &species);

} // namespace RMC::atat
