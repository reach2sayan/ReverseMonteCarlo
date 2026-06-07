#pragma once
// Parsers and writers for the ATAT text formats consumed/produced by corrdump.
//
// Convention: every coordinate here is expressed in the lat.in "axes" frame
// (i.e. fractional w.r.t. the coordinate-system vectors given on the first
// line(s) of lat.in / rndstr.in). This is exactly the frame ATAT uses for
// clusters.out point coordinates and sym.out operations, so the symmetry
// enumeration in ClusterEnumerator can stay in one consistent frame. The
// Cartesian `axes` matrix is only needed to emit physical (Å) coordinates.
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
  // Matches ATAT's atom_type convention for single-sublattice systems.
  [[nodiscard]] int occupation_index(const std::string &species) const;
};

// One space-group operation, in the axes frame (as stored in sym.out).
struct SymOp {
  mat3_t rot{mat3_t::Identity()};
  vec3_t trans{vec3_t::Zero()};
};

// One point of a cluster representative.
struct ClusterPoint {
  vec3_t coord{vec3_t::Zero()}; // axes coords
  int site_type = 0;            // (#species on the site) - 2  (0 for binary)
  int func = 0;                 // cluster-function index (0..site_type)
};

// One cluster orbit representative (clusters.out block).
struct RawOrbit {
  double multiplicity = 0.0;
  double length = 0.0; // longest pair distance (Cartesian); unused until C1
  std::vector<ClusterPoint> points;
};

// ---- parsers (throw std::runtime_error on malformed input) ----
AtatLattice parse_lattice(std::istream &in);
AtatLattice parse_lattice(const std::filesystem::path &path);

std::vector<SymOp> parse_sym(std::istream &in);
std::vector<SymOp> parse_sym(const std::filesystem::path &path);

std::vector<RawOrbit> parse_clusters(std::istream &in);
std::vector<RawOrbit> parse_clusters(const std::filesystem::path &path);

std::vector<double> parse_correlations(std::istream &in);

void write_str_out(const std::filesystem::path &path, const mat3_t &axes,
                   const mat3_t &supercell,
                   const std::vector<vec3_t> &positions,
                   const std::vector<std::string> &species);

} // namespace RMC::atat
