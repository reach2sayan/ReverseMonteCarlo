#pragma once
#include <RMC/core/BoundaryConditions.hpp>
#include <RMC/core/Types.hpp>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace RMC::analysis {

// Grid and options for a g(r) computation.
struct GrParams {
  double r_min = 0.0;
  double r_max = 10.0;
  int n_bins = 200;
  bool exclude_intra = false; // skip same-molecule pairs (needs molecule_ids)
};

// Total g(r) plus per-element-pair partials on a shared radial grid.
struct GrResult {
  vec_t r;                              // bin centers, length n_bins
  vec_t total;                          // total g(r)
  std::vector<std::string> pair_labels; // e.g. "Zr-Zr", "Zr-Cu", "Cu-Cu"
  std::vector<vec_t> partials;          // aligned with pair_labels
  std::vector<std::string> species;     // distinct elements, sorted
  std::vector<std::size_t> counts;      // atom count per species (aligned)
  double density = 0.0;                 // N / V used for normalization
};

// Compute g(r) from an in-memory configuration. Requires a periodic box (the
// partial densities ρ_b = N_b/V need the cell volume); returns a failed Result
// for a non-periodic (InfiniteBC) cell. Element labels distinguish species; the
// total uses ρ = N/V, matching PairDistributionConstraint's convention.
[[nodiscard]] Result<GrResult> compute_gr(const coords_t &coords,
                                          const BoundaryConditions &bc,
                                          std::span<const std::string> elements,
                                          const GrParams &params);

// Convenience overload: read a structure file (.pdb or LAMMPS .dat, by
// extension) and compute g(r). For LAMMPS the file's own cell is always used;
// for PDB the supplied `bc` provides the box. `type_to_element` maps LAMMPS
// atom types to element symbols (ignored for PDB).
[[nodiscard]] Result<GrResult>
compute_gr(const std::filesystem::path &path, const BoundaryConditions &bc,
           const GrParams &params,
           const std::vector<std::string> &type_to_element = {});

// Write r, total g(r) and each partial as a commented whitespace-separated
// table (the same format read_xy_data consumes).
[[nodiscard]] Result<void> write_gr(const GrResult &g,
                                    const std::filesystem::path &path);

} // namespace RMC::analysis
