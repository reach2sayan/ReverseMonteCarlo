#pragma once
#include <RMC/analysis/Composition.hpp>
#include <RMC/core/BoundaryConditions.hpp>
#include <RMC/core/Types.hpp>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace RMC::analysis {

// Angular distribution function (ADF) analysis (Special Glass Structure).
// Cite: S. Zhu et al., Acta Materialia 262, 119456 (2024).
//   doi:10.1016/j.actamat.2023.119456

// Grid and options for an angular distribution function (ADF) computation.
struct AdfParams {
  double max_dis = 3.4; // bond cutoff for neighbour pairs (Å)
  int n_bins = 100;     // angle bins over [0, π]
  int smooth_range = 2; // boxcar half-width (0 disables); applied in 2 passes
};

// Per-element-triplet partial ADFs plus their sum on a shared angle grid.
// Triplet ordering matches AngularDistributionConstraint (central element
// outer, unordered leg pair inner), so write_adf output is usable as its target.
struct AdfResult {
  vec_t theta;                        // bin centres (rad), length n_bins
  vec_t total;                       // sum over triplet columns
  std::vector<LabeledCurve> partials; // one per triplet (label "A-B-C" + adf)
  Composition species;               // distinct elements (sorted) with counts
  double max_dis = 0.0;              // cutoff used
};

// Compute the ADF from an in-memory configuration. Requires a periodic box
// (volume normalization needs the cell volume).
[[nodiscard]] Result<AdfResult>
compute_adf(const coords_t &coords, const BoundaryConditions &bc,
            std::span<const std::string> elements, const AdfParams &params);

// Convenience overload reading a structure file (.pdb / .vasp / .poscar /
// POSCAR, else LAMMPS data). LAMMPS/VASP use the file's cell; PDB uses `bc`.
// `type_to_element` maps LAMMPS atom types to symbols (ignored for PDB/VASP).
[[nodiscard]] Result<AdfResult>
compute_adf(const std::filesystem::path &path, const BoundaryConditions &bc,
            const AdfParams &params,
            const std::vector<std::string> &type_to_element = {});

// Write theta and per-triplet partials as a commented whitespace table in the
// column order AngularDistributionConstraint consumes via read_xy_data. Total
// is not a column (it is the sum of partials); plot_adf.py reconstructs it.
[[nodiscard]] Result<void> write_adf(const AdfResult &a,
                                     const std::filesystem::path &path);

} // namespace RMC::analysis
