#pragma once
#include <RMC/analysis/Composition.hpp>
#include <RMC/core/BoundaryConditions.hpp>
#include <RMC/core/Types.hpp>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace RMC::analysis {

// Angular distribution function (ADF) analysis for the Special Glass Structure
// workflow.
//
// If you use this code in your research, please cite:
//   S. Zhu, J. Schroers, S. Curtarolo, H. Eckert, and A. van de Walle,
//   "Special glass structures for first-principles studies of bulk metallic
//   glasses," Acta Materialia 262, 119456 (2024).
//   doi:10.1016/j.actamat.2023.119456

// Grid and options for an angular distribution function (ADF) computation.
struct AdfParams {
  double max_dis = 3.4; // bond cutoff for neighbour pairs (Å)
  int n_bins = 100;     // angle bins over [0, π]
  int smooth_range = 2; // boxcar half-width (0 disables); applied in 2 passes
};

// Per-element-triplet partial ADFs plus their sum, on a shared angle grid. The
// triplet ordering matches AngularDistributionConstraint's column order
// (central element outer, unordered leg pair inner), so an ADF written by
// write_adf is directly usable as that constraint's target.
struct AdfResult {
  vec_t theta;                        // bin centres (rad), length n_bins
  vec_t total;                       // sum over triplet columns
  std::vector<LabeledCurve> partials; // one per triplet (label "A-B-C" + adf)
  Composition species;               // distinct elements (sorted) with counts
  double max_dis = 0.0;              // cutoff used
};

// Compute the ADF from an in-memory configuration. Requires a periodic box (the
// volume normalization needs the cell volume); returns a failed Result for
// a non-periodic cell.
[[nodiscard]] Result<AdfResult>
compute_adf(const coords_t &coords, const BoundaryConditions &bc,
            std::span<const std::string> elements, const AdfParams &params);

// Convenience overload: read a structure file (.pdb, or .vasp /.poscar / "POSCAR" name,
// else LAMMPS data
// For LAMMPS and VASP the file's own cell is used; for PDB the supplied `bc`
// provides the box. `type_to_element` maps LAMMPS atom types to element symbols
// (ignored for PDB/VASP).
[[nodiscard]] Result<AdfResult>
compute_adf(const std::filesystem::path &path, const BoundaryConditions &bc,
            const AdfParams &params,
            const std::vector<std::string> &type_to_element = {});

// Write theta and the per-triplet partials as a commented whitespace table in
// the exact column order (and format) AngularDistributionConstraint consumes
// via read_xy_data — so a computed ADF can be fed straight back as a fit
// target. The total is *not* emitted as a column (it is the sum of the
// partials); plot_adf.py reconstructs it.
[[nodiscard]] Result<void> write_adf(const AdfResult &a,
                                     const std::filesystem::path &path);

} // namespace RMC::analysis
