#pragma once
#include <RMC/analysis/Composition.hpp>
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
  vec_t r;                            // bin centers, length n_bins
  vec_t total;                       // total g(r)
  std::vector<LabeledCurve> partials; // one per element pair (label + g_ab)
  Composition species;               // distinct elements (sorted) with counts
  double density = 0.0;              // N / V used for normalization
};

// Compute g(r) from an in-memory configuration. Requires a periodic box
// (partial densities ρ_b = N_b/V need the cell volume). Total uses ρ = N/V,
// matching PairDistributionConstraint's convention.
[[nodiscard]] Result<GrResult> compute_gr(const coords_t &coords,
                                          const BoundaryConditions &bc,
                                          std::span<const std::string> elements,
                                          const GrParams &params);

// CLI Convenience
[[nodiscard]] Result<GrResult>
compute_gr(const std::filesystem::path &path, const BoundaryConditions &bc,
           const GrParams &params,
           const std::vector<std::string> &type_to_element = {});

// Write r, total g(r) and each partial as a commented whitespace-separated
// table (the same format read_xy_data consumes).
[[nodiscard]] Result<void> write_gr(const GrResult &g,
                                    const std::filesystem::path &path);

} // namespace RMC::analysis
