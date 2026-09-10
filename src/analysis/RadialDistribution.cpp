#include <RMC/analysis/RadialDistribution.hpp>
#include <RMC/analysis/detail/InputCheck.hpp>
#include <RMC/constraints/PairHistogram.hpp>
#include <RMC/io/StructFormat.hpp>

#include <boost/leaf.hpp>

#include <format>
#include <string>
#include <vector>

namespace RMC::analysis {

Result<GrResult> compute_gr(const coords_t &coords,
                            const BoundaryConditions &bc,
                            std::span<const std::string> elements,
                            const GrParams &params) {
  const Eigen::Index N = coords.rows();
  if (!(params.r_max > params.r_min))
    return boost::leaf::new_error(std::string{"compute_gr: need r_max > r_min"});
  // Shared precondition check; returns the cell volume the densities need.
  BOOST_LEAF_AUTO(V, detail::check_periodic_inputs("compute_gr", coords,
                                                   elements, params.n_bins, bc));

  const SpeciesIndex sp(elements);
  const int S = static_cast<int>(sp.size());
  GrResult out;
  out.species = composition_of(sp);

  const double bin_width = (params.r_max - params.r_min) / params.n_bins;
  const vec_t shell = shell_volumes(params.r_min, bin_width, params.n_bins);
  // Bin centers.
  out.r =
      Eigen::VectorXd::LinSpaced(params.n_bins, 0, params.n_bins - 1).array() *
          bin_width +
      (params.r_min + 0.5 * bin_width);
  out.density = static_cast<double>(N) / V;

  // For each unordered species pair (a ≤ b), isolate with a weight matrix and
  // accumulate its histogram. accumulate_pair_histogram adds 2·w per i<j pair.
  vec_t total_hist = vec_t::Zero(params.n_bins);
  for (int a = 0; a < S; ++a) {
    for (int b = a; b < S; ++b) {
      PairWeightMatrix wm{S, true,
                          std::vector<double>(static_cast<std::size_t>(S * S))};
      wm.w[static_cast<std::size_t>(a * S + b)] = 1.0;
      wm.w[static_cast<std::size_t>(b * S + a)] = 1.0;

      vec_t hist = vec_t::Zero(params.n_bins);
      accumulate_pair_histogram(hist, coords, &bc, sp.id, wm, params.r_min,
                                params.r_max, params.n_bins, {},
                                params.exclude_intra);
      total_hist += hist;

      // Normalize: observed ordered pairs / expected (N_a·ρ_b·shell).
      //   hist = 2·(unordered {a,b} count). For a≠b ordered a→b = hist/2;
      //   for a==b ordered = hist. This collapses to a single factor:
      //   g_ab = hist·V / (shell · N_a · N_b · (a==b ? 1 : 2)).
      const double Na = static_cast<double>(sp.count[static_cast<std::size_t>(a)]);
      const double Nb = static_cast<double>(sp.count[static_cast<std::size_t>(b)]);
      const double pair_factor = (a == b) ? 1.0 : 2.0;
      vec_t g = vec_t::Zero(params.n_bins);
      if (Na > 0.0 && Nb > 0.0) {
        g.array() = hist.array() * (V / (pair_factor * Na * Nb)) / shell.array();
      }
      out.partials.push_back(
          {.label =
               std::format("{}-{}", sp.symbols[static_cast<std::size_t>(a)],
                           sp.symbols[static_cast<std::size_t>(b)]),
           .values = std::move(g)});
    }
  }

  // Total g(r): identical convention to PairDistributionConstraint —
  // total_hist (all ordered pairs) / (shell · ρ · N).
  out.total = total_hist.array() /
              (shell.array() * out.density * static_cast<double>(N));
  return out;
}

Result<GrResult> compute_gr(const std::filesystem::path &path,
                            const BoundaryConditions &bc,
                            const GrParams &params,
                            const std::vector<std::string> &type_to_element) {
  // Format from path: PDB uses bc; VASP/LAMMPS use the file's cell (default LAMMPS).
  BOOST_LEAF_AUTO(loaded, io::read_structure_by_ext(path, bc, type_to_element));
  return compute_gr(loaded.structure.coordinates, loaded.bc,
                    loaded.structure.elements, params);
}

Result<void> write_gr(const GrResult &g, const std::filesystem::path &path) {
  std::vector<LabeledCurve> curves{{"total", g.total}};
  curves.insert(curves.end(), g.partials.begin(), g.partials.end());
  return write_curves(path, std::format("g(r): density(rho)={:g}", g.density),
                      g.species, "r", g.r, curves, "g_");
}

} // namespace RMC::analysis
