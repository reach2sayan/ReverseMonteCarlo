#include <RMC/analysis/RadialDistribution.hpp>
#include <RMC/analysis/detail/InputCheck.hpp>
#include <RMC/constraints/PairHistogram.hpp>
#include <RMC/io/StructFormat.hpp>

#include <boost/leaf.hpp>
#include <boost/leaf/result.hpp>

#include <cstdint>
#include <format>
#include <fstream>
#include <map>
#include <numbers>
#include <ranges>
#include <string>
#include <vector>

namespace RMC::analysis {

namespace {

// Per-bin ideal-gas shell volumes: (4π/3)(r_hi³ − r_lo³). Matches
// PairConstraintBase::initialise.
vec_t shell_volumes(double r_min, double bin_width, int n_bins) {
  const auto idx = Eigen::ArrayXd::LinSpaced(n_bins, 0, n_bins - 1);
  const auto r_lo = r_min + idx * bin_width;
  const auto r_hi = r_lo + bin_width;
  return (4.0 * std::numbers::pi / 3.0) * (r_hi.cube() - r_lo.cube());
}

} // namespace

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

  // Map distinct element labels → contiguous, sorted species ids.
  std::map<std::string, std::uint8_t> id_of;
  for (const auto &e : elements) {
    id_of.try_emplace(e);
  }
  std::uint8_t next = 0;
  for (auto &[_, id] : id_of) {
    id = next++;
  }

  const int S = static_cast<int>(id_of.size());
  GrResult out;
  // Per-id symbol/count, folded into the result's (immutable) flat_set once done.
  std::vector<std::string> sym_of_id(static_cast<std::size_t>(S));
  for (const auto &[sym, id] : id_of) {
    sym_of_id[id] = sym;
  }
  std::vector<std::size_t> counts(static_cast<std::size_t>(S), 0);

  std::vector<std::uint8_t> elem_id(static_cast<std::size_t>(N));
  for (auto &&[id, e] : std::views::zip(elem_id, elements)) {
    id = id_of.at(e);
    ++counts[id];
  }
  for (auto &&[sym, count] : std::views::zip(sym_of_id, counts)) {
    out.species.insert({sym, count});
  }

  const double bin_width = (params.r_max - params.r_min) / params.n_bins;
  const vec_t shell = shell_volumes(params.r_min, bin_width, params.n_bins);

  // Bin centers.
  out.r.resize(params.n_bins);
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
      PairWeightMatrix wm;
      wm.n_types = S;
      wm.weighted = true;
      wm.w.assign(static_cast<std::size_t>(S) * static_cast<std::size_t>(S),
                  0.0);
      wm.w[static_cast<std::size_t>(a) * S + b] = 1.0;
      wm.w[static_cast<std::size_t>(b) * S + a] = 1.0;

      vec_t hist = vec_t::Zero(params.n_bins);
      accumulate_pair_histogram(hist, coords, &bc, elem_id, wm, params.r_min,
                                params.r_max, params.n_bins, {},
                                params.exclude_intra);
      total_hist += hist;

      // Normalize: observed ordered pairs / expected (N_a·ρ_b·shell).
      //   hist = 2·(unordered {a,b} count). For a≠b ordered a→b = hist/2;
      //   for a==b ordered = hist. This collapses to a single factor:
      //   g_ab = hist·V / (shell · N_a · N_b · (a==b ? 1 : 2)).
      const double Na = static_cast<double>(counts[a]);
      const double Nb = static_cast<double>(counts[b]);
      const double pair_factor = (a == b) ? 1.0 : 2.0;
      vec_t g = vec_t::Zero(params.n_bins);
      if (Na > 0.0 && Nb > 0.0) {
        const double denom_const = pair_factor * Na * Nb;
        g.array() = hist.array() * (V / denom_const) / shell.array();
      }
      out.partials.push_back(
          {std::format("{}-{}", sym_of_id[a], sym_of_id[b]), std::move(g)});
    }
  }

  // Total g(r): identical convention to PairDistributionConstraint —
  // total_hist (all ordered pairs) / (shell · ρ · N).
  out.total = vec_t::Zero(params.n_bins);
  out.total.array() = total_hist.array() /
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
  std::ofstream f(path);
  if (!f)
    return boost::leaf::new_error(
        std::string{"Cannot write g(r) file: " + path.string()});

  // Column header.
  f << "# g(r): density(rho)=" << g.density << ", species=";
  for (const auto &[i, s] : g.species | std::views::enumerate)
    f << (i ? "," : "") << s.symbol << "(" << s.count << ")";
  f << "\n# r  g_total";
  for (const auto &p : g.partials)
    f << "  g_" << p.label;
  f << "\n";

  for (Eigen::Index k = 0; k < g.r.size(); ++k) {
    f << std::format("{:.6f} {:.8f}", g.r(k), g.total(k));
    for (const auto &p : g.partials)
      f << std::format(" {:.8f}", p.values(k));
    f << "\n";
  }
  return {};
}

} // namespace RMC::analysis
