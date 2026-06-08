#include <RMC/analysis/RadialDistribution.hpp>
#include <RMC/constraints/PairHistogram.hpp>
#include <RMC/io/LammpsReader.hpp>
#include <RMC/io/PdbReader.hpp>

#include <boost/leaf.hpp>
#include <boost/leaf/result.hpp>

#include <cstdint>
#include <format>
#include <fstream>
#include <map>
#include <numbers>
#include <string>
#include <variant>
#include <vector>

namespace RMC::analysis {

namespace {

// Per-bin ideal-gas shell volumes: (4π/3)(r_hi³ − r_lo³). Same formula the
// pair constraints use (PairConstraintBase::initialise), so total g(r) here
// matches the fitting path bin-for-bin.
vec_t shell_volumes(double r_min, double bin_width, int n_bins) {
  const auto idx = Eigen::ArrayXd::LinSpaced(n_bins, 0, n_bins - 1);
  const auto r_lo = r_min + idx * bin_width;
  const auto r_hi = r_lo + bin_width;
  return (4.0 * std::numbers::pi / 3.0) * (r_hi.cube() - r_lo.cube());
}

} // namespace

Result<GrResult> compute_gr(const coords_t &coords, const BoundaryConditions &bc,
                            std::span<const std::string> elements,
                            const GrParams &params) {
  const Eigen::Index N = coords.rows();
  if (N == 0)
    return boost::leaf::new_error(std::string{"compute_gr: empty structure"});
  if (params.n_bins <= 0 || !(params.r_max > params.r_min))
    return boost::leaf::new_error(
        std::string{"compute_gr: need n_bins > 0 and r_max > r_min"});
  if (elements.size() != static_cast<std::size_t>(N))
    return boost::leaf::new_error(
        std::string{"compute_gr: elements size does not match atom count"});

  // Partials need per-species densities ρ_b = N_b/V, so a finite cell volume is
  // required; a non-periodic cell has no meaningful V.
  if (!std::holds_alternative<PeriodicBC>(bc))
    return boost::leaf::new_error(
        std::string{"compute_gr: a periodic box is required for g(r)"});
  const double V = bc_volume(bc);
  if (!(V > 0.0))
    return boost::leaf::new_error(
        std::string{"compute_gr: box volume must be positive"});

  // Map distinct element labels → contiguous, sorted species ids.
  std::map<std::string, std::uint8_t> id_of;
  for (const auto &e : elements)
    id_of.emplace(e, 0); // insert keys; values assigned below in sorted order
  std::uint8_t next = 0;
  for (auto &kv : id_of)
    kv.second = next++;
  const int S = static_cast<int>(id_of.size());

  GrResult out;
  out.species.resize(static_cast<std::size_t>(S));
  out.counts.assign(static_cast<std::size_t>(S), 0);
  for (const auto &[sym, id] : id_of)
    out.species[id] = sym;

  std::vector<std::uint8_t> elem_id(static_cast<std::size_t>(N));
  for (Eigen::Index i = 0; i < N; ++i) {
    const std::uint8_t id = id_of.at(elements[static_cast<std::size_t>(i)]);
    elem_id[static_cast<std::size_t>(i)] = id;
    ++out.counts[id];
  }

  const double bin_width = (params.r_max - params.r_min) / params.n_bins;
  const vec_t shell = shell_volumes(params.r_min, bin_width, params.n_bins);

  // Bin centers.
  out.r.resize(params.n_bins);
  for (int k = 0; k < params.n_bins; ++k)
    out.r(k) = params.r_min + (k + 0.5) * bin_width;

  out.density = static_cast<double>(N) / V;

  // For each unordered species pair (a ≤ b), isolate it with a weight matrix
  // and accumulate its histogram. Reuses the same tested, PBC-correct loop the
  // constraints use. accumulate_pair_histogram adds 2·w per i<j pair.
  vec_t total_hist = vec_t::Zero(params.n_bins);
  for (int a = 0; a < S; ++a) {
    for (int b = a; b < S; ++b) {
      PairWeightMatrix wm;
      wm.n_types = S;
      wm.weighted = true;
      wm.w.assign(static_cast<std::size_t>(S) * static_cast<std::size_t>(S), 0.0);
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
      const double Na = static_cast<double>(out.counts[a]);
      const double Nb = static_cast<double>(out.counts[b]);
      const double pair_factor = (a == b) ? 1.0 : 2.0;
      vec_t g = vec_t::Zero(params.n_bins);
      if (Na > 0.0 && Nb > 0.0) {
        const double denom_const = pair_factor * Na * Nb;
        g.array() = hist.array() * V / (shell.array() * denom_const);
      }
      out.partials.push_back(std::move(g));
      out.pair_labels.push_back(
          std::format("{}-{}", out.species[a], out.species[b]));
    }
  }

  // Total g(r): identical convention to PairDistributionConstraint —
  // total_hist (all ordered pairs) / (shell · ρ · N).
  out.total = vec_t::Zero(params.n_bins);
  out.total.array() =
      total_hist.array() / (shell.array() * out.density * static_cast<double>(N));

  return out;
}

Result<GrResult> compute_gr(const std::filesystem::path &path,
                            const BoundaryConditions &bc, const GrParams &params,
                            const std::vector<std::string> &type_to_element) {
  const std::string ext = path.extension().string();

  if (ext == ".pdb" || ext == ".PDB") {
    BOOST_LEAF_AUTO(s, io::read_pdb(path));
    return compute_gr(s.coordinates, bc, s.elements, params);
  }

  // Treat anything else as a LAMMPS data file; use its own cell.
  BOOST_LEAF_AUTO(data, io::read_lammps_data(path, type_to_element));
  const BoundaryConditions lammps_bc = data.periodic_bc();
  return compute_gr(data.structure.coordinates, lammps_bc,
                    data.structure.elements, params);
}

Result<void> write_gr(const GrResult &g, const std::filesystem::path &path) {
  std::ofstream f(path);
  if (!f)
    return boost::leaf::new_error(
        std::string{"Cannot write g(r) file: " + path.string()});

  // Column header.
  f << "# g(r): density(rho)=" << g.density << ", species=";
  for (std::size_t i = 0; i < g.species.size(); ++i)
    f << (i ? "," : "") << g.species[i] << "(" << g.counts[i] << ")";
  f << "\n# r  g_total";
  for (const auto &lbl : g.pair_labels)
    f << "  g_" << lbl;
  f << "\n";

  for (Eigen::Index k = 0; k < g.r.size(); ++k) {
    f << std::format("{:.6f} {:.8f}", g.r(k), g.total(k));
    for (const auto &p : g.partials)
      f << std::format(" {:.8f}", p(k));
    f << "\n";
  }
  return {};
}

} // namespace RMC::analysis
