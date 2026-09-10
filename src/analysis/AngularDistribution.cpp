#include <RMC/analysis/AngularDistribution.hpp>
#include <RMC/analysis/detail/InputCheck.hpp>
#include <RMC/constraints/AngularDistributionConstraint.hpp>
#include <RMC/io/StructFormat.hpp>

#include <boost/leaf.hpp>

#include <algorithm>
#include <format>
#include <numbers>
#include <string>
#include <vector>

namespace RMC::analysis {

Result<AdfResult> compute_adf(const coords_t &coords,
                              const BoundaryConditions &bc,
                              std::span<const std::string> elements,
                              const AdfParams &params) {
  const Eigen::Index N = coords.rows();
  // Shared precondition check; returns cell volume for normalisation.
  BOOST_LEAF_AUTO(V, detail::check_periodic_inputs(
                         "compute_adf", coords, elements, params.n_bins, bc));

  // Sorted species ids — identical convention to AngularDistributionConstraint.
  const SpeciesIndex sp(elements);
  const int S = static_cast<int>(sp.size());
  const int n_leg_pairs = adf_n_leg_pairs(S);
  const int n_cols = S * n_leg_pairs;

  AdfResult out;
  out.max_dis = params.max_dis;
  out.species = composition_of(sp);

  // Raw angle histogram (flat angle_bin × triplet_column).
  vec_t hist = vec_t::Zero(static_cast<Eigen::Index>(params.n_bins) * n_cols);
  accumulate_angle_histogram(hist, coords, &bc, sp.id, S, params.max_dis,
                             params.n_bins);

  // Per-column volume normalization: inc / (N_a·N_p·N_q),
  // inc = V · N / π · n_bins. Absent species → 0.
  const double inc = V * static_cast<double>(N) / std::numbers::pi *
                     static_cast<double>(params.n_bins);
  const auto count = [&](int s) {
    return static_cast<double>(sp.count[static_cast<std::size_t>(s)]);
  };
  vec_t col_scale = vec_t::Zero(n_cols);
  for (const auto &[a, p, q, col] : adf_columns(S)) {
    const double denom = count(a) * count(p) * count(q);
    col_scale(col) = denom > 0.0 ? inc / denom : 0.0;
  }

  // Per-column scaling: hist is row-major (bin × column) flattened; tiled
  // col_scale aligns element-wise (entry b·n_cols+c ↦ col_scale(c)).
  hist.array() *= col_scale.replicate(params.n_bins, 1).array();

  // Row-major (n_bins × n_cols) view over the flat histogram; reused below.
  using RowMajMat =
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  // Per-column boxcar smoothing (shrinking window), 2 passes.
  vec_t scratch;
  adf_smooth(hist, scratch, params.n_bins, n_cols, params.smooth_range);

  // Bin centres.
  const double bw = std::numbers::pi / static_cast<double>(params.n_bins);
  out.theta =
      (Eigen::VectorXd::LinSpaced(params.n_bins, 0, params.n_bins - 1).array() +
       0.5) *
      bw;

  // Split into partials (canonical column order) and accumulate the total.
  out.total = vec_t::Zero(params.n_bins);
  Eigen::Map<const RowMajMat> H(hist.data(), params.n_bins, n_cols);
  const auto sym = [&](int s) {
    return sp.symbols[static_cast<std::size_t>(s)];
  };
  for (const auto &[a, p, q, col] : adf_columns(S)) {
    vec_t partial = H.col(col);
    out.total += partial;
    out.partials.push_back(
        {.label = std::format("{}-{}-{}", sym(a), sym(p), sym(q)),
         .values = std::move(partial)});
  }

  return out;
}

Result<AdfResult> compute_adf(const std::filesystem::path &path,
                              const BoundaryConditions &bc,
                              const AdfParams &params,
                              const std::vector<std::string> &type_to_element) {
  // Format from path: PDB uses bc; VASP/LAMMPS use the file's cell (default
  // LAMMPS).
  BOOST_LEAF_AUTO(loaded, io::read_structure_by_ext(path, bc, type_to_element));
  return compute_adf(loaded.structure.coordinates, loaded.bc,
                     loaded.structure.elements, params);
}

Result<void> write_adf(const AdfResult &a, const std::filesystem::path &path) {
  return write_curves(path, std::format("adf: max_dis={:g}", a.max_dis),
                      a.species, "theta", a.theta, a.partials, "adf_");
}

} // namespace RMC::analysis
