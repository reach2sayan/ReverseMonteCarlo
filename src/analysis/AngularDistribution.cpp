#include <RMC/analysis/AngularDistribution.hpp>
#include <RMC/analysis/detail/InputCheck.hpp>
#include <RMC/constraints/AngularDistributionConstraint.hpp>
#include <RMC/io/StructFormat.hpp>

#include <boost/leaf.hpp>
#include <boost/leaf/result.hpp>

#include <algorithm>
#include <cstdint>
#include <format>
#include <fstream>
#include <map>
#include <numbers>
#include <set>
#include <string>
#include <vector>

namespace RMC::analysis {

Result<AdfResult> compute_adf(const coords_t &coords,
                              const BoundaryConditions &bc,
                              std::span<const std::string> elements,
                              const AdfParams &params) {
  const Eigen::Index N = coords.rows();
  // Shared precondition check (non-empty, matching labels, positive bins,
  // periodic box); returns the cell volume MAST's normalisation needs.
  BOOST_LEAF_AUTO(V, detail::check_periodic_inputs("compute_adf", coords,
                                                   elements, params.n_bins, bc));

  // Sorted species ids — identical convention to AngularDistributionConstraint.
  std::set<std::string> unique(elements.begin(), elements.end());
  std::map<std::string, std::uint8_t> id_of;
  std::uint8_t next = 0;
  for (const auto& e : unique) {
    id_of.emplace(e, next++);
  }
  const int S = static_cast<int>(id_of.size());
  const int n_leg_pairs = adf_n_leg_pairs(S);
  const int n_cols = S * n_leg_pairs;

  AdfResult out;
  out.max_dis = params.max_dis;
  // Per-id symbol and atom count, accumulated by integer species id; folded
  // into the result's flat_set once complete (set elements are immutable, so
  // the counts can't be incremented in place there).
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

  // Raw angle histogram (flat angle_bin × triplet_column).
  vec_t hist = vec_t::Zero(static_cast<Eigen::Index>(params.n_bins) * n_cols);
  accumulate_angle_histogram(hist, coords, &bc, elem_id, S, params.max_dis,
                             params.n_bins);

  // Per-column MAST normalization: inc / (N_a·N_p·N_q),
  // inc = V · N / π · n_bins. Absent species → 0.
  const double inc = V * static_cast<double>(N) / std::numbers::pi *
                     static_cast<double>(params.n_bins);
  vec_t col_scale = vec_t::Zero(n_cols);
  for (int a = 0; a < S; ++a)
    for (int p = 0; p < S; ++p)
      for (int q = p; q < S; ++q) {
        const int col = a * n_leg_pairs + adf_leg_pair_index(p, q, S);
        const double denom =
            static_cast<double>(counts[static_cast<std::size_t>(a)]) *
            static_cast<double>(counts[static_cast<std::size_t>(p)]) *
            static_cast<double>(counts[static_cast<std::size_t>(q)]);
        col_scale(col) = denom > 0.0 ? inc / denom : 0.0;
      }

  // Per-column scaling: `hist` is a row-major (bin × column) matrix flattened
  // into a vector, so a column-tiled copy of col_scale aligns element-for-
  // element with it (entry b·n_cols+c ↦ col_scale(c)).
  hist.array() *= col_scale.replicate(params.n_bins, 1).array();

  // Row-major (n_bins × n_cols) view over the flat histogram; reused below.
  using RowMajMat =
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  // Per-column boxcar smoothing (shrinking window), 2 passes — matches the
  // constraint and MAST's smooth(hist, 2, 2).
  if (params.smooth_range > 0 && params.n_bins > 1) {
    const int range = params.smooth_range;
    vec_t scratch = hist;
    for (int pass = 0; pass < 2; ++pass) {
      Eigen::Map<const RowMajMat> H(hist.data(), params.n_bins, n_cols);
      Eigen::Map<RowMajMat> Hs(scratch.data(), params.n_bins, n_cols);
      for (int c = 0; c < n_cols; ++c)
        for (int b = 0; b < params.n_bins; ++b) {
          const int lo = std::max(0, b - range);
          const int hi = std::min(params.n_bins - 1, b + range);
          Hs(b, c) = H.col(c).segment(lo, hi - lo + 1).sum() /
                     static_cast<double>(hi - lo + 1);
        }
      hist.swap(scratch);
    }
  }

  // Bin centres.
  const double bw = std::numbers::pi / static_cast<double>(params.n_bins);
  out.theta.resize(params.n_bins);
  out.theta =
      (Eigen::VectorXd::LinSpaced(params.n_bins, 0, params.n_bins - 1).array() +
       0.5) *
      bw;

  // Split into partials (canonical column order) and accumulate the total.
  out.total = vec_t::Zero(params.n_bins);
  Eigen::Map<const RowMajMat> H(hist.data(), params.n_bins, n_cols);
  for (int a = 0; a < S; ++a)
    for (int p = 0; p < S; ++p)
      for (int q = p; q < S; ++q) {
        const int col = a * n_leg_pairs + adf_leg_pair_index(p, q, S);
        vec_t partial = H.col(col);
        out.total += partial;
        out.partials.push_back(
            {std::format("{}-{}-{}", sym_of_id[a], sym_of_id[p], sym_of_id[q]),
             std::move(partial)});
      }

  return out;
}

Result<AdfResult> compute_adf(const std::filesystem::path &path,
                              const BoundaryConditions &bc,
                              const AdfParams &params,
                              const std::vector<std::string> &type_to_element) {
  // Format chosen from the path (PDB uses the supplied bc; VASP/LAMMPS use the
  // cell declared in the file; unknown extensions fall back to LAMMPS).
  BOOST_LEAF_AUTO(loaded, io::read_structure_by_ext(path, bc, type_to_element));
  return compute_adf(loaded.structure.coordinates, loaded.bc,
                     loaded.structure.elements, params);
}

Result<void> write_adf(const AdfResult &a, const std::filesystem::path &path) {
  std::ofstream f(path);
  if (!f)
    return boost::leaf::new_error(
        std::string{"Cannot write ADF file: " + path.string()});

  f << "# adf: max_dis=" << a.max_dis << ", species=";
  for (const auto &[i, s] : a.species | std::views::enumerate)
    f << (i ? "," : "") << s.symbol << "(" << s.count << ")";
  f << "\n# theta";
  for (const auto &p : a.partials)
    f << "  adf_" << p.label;
  f << "\n";

  for (Eigen::Index b = 0; b < a.theta.size(); ++b) {
    f << std::format("{:.6f}", a.theta(b));
    for (const auto &p : a.partials)
      f << std::format(" {:.8f}", p.values(b));
    f << "\n";
  }
  return {};
}

} // namespace RMC::analysis
