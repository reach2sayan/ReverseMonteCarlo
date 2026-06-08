#include <RMC/analysis/AngularDistribution.hpp>
#include <RMC/analysis/detail/InputCheck.hpp>
#include <RMC/constraints/AngularDistributionConstraint.hpp>
#include <RMC/io/LammpsReader.hpp>
#include <RMC/io/PdbReader.hpp>
#include <RMC/io/VaspReader.hpp>

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
  out.species.resize(static_cast<std::size_t>(S));
  out.counts.assign(static_cast<std::size_t>(S), 0);
  for (const auto &[sym, id] : id_of) {
    out.species[id] = sym;
  }

  std::vector<std::uint8_t> elem_id(static_cast<std::size_t>(N));
  for (Eigen::Index i = 0; i < N; ++i) {
    const std::uint8_t id = id_of.at(elements[static_cast<std::size_t>(i)]);
    elem_id[static_cast<std::size_t>(i)] = id;
    ++out.counts[id];
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
            static_cast<double>(out.counts[static_cast<std::size_t>(a)]) *
            static_cast<double>(out.counts[static_cast<std::size_t>(p)]) *
            static_cast<double>(out.counts[static_cast<std::size_t>(q)]);
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
        out.partials.push_back(std::move(partial));
        out.triplet_labels.push_back(std::format(
            "{}-{}-{}", out.species[a], out.species[p], out.species[q]));
      }

  return out;
}

Result<AdfResult> compute_adf(const std::filesystem::path &path,
                              const BoundaryConditions &bc,
                              const AdfParams &params,
                              const std::vector<std::string> &type_to_element) {
  const std::string ext = path.extension().string();
  const std::string stem = path.filename().string();

  if (ext == ".pdb" || ext == ".PDB") {
    BOOST_LEAF_AUTO(s, io::read_pdb(path));
    return compute_adf(s.coordinates, bc, s.elements, params);
  }
  if (ext == ".vasp" || ext == ".poscar" || ext == ".VASP" || stem == "POSCAR" ||
      stem == "CONTCAR") {
    BOOST_LEAF_AUTO(data, io::read_vasp(path));
    const BoundaryConditions vasp_bc = data.periodic_bc();
    return compute_adf(data.structure.coordinates, vasp_bc,
                       data.structure.elements, params);
  }

  // Otherwise a LAMMPS data file; use its own cell.
  BOOST_LEAF_AUTO(data, io::read_lammps_data(path, type_to_element));
  const BoundaryConditions lammps_bc = data.periodic_bc();
  return compute_adf(data.structure.coordinates, lammps_bc,
                     data.structure.elements, params);
}

Result<void> write_adf(const AdfResult &a, const std::filesystem::path &path) {
  std::ofstream f(path);
  if (!f)
    return boost::leaf::new_error(
        std::string{"Cannot write ADF file: " + path.string()});

  f << "# adf: max_dis=" << a.max_dis << ", species=";
  for (const auto &[i, species_and_count] :
       std::views::zip(a.species, a.counts) | std::views::enumerate) {
    const auto &[species, count] = species_and_count;
    f << (i ? "," : "") << species << "(" << count << ")";
  }
  f << "\n# theta";
  for (const auto &lbl : a.triplet_labels) {
    f << "  adf_" << lbl;
  }
  f << "\n";

  for (Eigen::Index b = 0; b < a.theta.size(); ++b) {
    f << std::format("{:.6f}", a.theta(b));
    for (const auto &p : a.partials) {
      f << std::format(" {:.8f}", p(b));
    }
    f << "\n";
  }
  return {};
}

} // namespace RMC::analysis
