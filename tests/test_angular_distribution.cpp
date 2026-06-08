// Tests for the MAST-derived additions: the angular distribution function (ADF)
// analysis tool and constraint, the multi-column data reader, and VASP POSCAR
// round-trip I/O.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <RMC/analysis/AngularDistribution.hpp>
#include <RMC/constraints/AngularDistributionConstraint.hpp>
#include <RMC/core/BoundaryConditions.hpp>
#include <RMC/core/RandomStructure.hpp>
#include <RMC/core/Structure.hpp>
#include <RMC/io/DataReader.hpp>
#include <RMC/io/VaspReader.hpp>

#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

using Catch::Matchers::WithinAbs;
using namespace RMC;

namespace {

struct TmpFile {
  std::filesystem::path path;
  explicit TmpFile(const std::string &name)
      : path(std::filesystem::temp_directory_path() / name) {}
  ~TmpFile() {
    std::error_code ec;
    std::filesystem::remove(path, ec);
  }
};

} // namespace

// ============================================================
// read_columns
// ============================================================

TEST_CASE("read_columns - multi-column with comments", "[io][adf]") {
  TmpFile tf("rmc_cols.dat");
  std::ofstream(tf.path) << "# header\n1 2 3\n4, 5, 6\n7 8 9  # inline\n";
  const auto r = io::read_columns(tf.path);
  REQUIRE(r);
  const mat_t m = *r;
  REQUIRE(m.rows() == 3);
  REQUIRE(m.cols() == 3);
  REQUIRE_THAT(m(0, 0), WithinAbs(1.0, 1e-12));
  REQUIRE_THAT(m(1, 1), WithinAbs(5.0, 1e-12));
  REQUIRE_THAT(m(2, 2), WithinAbs(9.0, 1e-12));
}

TEST_CASE("read_columns - rejects ragged rows", "[io][adf]") {
  TmpFile tf("rmc_ragged.dat");
  std::ofstream(tf.path) << "1 2 3\n4 5\n";
  REQUIRE_FALSE(io::read_columns(tf.path));
}

// ============================================================
// VASP POSCAR round-trip
// ============================================================

TEST_CASE("VASP write/read round-trips a triclinic cell", "[io][vasp]") {
  AtomicStructure s;
  s.coordinates.resize(4, 3);
  s.coordinates.row(0) << 1.0, 2.0, 3.0;
  s.coordinates.row(1) << 4.0, 1.0, 2.0;
  s.coordinates.row(2) << 7.0, 8.0, 1.0;
  s.coordinates.row(3) << 2.0, 3.0, 4.0;
  s.atomic_numbers.resize(4);
  s.atomic_numbers << 29, 29, 40, 40; // grouped: Cu, Cu, Zr, Zr
  for (const char *e : {"Cu", "Cu", "Zr", "Zr"}) {
    s.elements.emplace_back(e);
    s.names.emplace_back(e);
    s.residues.emplace_back(e);
    s.molecule_ids.push_back(1);
  }

  mat3_t box = mat3_t::Zero();
  box.col(0) = vec3_t{10.0, 0.0, 0.0};
  box.col(1) = vec3_t{1.0, 10.0, 0.0};
  box.col(2) = vec3_t{0.5, 0.5, 10.0};

  TmpFile tf("rmc_roundtrip.vasp");
  REQUIRE(io::write_vasp(s, box, tf.path));
  const auto rd = io::read_vasp(tf.path);
  REQUIRE(rd);

  REQUIRE(rd->structure.size() == 4);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      REQUIRE_THAT(rd->box(i, j), WithinAbs(box(i, j), 1e-6));
  for (Eigen::Index a = 0; a < 4; ++a)
    for (int c = 0; c < 3; ++c)
      REQUIRE_THAT(rd->structure.coordinates(a, c),
                   WithinAbs(s.coordinates(a, c), 1e-5));
  REQUIRE(rd->structure.elements == s.elements);
}

// ============================================================
// compute_adf
// ============================================================

TEST_CASE("compute_adf bins a known 90-degree angle", "[analysis][adf]") {
  // Cu at the origin with two Zr neighbours along x and y → a single 90° angle
  // centred on Cu. A large box keeps PBC from creating extra neighbours.
  AtomicStructure s;
  s.coordinates.resize(3, 3);
  s.coordinates.row(0) << 0.0, 0.0, 0.0;   // Cu
  s.coordinates.row(1) << 2.5, 0.0, 0.0;   // Zr
  s.coordinates.row(2) << 0.0, 2.5, 0.0;   // Zr
  s.atomic_numbers.resize(3);
  s.atomic_numbers << 29, 40, 40;
  for (const char *e : {"Cu", "Zr", "Zr"}) {
    s.elements.emplace_back(e);
    s.names.emplace_back(e);
    s.residues.emplace_back(e);
    s.molecule_ids.push_back(1);
  }
  const BoundaryConditions bc = PeriodicBC(mat3_t::Identity() * 20.0);

  analysis::AdfParams ap;
  ap.max_dis = 3.0;
  ap.n_bins = 18;       // 10° per bin
  ap.smooth_range = 0;  // no smoothing → exact bin counts

  const auto r = analysis::compute_adf(s.coordinates, bc, s.elements, ap);
  REQUIRE(r);
  REQUIRE(r->theta.size() == 18);
  // Sorted species → Cu(0), Zr(1); 2 types → 6 triplet columns.
  REQUIRE(r->partials.size() == 6);
  REQUIRE(r->triplet_labels[2] == "Cu-Zr-Zr");
  // 90° lands in bin 9; that is the only non-empty column/bin.
  REQUIRE(r->partials[2](9) > 0.0);
  for (std::size_t c = 0; c < r->partials.size(); ++c)
    for (Eigen::Index b = 0; b < r->partials[c].size(); ++b)
      if (!(c == 2 && b == 9))
        REQUIRE_THAT(r->partials[c](b), WithinAbs(0.0, 1e-12));
}

// ============================================================
// AngularDistributionConstraint
// ============================================================

TEST_CASE("ADF constraint chi2 vanishes against its own ADF",
          "[constraints][adf]") {
  const std::vector<std::string> els{"Cu", "Zr"};
  const std::vector<std::size_t> cnts{32, 32};
  auto cell = make_random_amorphous(els, cnts, 2.8, /*seed=*/7);
  REQUIRE(cell);
  const BoundaryConditions bc = cell->periodic_bc();

  analysis::AdfParams ap;
  ap.max_dis = 3.6;
  ap.n_bins = 45;
  const auto target =
      analysis::compute_adf(cell->structure.coordinates, bc,
                            cell->structure.elements, ap);
  REQUIRE(target);

  const int n_bins = static_cast<int>(target->theta.size());
  const int n_part = static_cast<int>(target->partials.size());
  mat_t data(n_bins, 1 + n_part);
  data.col(0) = target->theta;
  for (int j = 0; j < n_part; ++j)
    data.col(1 + j) = target->partials[static_cast<std::size_t>(j)];

  AngularDistributionConstraint c;
  c.set_experimental_data(data);
  c.set_cutoff(ap.max_dis);
  c.set_smoothing(ap.smooth_range);
  c.set_elements(cell->structure.elements);
  c.set_boundary_conditions(bc);
  c.initialise();

  const double err = c.compute_error(cell->structure.coordinates, {});
  REQUIRE_THAT(err, WithinAbs(0.0, 1e-6));
}

static_assert(CConstraint<AngularDistributionConstraint>);
