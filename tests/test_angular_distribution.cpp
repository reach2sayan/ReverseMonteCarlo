// Tests for the MAST-derived additions: the angular distribution function (ADF)
// analysis tool and constraint, the multi-column data reader, and VASP POSCAR
// round-trip I/O.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <RMC/analysis/AngularDistribution.hpp>
#include <RMC/constraints/AngularDistributionConstraint.hpp>
#include <RMC/core/AtomsCollector.hpp>
#include <RMC/core/BoundaryConditions.hpp>
#include <RMC/core/NeighborGrid.hpp>
#include <RMC/core/RandomStructure.hpp>
#include <RMC/core/Structure.hpp>
#include <RMC/io/DataReader.hpp>
#include <RMC/io/VaspReader.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
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

// Brute-force min-image neighbours of atom i within `cutoff` (sorted), the
// reference NeighborGrid must reproduce. Mirrors accumulate_angle_histogram's
// adjacency test exactly (<= cutoff², same min-image).
std::vector<std::uint32_t> brute_neighbors(const coords_t &coords,
                                           const BoundaryConditions *bc,
                                           double cutoff, std::size_t i) {
  std::vector<std::uint32_t> out;
  const double c2 = cutoff * cutoff;
  for (Eigen::Index j = 0; j < coords.rows(); ++j) {
    if (static_cast<std::size_t>(j) == i) {
      continue;
    }
    vec3_t d =
        (coords.row(j) - coords.row(static_cast<Eigen::Index>(i))).transpose();
    if (bc) {
      d = bc_min_image(*bc, d);
    }
    if (d.squaredNorm() <= c2) {
      out.push_back(static_cast<std::uint32_t>(j));
    }
  }
  std::ranges::sort(out);
  return out;
}

// Assert every atom's grid neighbour set equals the brute-force reference.
void require_grid_matches_brute(const NeighborGrid &grid,
                                const coords_t &coords,
                                const BoundaryConditions &bc, double cutoff,
                                const AtomsCollector *col = nullptr) {
  std::vector<std::uint32_t> got;
  for (Eigen::Index i = 0; i < coords.rows(); ++i) {
    const std::size_t ii = static_cast<std::size_t>(i);
    if (col && col->absent(ii)) {
      continue;
    }
    grid.neighbors_of(ii, coords, &bc, cutoff, col, got);
    std::ranges::sort(got);
    auto ref = brute_neighbors(coords, &bc, cutoff, ii);
    if (col) {
      std::erase_if(ref, [&](std::uint32_t j) { return col->absent(j); });
    }
    REQUIRE(got == ref);
  }
}

void require_close(const vec_t &a, const vec_t &b, double tol = 1e-9) {
  REQUIRE(a.size() == b.size());
  for (Eigen::Index i = 0; i < a.size(); ++i) {
    REQUIRE_THAT(a(i), WithinAbs(b(i), tol));
  }
}

// Build an initialised single-frame ADF constraint. The target is the
// structure's own ADF (only its dimensions matter for computed() comparisons).
// NOTE: set_elements stores a non-owning span, so `s` must outlive the result.
AngularDistributionConstraint make_adf(const AtomicStructure &s,
                                       const BoundaryConditions &bc,
                                       const analysis::AdfParams &ap,
                                       const AtomsCollector *col = nullptr) {
  const auto t = analysis::compute_adf(s.coordinates, bc, s.elements, ap);
  REQUIRE(t);
  const int nb = static_cast<int>(t->theta.size());
  const int np = static_cast<int>(t->partials.size());
  mat_t data(nb, 1 + np);
  data.col(0) = t->theta;
  for (int j = 0; j < np; ++j) {
    data.col(1 + j) = t->partials[static_cast<std::size_t>(j)];
  }
  AngularDistributionConstraint c;
  c.set_experimental_data(data);
  c.set_cutoff(ap.max_dis);
  c.set_smoothing(ap.smooth_range);
  c.set_elements(s.elements);
  c.set_boundary_conditions(bc);
  if (col) {
    c.set_collector(col);
  }
  c.initialise();
  return c;
}

// computed() after a full (moved == {}) evaluation of a fresh constraint at
// `coords` — the reference any incremental result must reproduce.
vec_t full_computed(const AtomicStructure &s, const BoundaryConditions &bc,
                    const analysis::AdfParams &ap, const coords_t &coords,
                    const AtomsCollector *col = nullptr) {
  auto ref = make_adf(s, bc, ap, col);
  (void)ref.compute_error(coords, {});
  return ref.computed();
}

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

// ============================================================
// NeighborGrid
// ============================================================

TEST_CASE("NeighborGrid matches brute force - orthorhombic", "[neighbor_grid]") {
  const std::vector<std::string> els{"Cu", "Zr"};
  const std::vector<std::size_t> cnts{32, 32};
  auto cell = make_random_amorphous(els, cnts, 2.8, /*seed=*/3);
  REQUIRE(cell);
  const BoundaryConditions bc = cell->periodic_bc();
  const double cutoff = 3.6; // box side 11.2 → 3 cells/axis (exercises ±1 wrap)

  NeighborGrid grid;
  grid.build(cell->structure.coordinates, &bc, cutoff, nullptr);
  require_grid_matches_brute(grid, cell->structure.coordinates, bc, cutoff);
}

TEST_CASE("NeighborGrid matches brute force - triclinic and tiny box",
          "[neighbor_grid]") {
  // A skewed cell (lattice vectors as columns, the codebase convention) plus a
  // box too small to subdivide (1 cell/axis → the scan-all stencil path).
  auto frac_grid = [](const mat3_t &box, int per_side) {
    const int N = per_side * per_side * per_side;
    coords_t c(N, 3);
    int idx = 0;
    for (int a = 0; a < per_side; ++a)
      for (int b = 0; b < per_side; ++b)
        for (int d = 0; d < per_side; ++d) {
          const vec3_t f{(a + 0.5) / per_side, (b + 0.5) / per_side,
                         (d + 0.5) / per_side};
          c.row(idx++) = (box * f).transpose();
        }
    return c;
  };

  SECTION("triclinic, 3 cells/axis") {
    mat3_t box = mat3_t::Zero();
    box.col(0) = vec3_t{12.0, 0.0, 0.0};
    box.col(1) = vec3_t{3.0, 12.0, 0.0};
    box.col(2) = vec3_t{1.0, 2.0, 12.0};
    const BoundaryConditions bc = PeriodicBC(box);
    const coords_t coords = frac_grid(box, 4);
    const double cutoff = 3.5;
    NeighborGrid grid;
    grid.build(coords, &bc, cutoff, nullptr);
    require_grid_matches_brute(grid, coords, bc, cutoff);
  }

  SECTION("tiny box, 1 cell/axis") {
    const mat3_t box = mat3_t::Identity() * 5.0;
    const BoundaryConditions bc = PeriodicBC(box);
    const coords_t coords = frac_grid(box, 3);
    const double cutoff = 3.5; // 5/3.5 < 2 → single cell, scan-all path
    NeighborGrid grid;
    grid.build(coords, &bc, cutoff, nullptr);
    require_grid_matches_brute(grid, coords, bc, cutoff);
  }
}

TEST_CASE("NeighborGrid relocate / remove / restore stay consistent",
          "[neighbor_grid]") {
  const std::vector<std::string> els{"Cu", "Zr"};
  const std::vector<std::size_t> cnts{32, 32};
  auto cell = make_random_amorphous(els, cnts, 2.8, /*seed=*/4);
  REQUIRE(cell);
  const BoundaryConditions bc = cell->periodic_bc();
  const double cutoff = 3.6;
  const coords_t c0 = cell->structure.coordinates;

  NeighborGrid grid;
  grid.build(c0, &bc, cutoff, nullptr);

  const std::size_t m = 7;
  coords_t c1 = c0;
  c1.row(m) += vec3_t{1.3, -0.9, 0.7}.transpose();

  // Snapshot, then relocate the moved atom: queries must match brute force at
  // the new coords.
  grid.save_cells(std::array<std::size_t, 1>{m});
  grid.relocate(m, c1);
  require_grid_matches_brute(grid, c1, bc, cutoff);

  // Restore: back to the original neighbours.
  grid.restore_cells();
  require_grid_matches_brute(grid, c0, bc, cutoff);

  // Remove: m vanishes from every neighbour list; restore brings it back.
  grid.save_cells(std::array<std::size_t, 1>{m});
  grid.remove(m);
  AtomsCollector col;
  col.stage_removal(std::array<std::size_t, 1>{m}); // mark m absent for the ref
  require_grid_matches_brute(grid, c0, bc, cutoff, &col);
  grid.restore_cells();
  require_grid_matches_brute(grid, c0, bc, cutoff);
}

// ============================================================
// AngularDistributionConstraint - incremental cache
// ============================================================

TEST_CASE("ADF incremental delta matches full recompute (single-frame)",
          "[constraints][adf][incremental]") {
  const std::vector<std::string> els{"Cu", "Zr"};
  const std::vector<std::size_t> cnts{32, 32};
  auto cell = make_random_amorphous(els, cnts, 2.8, /*seed=*/11);
  REQUIRE(cell);
  const AtomicStructure &s = cell->structure;
  const BoundaryConditions bc = cell->periodic_bc();
  analysis::AdfParams ap;
  ap.max_dis = 3.6;
  ap.n_bins = 45;

  auto c = make_adf(s, bc, ap);
  (void)c.compute_error(s.coordinates, {}); // seed the cached histogram + grid

  SECTION("single-atom translate") {
    const std::array<std::size_t, 1> mv{5};
    coords_t c1 = s.coordinates;
    c1.row(5) += vec3_t{0.4, -0.3, 0.25}.transpose();
    (void)c.compute_error(s.coordinates, mv); // before-move (old coords)
    (void)c.compute_error(c1, mv);            // after-move (new coords)
    require_close(c.computed(), full_computed(s, bc, ap, c1));
  }

  SECTION("large move that forms/breaks angles") {
    const std::array<std::size_t, 1> mv{20};
    coords_t c1 = s.coordinates;
    c1.row(20) += vec3_t{1.7, 1.4, -1.5}.transpose(); // crosses the cutoff
    (void)c.compute_error(s.coordinates, mv);
    (void)c.compute_error(c1, mv);
    require_close(c.computed(), full_computed(s, bc, ap, c1));
  }

  SECTION("two-atom swap") {
    const std::size_t a = 3, b = 40; // different species rows
    const std::array<std::size_t, 2> mv{a, b};
    coords_t c1 = s.coordinates;
    c1.row(a) = s.coordinates.row(b);
    c1.row(b) = s.coordinates.row(a);
    (void)c.compute_error(s.coordinates, mv);
    (void)c.compute_error(c1, mv);
    require_close(c.computed(), full_computed(s, bc, ap, c1));
  }
}

TEST_CASE("ADF rollback restores the cached histogram and grid",
          "[constraints][adf][incremental]") {
  const std::vector<std::string> els{"Cu", "Zr"};
  const std::vector<std::size_t> cnts{32, 32};
  auto cell = make_random_amorphous(els, cnts, 2.8, /*seed=*/12);
  REQUIRE(cell);
  const AtomicStructure &s = cell->structure;
  const BoundaryConditions bc = cell->periodic_bc();
  analysis::AdfParams ap;
  ap.max_dis = 3.6;
  ap.n_bins = 45;

  auto c = make_adf(s, bc, ap);
  (void)c.compute_error(s.coordinates, {});
  const vec_t before = full_computed(s, bc, ap, s.coordinates);

  const std::array<std::size_t, 1> mv{9};
  coords_t c1 = s.coordinates;
  c1.row(9) += vec3_t{0.8, 0.6, -0.5}.transpose();
  (void)c.compute_error(s.coordinates, mv);
  (void)c.compute_error(c1, mv); // after-move mutates the grid
  c.rollback_frame();            // reject → undo histogram + grid

  // A fresh before-move at the original coords must reproduce the pre-move state
  // (and the grid must be back so a subsequent incremental step is correct).
  (void)c.compute_error(s.coordinates, mv);
  require_close(c.computed(), before);
  coords_t c2 = s.coordinates;
  c2.row(9) += vec3_t{-0.3, 0.4, 0.2}.transpose();
  (void)c.compute_error(c2, mv);
  require_close(c.computed(), full_computed(s, bc, ap, c2));
}

TEST_CASE("ADF incremental handles atom removal", "[constraints][adf][incremental]") {
  const std::vector<std::string> els{"Cu", "Zr"};
  const std::vector<std::size_t> cnts{32, 32};
  auto cell = make_random_amorphous(els, cnts, 2.8, /*seed=*/13);
  REQUIRE(cell);
  const AtomicStructure &s = cell->structure;
  const BoundaryConditions bc = cell->periodic_bc();
  analysis::AdfParams ap;
  ap.max_dis = 3.6;
  ap.n_bins = 45;

  AtomsCollector col;
  auto c = make_adf(s, bc, ap, &col);
  (void)c.compute_error(s.coordinates, {}); // seed with all atoms present

  const std::size_t m = 15;
  const std::array<std::size_t, 1> mv{m};
  (void)c.compute_error(s.coordinates, mv); // before-move: m still present
  col.stage_removal(std::array<std::size_t, 1>{m}); // engine stages removal
  (void)c.compute_error(s.coordinates, mv); // after-move: m now absent

  require_close(c.computed(), full_computed(s, bc, ap, s.coordinates, &col));
}

namespace {
// Build a 2-frame ADF with each frame's histogram populated from f0 / f1.
AngularDistributionConstraint make_2frame_adf(const AtomicStructure &s,
                                              const coords_t &f0,
                                              const coords_t &f1,
                                              const BoundaryConditions &bc,
                                              const analysis::AdfParams &ap) {
  auto c = make_adf(s, bc, ap);
  c.set_n_frames(2);
  c.set_active_frame_idx(0);
  (void)c.compute_error(f0, {});
  c.set_active_frame_idx(1);
  (void)c.compute_error(f1, {});
  return c;
}
} // namespace

TEST_CASE("ADF multi-frame incremental matches full + frame-switch reset",
          "[constraints][adf][incremental][multiframe]") {
  const std::vector<std::string> els{"Cu", "Zr"};
  const std::vector<std::size_t> cnts{32, 32};
  auto cell0 = make_random_amorphous(els, cnts, 2.8, /*seed=*/21);
  auto cell1 = make_random_amorphous(els, cnts, 2.85, /*seed=*/22);
  REQUIRE(cell0);
  REQUIRE(cell1);
  const AtomicStructure &s = cell0->structure;
  const BoundaryConditions bc = cell0->periodic_bc();
  const coords_t f0 = cell0->structure.coordinates;
  const coords_t f1 = cell1->structure.coordinates;
  analysis::AdfParams ap;
  ap.max_dis = 3.6;
  ap.n_bins = 45;

  SECTION("incremental step on frame 0 matches full rebuild") {
    coords_t f0_mod = f0;
    const std::array<std::size_t, 1> mv{6};
    f0_mod.row(6) += vec3_t{0.5, -0.4, 0.3}.transpose();

    auto inc = make_2frame_adf(s, f0, f1, bc, ap);
    inc.set_active_frame_idx(0);
    (void)inc.compute_error(f0, mv);     // before-move
    (void)inc.compute_error(f0_mod, mv); // after-move (averaged over frames)

    // Reference: frame 0 built directly from the modified coords.
    auto ref = make_2frame_adf(s, f0_mod, f1, bc, ap);
    require_close(inc.computed(), ref.computed());
  }

  SECTION("frame switch resets pending incremental state") {
    auto inc = make_2frame_adf(s, f0, f1, bc, ap);
    const std::array<std::size_t, 1> mv{6};
    // Stage a before-move delta on frame 0, then switch away WITHOUT an
    // after-move. The switch must clear incremental_ready_ so the frame-1
    // before-move is not mistaken for an after-move using frame 0's delta.
    inc.set_active_frame_idx(0);
    (void)inc.compute_error(f0, mv);

    coords_t f1_mod = f1;
    f1_mod.row(6) += vec3_t{0.45, 0.35, -0.4}.transpose();
    inc.set_active_frame_idx(1);
    (void)inc.compute_error(f1, mv);     // before-move on frame 1
    (void)inc.compute_error(f1_mod, mv); // after-move on frame 1

    auto ref = make_2frame_adf(s, f0, f1_mod, bc, ap);
    require_close(inc.computed(), ref.computed());
  }
}

TEST_CASE("ADF incremental tracks a full accept/reject trajectory",
          "[constraints][adf][incremental]") {
  const std::vector<std::string> els{"Cu", "Zr"};
  const std::vector<std::size_t> cnts{32, 32};
  auto cell = make_random_amorphous(els, cnts, 2.8, /*seed=*/31);
  REQUIRE(cell);
  const AtomicStructure &s = cell->structure;
  const BoundaryConditions bc = cell->periodic_bc();
  analysis::AdfParams ap;
  ap.max_dis = 3.6;
  ap.n_bins = 45;

  auto c = make_adf(s, bc, ap);
  coords_t cur = s.coordinates;
  (void)c.compute_error(cur, {}); // seed

  // Deterministic LCG so the trajectory is reproducible without <random>.
  std::uint64_t rng = 0x9e3779b97f4a7c15ULL;
  auto next = [&]() {
    rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<double>(rng >> 11) / static_cast<double>(1ULL << 53);
  };
  const auto N = static_cast<std::size_t>(cur.rows());

  for (int step = 0; step < 60; ++step) {
    const std::size_t m = static_cast<std::size_t>(next() * N) % N;
    const std::array<std::size_t, 1> mv{m};
    // Mix of small and cutoff-crossing displacements.
    const double mag = (step % 4 == 0) ? 1.8 : 0.35;
    coords_t trial = cur;
    trial.row(static_cast<Eigen::Index>(m)) +=
        (vec3_t{next() - 0.5, next() - 0.5, next() - 0.5} * mag).transpose();

    (void)c.compute_error(cur, mv);   // before-move
    (void)c.compute_error(trial, mv); // after-move
    require_close(c.computed(), full_computed(s, bc, ap, trial), 1e-9);

    if (next() < 0.6) { // accept
      cur = trial;
      c.commit_frame();
    } else { // reject
      c.rollback_frame();
    }
  }
}

static_assert(CConstraint<AngularDistributionConstraint>);
