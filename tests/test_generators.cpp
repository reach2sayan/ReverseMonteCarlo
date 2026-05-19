#include <RMC/generators/Agitations.hpp>
#include <RMC/generators/Combined.hpp>
#include <RMC/generators/Path.hpp>
#include <RMC/generators/Rotations.hpp>
#include <RMC/generators/Swaps.hpp>
#include <RMC/generators/Translations.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <numbers> // std::numbers::pi
#include <vector>

using namespace RMC;
using Catch::Matchers::WithinAbs;

static coords_t
make_coords(std::initializer_list<std::array<double, 3>> atoms) {
  coords_t c(static_cast<Eigen::Index>(atoms.size()), 3);
  Eigen::Index i = 0;
  for (auto &a : atoms) {
    c(i, 0) = a[0];
    c(i, 1) = a[1];
    c(i, 2) = a[2];
    ++i;
  }
  return c;
}

TEST_CASE("TranslationGenerator - amplitude within bounds", "[generators]") {
  IMoveGenerator gen = TranslationGenerator(0.1, 0.5, /*seed=*/99);
  coords_t c = make_coords({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}});
  coords_t orig = c;
  std::vector<index_t> all = {0, 1, 2};

  gen.generate(c, all);

  double dx = c(0, 0) - orig(0, 0);
  double dy = c(0, 1) - orig(0, 1);
  double dz = c(0, 2) - orig(0, 2);
  double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
  REQUIRE(dist >= 0.1 - 1e-9);
  REQUIRE(dist <= 0.5 + 1e-9);

  for (int i = 1; i < 3; ++i) {
    REQUIRE_THAT(c(i, 0) - orig(i, 0), WithinAbs(dx, 1e-12));
    REQUIRE_THAT(c(i, 1) - orig(i, 1), WithinAbs(dy, 1e-12));
    REQUIRE_THAT(c(i, 2) - orig(i, 2), WithinAbs(dz, 1e-12));
  }
}

TEST_CASE("TranslationGenerator - subset indices", "[generators]") {
  IMoveGenerator gen = TranslationGenerator(0.0, 0.2, 7);
  coords_t c = make_coords({{0, 0, 0}, {1, 0, 0}, {2, 0, 0}});
  std::vector<index_t> subset = {0, 2};

  gen.generate(c, subset);

  REQUIRE_THAT(c(1, 0), WithinAbs(1.0, 1e-12));
  REQUIRE_THAT(c(1, 1), WithinAbs(0.0, 1e-12));
  REQUIRE_THAT(c(1, 2), WithinAbs(0.0, 1e-12));
}

TEST_CASE("RotationGenerator - preserves pairwise distances", "[generators]") {
  IMoveGenerator gen = RotationGenerator(0.1, 0.5, 42);
  coords_t c = make_coords({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}});
  std::vector<index_t> all = {0, 1, 2, 3};

  auto dist_ij = [&](int i, int j) { return (c.row(i) - c.row(j)).norm(); };
  double d01 = dist_ij(0, 1), d02 = dist_ij(0, 2), d12 = dist_ij(1, 2);

  gen.generate(c, all);

  REQUIRE_THAT(dist_ij(0, 1), WithinAbs(d01, 1e-9));
  REQUIRE_THAT(dist_ij(0, 2), WithinAbs(d02, 1e-9));
  REQUIRE_THAT(dist_ij(1, 2), WithinAbs(d12, 1e-9));
}

TEST_CASE("SwapGenerator - exchanges positions correctly", "[generators]") {
  IMoveGenerator gen = SwapGenerator({{{2, 3}}}, 1);
  coords_t c = make_coords({{1, 0, 0}, {2, 0, 0}, {3, 0, 0}, {4, 0, 0}});
  std::vector<index_t> groupA = {0, 1};

  gen.generate(c, groupA);

  REQUIRE_THAT(c(0, 0), WithinAbs(3.0, 1e-12));
  REQUIRE_THAT(c(1, 0), WithinAbs(4.0, 1e-12));
  REQUIRE_THAT(c(2, 0), WithinAbs(1.0, 1e-12));
  REQUIRE_THAT(c(3, 0), WithinAbs(2.0, 1e-12));
}

TEST_CASE("CombinedMoveGenerator - applies both generators", "[generators]") {
  IMoveGenerator gen = CombinedMoveGenerator{TranslationGenerator(0.2, 0.2, 10),
                                             RotationGenerator(0.0, 0.0, 10)};

  coords_t c = make_coords({{0, 0, 0}});
  coords_t orig = c;
  std::vector<index_t> idx = {0};
  gen.generate(c, idx);

  double d = (c.row(0) - orig.row(0)).norm();
  REQUIRE_THAT(d, WithinAbs(0.2, 1e-9));
}

// ---- TranslationTowardsAxisGenerator ----
TEST_CASE("TranslationTowardsAxisGenerator - moves centroid closer to axis",
          "[generators]") {
  // Axis is the Z-axis through origin. Group centroid at (3, 4, 0).
  // After the move the perpendicular distance to Z should decrease.
  vec3_t pt{0, 0, 0};
  vec3_t dir{0, 0, 1};
  IMoveGenerator gen = TranslationTowardsAxisGenerator(pt, dir, 0.5, 0.5, 1);
  coords_t c = make_coords({{3, 4, 0}, {3, 4, 1}});
  std::vector<index_t> all = {0, 1};

  // Distance from Z-axis before = sqrt(9+16) = 5.
  auto perp_dist = [&]() {
    double cx = c(0, 0) + c(1, 0);
    double cy = c(0, 1) + c(1, 1);
    return std::sqrt(cx * cx + cy * cy) / 2.0;
  };
  double before = perp_dist();
  gen.generate(c, all);
  REQUIRE(perp_dist() < before);
}

TEST_CASE("TranslationTowardsAxisGenerator - no-op when centroid on axis",
          "[generators]") {
  vec3_t pt{0, 0, 0};
  vec3_t dir{0, 0, 1};
  IMoveGenerator gen = TranslationTowardsAxisGenerator(pt, dir, 0.5, 0.5, 1);
  coords_t c = make_coords({{0, 0, 0}, {0, 0, 2}});
  coords_t orig = c;
  std::vector<index_t> all = {0, 1};
  gen.generate(c, all);
  REQUIRE_THAT((c - orig).norm(), WithinAbs(0.0, 1e-12));
}

// ---- TranslationAlongSymmetryAxisGenerator ----
TEST_CASE("TranslationAlongSymmetryAxisGenerator - moves only along chosen axis",
          "[generators]") {
  for (auto [ax, col] : {std::pair{SymmetryAxis::X, 0},
                          std::pair{SymmetryAxis::Y, 1},
                          std::pair{SymmetryAxis::Z, 2}}) {
    IMoveGenerator gen = TranslationAlongSymmetryAxisGenerator(ax, 0.3, 0.3, 5);
    coords_t c = make_coords({{1, 1, 1}});
    coords_t orig = c;
    std::vector<index_t> idx = {0};
    gen.generate(c, idx);
    for (int i = 0; i < 3; ++i) {
      if (i == col)
        REQUIRE(std::abs(c(0, i) - orig(0, i)) > 1e-9);
      else
        REQUIRE_THAT(c(0, i), WithinAbs(orig(0, i), 1e-12));
    }
  }
}

// ---- TranslationTowardsSymmetryAxisGenerator ----
TEST_CASE("TranslationTowardsSymmetryAxisGenerator - perpendicular dist decreases",
          "[generators]") {
  // Group centroid at (3, 4, 7), symmetry axis Z. Perp dist = 5.
  IMoveGenerator gen =
      TranslationTowardsSymmetryAxisGenerator(SymmetryAxis::Z, 1.0, 1.0, 2);
  coords_t c = make_coords({{3, 4, 7}});
  std::vector<index_t> idx = {0};
  double before = std::sqrt(c(0, 0) * c(0, 0) + c(0, 1) * c(0, 1));
  gen.generate(c, idx);
  double after = std::sqrt(c(0, 0) * c(0, 0) + c(0, 1) * c(0, 1));
  REQUIRE(after < before);
  // Z coordinate must be unchanged.
  REQUIRE_THAT(c(0, 2), WithinAbs(7.0, 1e-12));
}

// ---- RotationAboutSymmetryAxisGenerator ----
TEST_CASE("RotationAboutSymmetryAxisGenerator - preserves pairwise distances",
          "[generators]") {
  for (auto ax : {SymmetryAxis::X, SymmetryAxis::Y, SymmetryAxis::Z}) {
    IMoveGenerator gen =
        RotationAboutSymmetryAxisGenerator(ax, 0.3, 0.3, 7);
    coords_t c = make_coords({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}});
    double d01 = (c.row(0) - c.row(1)).norm();
    double d23 = (c.row(2) - c.row(3)).norm();
    std::vector<index_t> all = {0, 1, 2, 3};
    gen.generate(c, all);
    REQUIRE_THAT((c.row(0) - c.row(1)).norm(), WithinAbs(d01, 1e-9));
    REQUIRE_THAT((c.row(2) - c.row(3)).norm(), WithinAbs(d23, 1e-9));
  }
}

// ---- OrientationGenerator ----
TEST_CASE("OrientationGenerator - preserves pairwise distances", "[generators]") {
  vec3_t target{0, 0, 1};
  IMoveGenerator gen = OrientationGenerator(target, 0.05, 3);
  coords_t c = make_coords({{0, 0, 0}, {1, 0, 0}, {2, 0, 0}});
  double d01 = (c.row(0) - c.row(1)).norm();
  double d12 = (c.row(1) - c.row(2)).norm();
  std::vector<index_t> all = {0, 1, 2};
  gen.generate(c, all);
  REQUIRE_THAT((c.row(0) - c.row(1)).norm(), WithinAbs(d01, 1e-9));
  REQUIRE_THAT((c.row(1) - c.row(2)).norm(), WithinAbs(d12, 1e-9));
}

TEST_CASE("OrientationGenerator - aligns axis approximately toward target",
          "[generators]") {
  // Group along X. Target is Z. Zero offset → exact alignment.
  vec3_t target{0, 0, 1};
  IMoveGenerator gen = OrientationGenerator(target, 0.0, 11);
  coords_t c = make_coords({{-1, 0, 0}, {0, 0, 0}, {1, 0, 0}});
  std::vector<index_t> all = {0, 1, 2};
  gen.generate(c, all);
  // Principal axis after move: indices[0] → indices[2]
  vec3_t ax = (c.row(2) - c.row(0)).transpose();
  ax.normalize();
  double cos_a = std::abs(ax.dot(target)); // abs for head/tail ambiguity
  REQUIRE(cos_a > 0.99);
}

// ---- DistanceAgitationGenerator ----
TEST_CASE("DistanceAgitationGenerator - midpoint preserved", "[generators]") {
  IMoveGenerator gen = DistanceAgitationGenerator(0, 1, 0.1, 0.1, 4);
  coords_t c = make_coords({{0, 0, 0}, {2, 0, 0}});
  coords_t orig = c;
  std::vector<index_t> all = {0, 1};
  gen.generate(c, all);
  // Midpoint must stay at (1,0,0).
  double mid_x = 0.5 * (c(0, 0) + c(1, 0));
  double mid_y = 0.5 * (c(0, 1) + c(1, 1));
  double mid_z = 0.5 * (c(0, 2) + c(1, 2));
  REQUIRE_THAT(mid_x, WithinAbs(1.0, 1e-12));
  REQUIRE_THAT(mid_y, WithinAbs(0.0, 1e-12));
  REQUIRE_THAT(mid_z, WithinAbs(0.0, 1e-12));
}

TEST_CASE("DistanceAgitationGenerator - bond length changes", "[generators]") {
  IMoveGenerator gen = DistanceAgitationGenerator(0, 1, 0.2, 0.2, 4);
  coords_t c = make_coords({{0, 0, 0}, {2, 0, 0}});
  std::vector<index_t> all = {0, 1};
  double before = (c.row(1) - c.row(0)).norm();
  gen.generate(c, all);
  double after = (c.row(1) - c.row(0)).norm();
  REQUIRE(std::abs(after - before) > 1e-9);
}

// ---- AngleAgitationGenerator ----
TEST_CASE("AngleAgitationGenerator - bond lengths preserved", "[generators]") {
  // Triplet (0,1,2): atom 1 at origin, atom 0 along x, atom 2 along y.
  IMoveGenerator gen = AngleAgitationGenerator(0, 1, 2, 0.1, 0.1, 5);
  coords_t c = make_coords({{1, 0, 0}, {0, 0, 0}, {0, 1, 0}});
  std::vector<index_t> all = {0, 1, 2};
  double d01 = (c.row(0) - c.row(1)).norm();
  double d21 = (c.row(2) - c.row(1)).norm();
  gen.generate(c, all);
  REQUIRE_THAT((c.row(0) - c.row(1)).norm(), WithinAbs(d01, 1e-9));
  REQUIRE_THAT((c.row(2) - c.row(1)).norm(), WithinAbs(d21, 1e-9));
}

TEST_CASE("AngleAgitationGenerator - angle changes", "[generators]") {
  IMoveGenerator gen = AngleAgitationGenerator(0, 1, 2, 0.2, 0.2, 5);
  coords_t c = make_coords({{1, 0, 0}, {0, 0, 0}, {0, 1, 0}});
  std::vector<index_t> all = {0, 1, 2};
  // Initial angle at vertex 1 = 90 degrees.
  auto angle_at_1 = [&]() {
    vec3_t a = (c.row(0) - c.row(1)).transpose().normalized();
    vec3_t b = (c.row(2) - c.row(1)).transpose().normalized();
    return std::acos(std::clamp(a.dot(b), -1.0, 1.0));
  };
  double before = angle_at_1();
  gen.generate(c, all);
  double after = angle_at_1();
  REQUIRE(std::abs(after - before) > 1e-9);
}

// ---- SwapCentersGenerator ----
TEST_CASE("SwapCentersGenerator - moves group centroid to candidate centroid",
          "[generators]") {
  // Group = {0}, candidate = {1}. After move, atom 0 should be at atom 1's pos.
  IMoveGenerator gen = SwapCentersGenerator({{std::vector<std::size_t>{1}}}, 1);
  coords_t c = make_coords({{0, 0, 0}, {5, 3, 1}});
  std::vector<index_t> group = {0};
  gen.generate(c, group);
  REQUIRE_THAT(c(0, 0), WithinAbs(5.0, 1e-12));
  REQUIRE_THAT(c(0, 1), WithinAbs(3.0, 1e-12));
  REQUIRE_THAT(c(0, 2), WithinAbs(1.0, 1e-12));
  // Candidate atom must be unmoved.
  REQUIRE_THAT(c(1, 0), WithinAbs(5.0, 1e-12));
}

TEST_CASE("SwapCentersGenerator - no-op with empty candidates", "[generators]") {
  IMoveGenerator gen = SwapCentersGenerator({}, 1);
  coords_t c = make_coords({{1, 2, 3}});
  coords_t orig = c;
  std::vector<index_t> idx = {0};
  gen.generate(c, idx);
  REQUIRE_THAT((c - orig).norm(), WithinAbs(0.0, 1e-12));
}

// ---- MoveGeneratorCollector ----
TEST_CASE("MoveGeneratorCollector - applies exactly one generator",
          "[generators]") {
  // Two generators: one shifts +x by 1.0, one shifts +y by 1.0.
  // After N calls, every atom has moved exactly once per axis (not both).
  // We check that each call shifts only one of x or y (not both).
  MoveGeneratorCollector col(99);
  col.add(TranslationAlongSymmetryAxisGenerator(SymmetryAxis::X, 1.0, 1.0, 1));
  col.add(TranslationAlongSymmetryAxisGenerator(SymmetryAxis::Y, 1.0, 1.0, 2));
  IMoveGenerator gen = std::move(col);

  int x_moves = 0, y_moves = 0;
  for (int trial = 0; trial < 40; ++trial) {
    coords_t c = make_coords({{0, 0, 0}});
    std::vector<index_t> idx = {0};
    gen.generate(c, idx);
    bool moved_x = std::abs(c(0, 0)) > 1e-9;
    bool moved_y = std::abs(c(0, 1)) > 1e-9;
    REQUIRE((moved_x ^ moved_y)); // exactly one must be true
    x_moves += moved_x ? 1 : 0;
    y_moves += moved_y ? 1 : 0;
  }
  // With 40 trials both generators should have been selected at least once.
  REQUIRE(x_moves > 0);
  REQUIRE(y_moves > 0);
}

// ---- TranslationAlongAxisPath ----
TEST_CASE("TranslationAlongAxisPath - applies steps sequentially and cycles",
          "[generators]") {
  std::vector<double> steps = {1.0, -2.0, 0.5};
  IMoveGenerator gen = TranslationAlongAxisPath({1, 0, 0}, steps);
  coords_t c = make_coords({{0, 0, 0}});
  std::vector<index_t> idx = {0};

  double expected_x = 0.0;
  for (int rep = 0; rep < 6; ++rep) {
    gen.generate(c, idx);
    expected_x += steps[rep % 3];
    REQUIRE_THAT(c(0, 0), WithinAbs(expected_x, 1e-12));
  }
}

// ---- RotationAboutAxisPath ----
TEST_CASE("RotationAboutAxisPath - preserves distances and cycles",
          "[generators]") {
  std::vector<double> angles = {std::numbers::pi / 6, -std::numbers::pi / 6};
  IMoveGenerator gen = RotationAboutAxisPath({0, 0, 1}, angles);
  coords_t c = make_coords({{1, 0, 0}, {0, 1, 0}, {-1, 0, 0}});
  std::vector<index_t> all = {0, 1, 2};
  double d01 = (c.row(0) - c.row(1)).norm();
  double d12 = (c.row(1) - c.row(2)).norm();

  for (int rep = 0; rep < 4; ++rep) {
    gen.generate(c, all);
    REQUIRE_THAT((c.row(0) - c.row(1)).norm(), WithinAbs(d01, 1e-9));
    REQUIRE_THAT((c.row(1) - c.row(2)).norm(), WithinAbs(d12, 1e-9));
  }
}

TEST_CASE("RotationAboutAxisPath - net rotation after full cycle is identity",
          "[generators]") {
  // +π/2 then -π/2 → net zero.
  std::vector<double> angles = {std::numbers::pi / 2, -std::numbers::pi / 2};
  IMoveGenerator gen = RotationAboutAxisPath({0, 0, 1}, angles);
  coords_t c = make_coords({{1, 0, 0}, {0, 0, 0}});
  coords_t orig = c;
  std::vector<index_t> all = {0, 1};
  gen.generate(c, all);
  gen.generate(c, all);
  REQUIRE_THAT((c - orig).norm(), WithinAbs(0.0, 1e-9));
}
