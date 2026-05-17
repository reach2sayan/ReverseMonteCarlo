#include <RMC/generators/Combined.hpp>
#include <RMC/generators/Rotations.hpp>
#include <RMC/generators/Swaps.hpp>
#include <RMC/generators/Translations.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
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
