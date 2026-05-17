#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <fullrmc/generators/Translations.hpp>
#include <fullrmc/generators/Rotations.hpp>
#include <fullrmc/generators/Swaps.hpp>
#include <fullrmc/generators/Combined.hpp>
#include <cmath>
#include <vector>
#include <numeric>

using namespace fullrmc;
using Catch::Matchers::WithinAbs;

static coords_t make_coords(std::initializer_list<std::array<double,3>> atoms) {
    coords_t c(static_cast<Eigen::Index>(atoms.size()), 3);
    Eigen::Index i = 0;
    for (auto& a : atoms) { c(i,0)=a[0]; c(i,1)=a[1]; c(i,2)=a[2]; ++i; }
    return c;
}

TEST_CASE("TranslationGenerator - amplitude within bounds", "[generators]") {
    TranslationGenerator gen(0.1, 0.5, /*seed=*/99);
    coords_t c = make_coords({{0,0,0},{1,0,0},{0,1,0}});
    coords_t orig = c;
    std::vector<index_t> all = {0, 1, 2};

    // All atoms in the group move by the same displacement.
    gen.generate(c, all);

    // Displacement magnitude must be in [0.1, 0.5].
    double dx = c(0,0) - orig(0,0);
    double dy = c(0,1) - orig(0,1);
    double dz = c(0,2) - orig(0,2);
    double dist = std::sqrt(dx*dx + dy*dy + dz*dz);
    REQUIRE(dist >= 0.1 - 1e-9);
    REQUIRE(dist <= 0.5 + 1e-9);

    // All atoms shift by exactly the same vector.
    for (int i = 1; i < 3; ++i) {
        REQUIRE_THAT(c(i,0) - orig(i,0), WithinAbs(dx, 1e-12));
        REQUIRE_THAT(c(i,1) - orig(i,1), WithinAbs(dy, 1e-12));
        REQUIRE_THAT(c(i,2) - orig(i,2), WithinAbs(dz, 1e-12));
    }
}

TEST_CASE("TranslationGenerator - subset indices", "[generators]") {
    TranslationGenerator gen(0.0, 0.2, 7);
    coords_t c = make_coords({{0,0,0},{1,0,0},{2,0,0}});
    coords_t orig = c;
    std::vector<index_t> subset = {0, 2};  // skip atom 1

    gen.generate(c, subset);

    // Atom 1 must be unmoved.
    REQUIRE_THAT(c(1,0), WithinAbs(1.0, 1e-12));
    REQUIRE_THAT(c(1,1), WithinAbs(0.0, 1e-12));
    REQUIRE_THAT(c(1,2), WithinAbs(0.0, 1e-12));
}

TEST_CASE("RotationGenerator - preserves pairwise distances", "[generators]") {
    RotationGenerator gen(0.1, 0.5, 42);
    coords_t c = make_coords({{0,0,0},{1,0,0},{0,1,0},{0,0,1}});
    std::vector<index_t> all = {0,1,2,3};

    // Record all pairwise distances before.
    auto dist_ij = [&](int i, int j) {
        return (c.row(i) - c.row(j)).norm();
    };
    double d01 = dist_ij(0,1), d02 = dist_ij(0,2), d12 = dist_ij(1,2);

    gen.generate(c, all);

    // Pairwise distances must be preserved (rigid body rotation).
    REQUIRE_THAT(dist_ij(0,1), WithinAbs(d01, 1e-9));
    REQUIRE_THAT(dist_ij(0,2), WithinAbs(d02, 1e-9));
    REQUIRE_THAT(dist_ij(1,2), WithinAbs(d12, 1e-9));
}

TEST_CASE("SwapGenerator - exchanges positions correctly", "[generators]") {
    coords_t c = make_coords({{1,0,0},{2,0,0},{3,0,0},{4,0,0}});
    // Group A = {0,1}, pool has one candidate: {2,3}
    SwapGenerator gen({{{2,3}}}, 1);
    std::vector<index_t> groupA = {0,1};

    gen.generate(c, groupA);

    // After swap: rows 0,1 should have what was rows 2,3 and vice versa.
    REQUIRE_THAT(c(0,0), WithinAbs(3.0, 1e-12));
    REQUIRE_THAT(c(1,0), WithinAbs(4.0, 1e-12));
    REQUIRE_THAT(c(2,0), WithinAbs(1.0, 1e-12));
    REQUIRE_THAT(c(3,0), WithinAbs(2.0, 1e-12));
}

TEST_CASE("CombinedMoveGenerator - applies both generators", "[generators]") {
    auto t = std::make_shared<TranslationGenerator>(0.2, 0.2, 10);
    auto r = std::make_shared<RotationGenerator>(0.0, 0.0, 10); // zero rotation
    CombinedMoveGenerator comb({t, r});

    coords_t c = make_coords({{0,0,0}});
    coords_t orig = c;
    std::vector<index_t> idx = {0};
    comb.generate(c, idx);

    // With zero rotation, only translation applies.
    double d = (c.row(0) - orig.row(0)).norm();
    REQUIRE_THAT(d, WithinAbs(0.2, 1e-9));
}
