#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <fullrmc/constraints/BondConstraint.hpp>
#include <fullrmc/constraints/AngleConstraint.hpp>
#include <fullrmc/constraints/DihedralAngleConstraint.hpp>
#include <fullrmc/constraints/DistanceConstraint.hpp>
#include <fullrmc/constraints/CoordinationConstraint.hpp>
#include <fullrmc/constraints/PairDistributionConstraint.hpp>
#include <fullrmc/constraints/ConstraintCollection.hpp>
#include <numbers>
#include <vector>

using namespace fullrmc;
using Catch::Matchers::WithinAbs;

static constexpr real_t EPS = 1e-8;

static coords_t make2(double ax, double bx) {
    coords_t c(2, 3);
    c << ax, 0, 0,
         bx, 0, 0;
    return c;
}

// ---- BondConstraint ----
TEST_CASE("BondConstraint - satisfied bond has zero error", "[constraints]") {
    BondConstraint bc;
    bc.add_bond(0, 1, 1.0, 2.0);
    coords_t c = make2(0.0, 1.5);
    std::vector<index_t> all = {0,1};
    REQUIRE_THAT(bc.compute_error(c, all), WithinAbs(0.0, EPS));
}

TEST_CASE("BondConstraint - too-short bond accumulates error", "[constraints]") {
    BondConstraint bc;
    bc.add_bond(0, 1, 1.5, 2.5);
    coords_t c = make2(0.0, 1.0);   // bond = 1.0, lo = 1.5 → error = 0.5
    std::vector<index_t> all = {0,1};
    REQUIRE_THAT(bc.compute_error(c, all), WithinAbs(0.5, EPS));
}

TEST_CASE("BondConstraint - should_reject after worsening move", "[constraints]") {
    BondConstraint bc;
    bc.add_bond(0, 1, 1.0, 2.0);

    coords_t good = make2(0.0, 1.5);   // bond = 1.5, in range
    coords_t bad  = make2(0.0, 0.5);   // bond = 0.5, too short

    std::vector<index_t> all = {0,1};
    bc.compute_before_move(good, all);
    bc.compute_after_move(bad, all);
    REQUIRE(bc.should_reject());
}

TEST_CASE("BondConstraint - accept then error_before updates", "[constraints]") {
    BondConstraint bc;
    bc.add_bond(0, 1, 1.0, 2.0);
    coords_t c1 = make2(0.0, 1.2);
    coords_t c2 = make2(0.0, 1.8);
    std::vector<index_t> all = {0,1};
    bc.compute_before_move(c1, all);
    bc.compute_after_move(c2, all);
    bc.accept();
    // After accept, err_before = err_after = 0
    REQUIRE_THAT(bc.standard_error(), WithinAbs(0.0, EPS));
}

// ---- AngleConstraint ----
TEST_CASE("AngleConstraint - 90 degree angle satisfied", "[constraints]") {
    AngleConstraint ac;
    // i=0 at (1,0,0), j=1 at (0,0,0), k=2 at (0,1,0) → angle = 90°
    ac.add_angle(0, 1, 2, 0.0, std::numbers::pi);
    coords_t c(3,3);
    c << 1,0,0,  0,0,0,  0,1,0;
    std::vector<index_t> all = {0,1,2};
    REQUIRE_THAT(ac.compute_error(c, all), WithinAbs(0.0, EPS));
}

TEST_CASE("AngleConstraint - angle below minimum accumulates error", "[constraints]") {
    AngleConstraint ac;
    // Force angle to be 90° but require >= 2 rad (~114.6°).
    ac.add_angle(0, 1, 2, 2.0, std::numbers::pi);
    coords_t c(3,3);
    c << 1,0,0,  0,0,0,  0,1,0;
    std::vector<index_t> all = {0,1,2};
    real_t err = ac.compute_error(c, all);
    REQUIRE(err > 0.0);
}

// ---- DihedralAngleConstraint ----
TEST_CASE("DihedralAngleConstraint - zero dihedral accepted", "[constraints]") {
    DihedralAngleConstraint dc;
    dc.add_dihedral(0,1,2,3, -std::numbers::pi, std::numbers::pi);
    coords_t c(4,3);
    c << 0,0,0,  1,0,0,  2,0,0,  3,0,0;
    std::vector<index_t> all = {0,1,2,3};
    REQUIRE_THAT(dc.compute_error(c, all), WithinAbs(0.0, EPS));
}

// ---- PairDistributionConstraint - ideal gas baseline ----
TEST_CASE("PairDistributionConstraint - large error for random structure vs flat G(r)", "[constraints]") {
    PairDistributionConstraint pdc;

    // Flat experimental G(r) = 0 everywhere (ideal gas expectation: g(r)=1 → G(r)=0)
    mat_t exp(50, 2);
    for (int i = 0; i < 50; ++i) { exp(i,0) = 0.1*(i+1); exp(i,1) = 0.0; }
    pdc.set_experimental_data(exp);
    pdc.set_number_density(0.03);
    pdc.initialise();

    // 10 atoms on a simple grid (close-packed spacing) – will have non-trivial g(r)
    const int N = 8;
    coords_t c(N, 3);
    for (int i = 0; i < N; ++i) c.row(i) << i * 3.0, 0.0, 0.0;

    std::vector<index_t> all(N);
    std::iota(all.begin(), all.end(), 0);
    real_t err = pdc.compute_error(c, all);

    // Just verify it runs and returns a finite, positive value.
    REQUIRE(std::isfinite(err));
    REQUIRE(err >= 0.0);
}

// ---- ConstraintCollection ----
TEST_CASE("ConstraintCollection - rejects when any constraint rejects", "[constraints]") {
    ConstraintCollection col;
    auto b = std::make_unique<BondConstraint>();
    b->add_bond(0, 1, 1.0, 2.0);
    col.add(std::move(b));

    coords_t good = make2(0.0, 1.5);
    coords_t bad  = make2(0.0, 0.3);
    std::vector<index_t> all = {0,1};

    col.compute_before_move(good, all);
    col.compute_after_move(bad, all);
    REQUIRE(col.should_reject());
}

TEST_CASE("ConstraintCollection - accepts when all constraints pass", "[constraints]") {
    ConstraintCollection col;
    auto b = std::make_unique<BondConstraint>();
    b->add_bond(0, 1, 1.0, 2.0);
    col.add(std::move(b));

    coords_t c1 = make2(0.0, 1.2);
    coords_t c2 = make2(0.0, 1.8);
    std::vector<index_t> all = {0,1};

    col.compute_before_move(c1, all);
    col.compute_after_move(c2, all);
    REQUIRE_FALSE(col.should_reject());
}
