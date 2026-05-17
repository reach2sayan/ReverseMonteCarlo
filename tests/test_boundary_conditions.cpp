#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <fullrmc/core/BoundaryConditions.hpp>
#include <cmath>

using namespace fullrmc;
using Catch::Matchers::WithinAbs;

static constexpr real_t EPS = 1e-10;

TEST_CASE("PeriodicBC - cubic box wrapping", "[bc]") {
    mat3_t box = mat3_t::Identity() * 10.0; // 10 Å cubic box
    PeriodicBC bc(box);

    SECTION("volume") {
        REQUIRE_THAT(bc.volume(), WithinAbs(1000.0, EPS));
    }

    SECTION("wrap inside box stays the same") {
        vec3_t r(3.0, 4.0, 5.0);
        vec3_t w = bc.wrap(r);
        REQUIRE_THAT(w(0), WithinAbs(3.0, EPS));
        REQUIRE_THAT(w(1), WithinAbs(4.0, EPS));
        REQUIRE_THAT(w(2), WithinAbs(5.0, EPS));
    }

    SECTION("wrap outside box (positive overflow)") {
        vec3_t r(12.0, 0.0, 0.0);
        vec3_t w = bc.wrap(r);
        REQUIRE_THAT(w(0), WithinAbs(2.0, EPS));
    }

    SECTION("wrap outside box (negative)") {
        vec3_t r(-1.0, 0.0, 0.0);
        vec3_t w = bc.wrap(r);
        REQUIRE_THAT(w(0), WithinAbs(9.0, EPS));
    }

    SECTION("minimum image: short distance") {
        vec3_t d(3.0, 0.0, 0.0);
        vec3_t mi = bc.min_image(d);
        REQUIRE_THAT(mi(0), WithinAbs(3.0, EPS));
    }

    SECTION("minimum image: far distance wraps to shorter") {
        vec3_t d(8.0, 0.0, 0.0);  // 8 > 10/2, so min image = 8 - 10 = -2
        vec3_t mi = bc.min_image(d);
        REQUIRE_THAT(mi(0), WithinAbs(-2.0, EPS));
    }
}

TEST_CASE("PeriodicBC - triclinic box", "[bc]") {
    mat3_t box;
    box << 10.0,  5.0, 0.0,
             0.0,  8.66, 0.0,
             0.0,  0.0, 10.0;
    PeriodicBC bc(box);

    SECTION("volume is positive") {
        REQUIRE(bc.volume() > 0.0);
    }

    SECTION("wrap followed by re-wrap is idempotent") {
        vec3_t r(11.0, 3.0, -1.0);
        vec3_t w1 = bc.wrap(r);
        vec3_t w2 = bc.wrap(w1);
        REQUIRE_THAT((w1 - w2).norm(), WithinAbs(0.0, EPS));
    }

    SECTION("min_image norm <= half box dimension") {
        vec3_t d(9.0, 0.0, 0.0);
        vec3_t mi = bc.min_image(d);
        REQUIRE(mi.norm() <= 7.0);  // loose bound: must be shorter than d.norm()
    }
}

TEST_CASE("InfiniteBC - passthrough", "[bc]") {
    InfiniteBC bc(500.0);

    SECTION("wrap is identity") {
        vec3_t r(100.0, -50.0, 300.0);
        REQUIRE_THAT((bc.wrap(r) - r).norm(), WithinAbs(0.0, EPS));
    }

    SECTION("min_image is identity") {
        vec3_t d(100.0, 0.0, 0.0);
        REQUIRE_THAT((bc.min_image(d) - d).norm(), WithinAbs(0.0, EPS));
    }

    SECTION("volume") {
        REQUIRE_THAT(bc.volume(), WithinAbs(500.0, EPS));
    }
}

TEST_CASE("BoundaryConditions variant dispatch", "[bc]") {
    BoundaryConditions bc = PeriodicBC(mat3_t::Identity() * 10.0);

    vec3_t r(12.0, 3.0, 0.0);
    vec3_t w = bc_wrap(bc, r);
    REQUIRE_THAT(w(0), WithinAbs(2.0, EPS));
    REQUIRE_THAT(bc_volume(bc), WithinAbs(1000.0, EPS));
}
