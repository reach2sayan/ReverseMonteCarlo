#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <fullrmc/selectors/RandomSelector.hpp>
#include <fullrmc/selectors/OrderedSelector.hpp>
#include <fullrmc/selectors/SmartRandomSelector.hpp>
#include <unordered_map>
#include <cstddef>

using namespace fullrmc;
using Catch::Matchers::WithinAbs;

TEST_CASE("RandomSelector - always returns valid index", "[selectors]") {
    RandomSelector sel(42);
    for (int i = 0; i < 1000; ++i) {
        std::size_t idx = sel.select(10);
        REQUIRE(idx < 10);
    }
}

TEST_CASE("OrderedSelector - cycles deterministically", "[selectors]") {
    OrderedSelector sel;
    for (std::size_t round = 0; round < 3; ++round)
        for (std::size_t i = 0; i < 5; ++i)
            REQUIRE(sel.select(5) == i);
}

TEST_CASE("WeightedRandomSelector - highly-weighted group selected more often", "[selectors]") {
    WeightedRandomSelector sel({1.0, 100.0, 1.0}, /*seed=*/42);
    std::unordered_map<std::size_t, int> counts;
    for (int i = 0; i < 10000; ++i)
        ++counts[sel.select(3)];

    // Group 1 (weight 100) should be selected >90% of the time.
    double frac = static_cast<double>(counts[1]) / 10000.0;
    REQUIRE(frac > 0.90);
}

TEST_CASE("SmartRandomSelector - accepted feedback increases weight", "[selectors]") {
    SmartRandomSelector sel(2.0, 42);
    sel.initialise(3);

    // Positive feedback on group 0 many times.
    for (int i = 0; i < 20; ++i) sel.feedback(0, true);

    // Group 0 weight should now dominate.
    std::unordered_map<std::size_t, int> counts;
    for (int i = 0; i < 1000; ++i) ++counts[sel.select(3)];
    REQUIRE(counts[0] > 700);
}

TEST_CASE("SmartRandomSelector - rejection decreases weight", "[selectors]") {
    SmartRandomSelector sel(2.0, 42);
    sel.initialise(3);

    for (int i = 0; i < 20; ++i) sel.feedback(0, false);

    // Group 0 weight should be suppressed.
    std::unordered_map<std::size_t, int> counts;
    for (int i = 0; i < 1000; ++i) ++counts[sel.select(3)];
    REQUIRE(counts[0] < 300);
}
