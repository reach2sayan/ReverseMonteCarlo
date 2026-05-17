#include <RMC/selectors/OrderedSelector.hpp>
#include <RMC/selectors/RandomSelector.hpp>
#include <RMC/selectors/SmartRandomSelector.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <unordered_map>

using namespace RMC;

TEST_CASE("RandomSelector - always returns valid index", "[selectors]") {
  IGroupSelector sel = RandomSelector{42};
  for (int i = 0; i < 1000; ++i) {
    std::size_t idx = sel.select(10);
    REQUIRE(idx < 10);
  }
}

TEST_CASE("OrderedSelector - cycles deterministically", "[selectors]") {
  IGroupSelector sel = OrderedSelector{};
  for (std::size_t round = 0; round < 3; ++round)
    for (std::size_t i = 0; i < 5; ++i)
      REQUIRE(sel.select(5) == i);
}

TEST_CASE("WeightedRandomSelector - highly-weighted group selected more often",
          "[selectors]") {
  IGroupSelector sel = WeightedRandomSelector{{1.0, 100.0, 1.0}, /*seed=*/42};
  std::unordered_map<std::size_t, int> counts;
  for (int i = 0; i < 10000; ++i)
    ++counts[sel.select(3)];

  // Group 1 (weight 100) should be selected >90% of the time.
  double frac = static_cast<double>(counts[1]) / 10000.0;
  REQUIRE(frac > 0.90);
}

TEST_CASE("SmartRandomSelector - accepted feedback increases weight",
          "[selectors]") {
  IGroupSelector sel = SmartRandomSelector{2.0, 42};
  sel.select(3); // triggers lazy initialise(3)

  for (int i = 0; i < 20; ++i)
    sel.feedback(0, true);

  std::unordered_map<std::size_t, int> counts;
  for (int i = 0; i < 1000; ++i)
    ++counts[sel.select(3)];
  REQUIRE(counts[0] > 700);
}

TEST_CASE("SmartRandomSelector - rejection decreases weight", "[selectors]") {
  IGroupSelector sel = SmartRandomSelector{2.0, 42};
  sel.select(3); // triggers lazy initialise(3)

  for (int i = 0; i < 20; ++i)
    sel.feedback(0, false);

  std::unordered_map<std::size_t, int> counts;
  for (int i = 0; i < 1000; ++i)
    ++counts[sel.select(3)];
  REQUIRE(counts[0] < 300);
}
