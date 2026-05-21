#include <RMC/selectors/OrderedSelector.hpp>
#include <RMC/selectors/RandomSelector.hpp>
#include <RMC/selectors/RecursiveGroupSelector.hpp>
#include <RMC/selectors/SmartRandomSelector.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <unordered_map>

using namespace RMC;

TEST_CASE("RandomSelector - always returns valid index", "[selectors]") {
  GroupSelector sel = RandomSelector{42};
  for (int i = 0; i < 1000; ++i) {
    std::size_t idx = sel.select(10);
    REQUIRE(idx < 10);
  }
}

TEST_CASE("OrderedSelector - cycles deterministically", "[selectors]") {
  GroupSelector sel = OrderedSelector{};
  for (std::size_t round = 0; round < 3; ++round)
    for (std::size_t i = 0; i < 5; ++i)
      REQUIRE(sel.select(5) == i);
}

TEST_CASE("WeightedRandomSelector - highly-weighted group selected more often",
          "[selectors]") {
  GroupSelector sel = WeightedRandomSelector{{1.0, 100.0, 1.0}, /*seed=*/42};
  std::unordered_map<std::size_t, int> counts;
  for (int i = 0; i < 10000; ++i)
    ++counts[sel.select(3)];

  // Group 1 (weight 100) should be selected >90% of the time.
  double frac = static_cast<double>(counts[1]) / 10000.0;
  REQUIRE(frac > 0.90);
}

TEST_CASE("SmartRandomSelector - accepted feedback increases weight",
          "[selectors]") {
  GroupSelector sel = SmartRandomSelector{2.0, 42};
  sel.select(3); // triggers lazy initialise(3)

  for (int i = 0; i < 20; ++i)
    sel.feedback(0, true);

  std::unordered_map<std::size_t, int> counts;
  for (int i = 0; i < 1000; ++i)
    ++counts[sel.select(3)];
  REQUIRE(counts[0] > 700);
}

TEST_CASE("SmartRandomSelector - rejection decreases weight", "[selectors]") {
  GroupSelector sel = SmartRandomSelector{2.0, 42};
  sel.select(3); // triggers lazy initialise(3)

  for (int i = 0; i < 20; ++i)
    sel.feedback(0, false);

  std::unordered_map<std::size_t, int> counts;
  for (int i = 0; i < 1000; ++i)
    ++counts[sel.select(3)];
  REQUIRE(counts[0] < 300);
}

// ---- RecursiveGroupSelector ----
TEST_CASE("RecursiveGroupSelector Refine - retries same group after acceptance",
          "[selectors]") {
  // Inner selector always picks group 0 first via OrderedSelector.
  // After accepting, the same group should be returned for max_retries more
  // steps.
  GroupSelector sel = RecursiveGroupSelector{
      OrderedSelector{}, RecursiveMode::Refine, /*max_retries=*/3};

  std::size_t first = sel.select(5); // picks 0 from OrderedSelector
  sel.feedback(first, /*accepted=*/true);

  // Next 3 selects should return the same group.
  for (int i = 0; i < 3; ++i)
    REQUIRE(sel.select(5) == first);

  // After retries exhausted, delegates back to inner selector (next group = 1).
  std::size_t after = sel.select(5);
  REQUIRE(after != first);
}

TEST_CASE("RecursiveGroupSelector Refine - rejection cancels retry",
          "[selectors]") {
  GroupSelector sel = RecursiveGroupSelector{
      OrderedSelector{}, RecursiveMode::Refine, /*max_retries=*/5};

  std::size_t first = sel.select(5);
  sel.feedback(first, /*accepted=*/true); // start retry lock

  std::size_t second = sel.select(5);
  REQUIRE(second == first); // still locked

  sel.feedback(second, /*accepted=*/false); // rejection cancels lock
  std::size_t third = sel.select(5);
  REQUIRE(third != first); // inner selector now in control
}

TEST_CASE("RecursiveGroupSelector Explore - retries same group after rejection",
          "[selectors]") {
  GroupSelector sel = RecursiveGroupSelector{
      OrderedSelector{}, RecursiveMode::Explore, /*max_retries=*/3};

  std::size_t first = sel.select(5);
  sel.feedback(first, /*accepted=*/false); // rejected → start retry lock

  for (int i = 0; i < 3; ++i)
    REQUIRE(sel.select(5) == first);

  // After retries exhausted, moves to next group.
  REQUIRE(sel.select(5) != first);
}

TEST_CASE("RecursiveGroupSelector Explore - acceptance cancels retry",
          "[selectors]") {
  GroupSelector sel = RecursiveGroupSelector{
      OrderedSelector{}, RecursiveMode::Explore, /*max_retries=*/5};

  std::size_t first = sel.select(5);
  sel.feedback(first, /*accepted=*/false); // start retry lock

  std::size_t second = sel.select(5);
  REQUIRE(second == first);

  sel.feedback(second, /*accepted=*/true); // acceptance cancels lock
  REQUIRE(sel.select(5) != first);
}
