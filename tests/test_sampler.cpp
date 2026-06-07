#include <RMC/Engine.hpp>
#include <RMC/constraints/ClusterCorrelationConstraint.hpp>
#include <RMC/core/Structure.hpp>
#include <RMC/generators/SpeciesSwap.hpp>
#include <RMC/sampling/AnnealingSampler.hpp>
#include <RMC/sampling/GreedySampler.hpp>
#include <RMC/sampling/MetropolisSampler.hpp>
#include <RMC/sampling/Sampler.hpp>
#include <RMC/selectors/OrderedSelector.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <string>
#include <vector>

using namespace RMC;
using Catch::Matchers::WithinAbs;

// ============================================================
// GreedySampler
// ============================================================

TEST_CASE("GreedySampler - accepts downhill and equal, rejects uphill",
          "[sampler]") {
  Sampler s{GreedySampler{}}; // tolerance 0
  REQUIRE(s.accept(/*before=*/1.0, /*after=*/0.5, 0, 0.99)); // downhill
  REQUIRE(s.accept(1.0, 1.0, 0, 0.99));                      // equal
  REQUIRE_FALSE(s.accept(1.0, 1.5, 0, 0.0));                 // uphill
}

TEST_CASE("GreedySampler - tolerance admits small uphill moves", "[sampler]") {
  Sampler s{GreedySampler{0.6}};
  REQUIRE(s.accept(1.0, 1.5, 0, 0.0));       // ΔE=0.5 ≤ tol 0.6 → accept
  REQUIRE(s.accept(1.0, 1.6, 0, 0.0));       // ΔE=0.6 == tol → accept
  REQUIRE_FALSE(s.accept(1.0, 1.7, 0, 0.0)); // ΔE=0.7 > tol → reject
}

// ============================================================
// MetropolisSampler
// ============================================================

TEST_CASE("MetropolisSampler - always accepts non-uphill moves", "[sampler]") {
  Sampler s{MetropolisSampler{1.0}};
  REQUIRE(s.accept(1.0, 0.5, 0, 0.999)); // ΔE<0 regardless of u01
  REQUIRE(s.accept(1.0, 1.0, 0, 0.999)); // ΔE=0
}

TEST_CASE("MetropolisSampler - uphill obeys exp(-dE/T) threshold",
          "[sampler]") {
  Sampler s{MetropolisSampler{1.0}};
  const double dE = 1.0;
  const double thr = std::exp(-dE / 1.0); // ≈ 0.3679
  REQUIRE(s.accept(0.0, dE, 0, thr - 0.05)); // u01 below threshold → accept
  REQUIRE_FALSE(s.accept(0.0, dE, 0, thr + 0.05)); // above → reject
}

TEST_CASE("MetropolisSampler - T<=0 degenerates to greedy", "[sampler]") {
  Sampler s{MetropolisSampler{0.0}};
  REQUIRE(s.accept(1.0, 0.5, 0, 0.0));       // downhill ok
  REQUIRE_FALSE(s.accept(0.0, 1.0, 0, 0.0)); // any uphill rejected
}

TEST_CASE("MetropolisSampler - higher T accepts more uphill moves",
          "[sampler]") {
  const double dE = 1.0;
  const double u = 0.5;
  Sampler cold{MetropolisSampler{0.5}}; // exp(-2)=0.135 < 0.5 → reject
  Sampler hot{MetropolisSampler{100.0}}; // exp(-0.01)≈0.99 > 0.5 → accept
  REQUIRE_FALSE(cold.accept(0.0, dE, 0, u));
  REQUIRE(hot.accept(0.0, dE, 0, u));
}

// ============================================================
// AnnealingSampler
// ============================================================

TEST_CASE("AnnealingSampler - geometric cooling schedule", "[sampler]") {
  AnnealingSampler a{
      AnnealingSampler::Schedule{.t0 = 2.0, .cooling = 0.5, .interval = 10}};
  REQUIRE_THAT(a.temperature(0), WithinAbs(2.0, 1e-12));    // 2·0.5^0
  REQUIRE_THAT(a.temperature(9), WithinAbs(2.0, 1e-12));    // still interval 0
  REQUIRE_THAT(a.temperature(10), WithinAbs(1.0, 1e-12));   // 2·0.5^1
  REQUIRE_THAT(a.temperature(20), WithinAbs(0.5, 1e-12));   // 2·0.5^2
  // Monotonically non-increasing.
  REQUIRE(a.temperature(0) >= a.temperature(10));
  REQUIRE(a.temperature(10) >= a.temperature(100));
  // Floored at t_min.
  REQUIRE(a.temperature(1'000'000) >= 1e-6);
}

TEST_CASE("AnnealingSampler - cooling makes uphill acceptance harder over time",
          "[sampler]") {
  Sampler s{AnnealingSampler{
      AnnealingSampler::Schedule{.t0 = 2.0, .cooling = 0.5, .interval = 10}}};
  const double dE = 1.0;
  const double u = 0.3;
  // Early (T=2): exp(-0.5)=0.606 > 0.3 → accept.
  REQUIRE(s.accept(0.0, dE, 0, u));
  // Late (T≈0.002): exp(-dE/T)≈0 < 0.3 → reject.
  REQUIRE_FALSE(s.accept(0.0, dE, 100, u));
}

// ============================================================
// Engine integration: the sampler actually changes acceptance
// ============================================================

static AtomicStructure make_4site_alternating() {
  AtomicStructure s;
  s.coordinates.resize(4, 3);
  s.coordinates.setZero();
  s.atomic_numbers.resize(4);
  s.elements = {"Cu", "Au", "Cu", "Au"};
  for (int i = 0; i < 4; ++i) {
    s.atomic_numbers[i] = (s.elements[i] == "Cu") ? 29 : 79;
    s.names.push_back("X");
    s.residues.push_back("ALL");
    s.molecule_ids.push_back(static_cast<std::size_t>(i));
    s.coordinates(i, 0) = static_cast<double>(i);
  }
  return s;
}

// Populate an engine whose starting state is exactly at the orbit target, so
// every species swap is strictly uphill (ΔE > 0). Greedy must reject all; a
// high-T Metropolis sampler must accept some. `eng` is taken by reference and
// never moved afterwards — the constraint holds a reference into its structure.
static void setup_uphill_engine(Engine &eng) {
  ClusterOrbit perfect_nn;
  perfect_nn.target = -1.0; // start already perfect (NN correlation = -1)
  perfect_nn.weight = 1.0;
  perfect_nn.set_instances({{0, 1}, {1, 2}, {2, 3}, {3, 0}});

  ClusterCorrelationConstraint::SpeciesMap sm{{"Cu", +1.0}, {"Au", -1.0}};
  eng.add_constraint(Constraint{
      ClusterCorrelationConstraint{eng.structure(), sm, {perfect_nn}}});

  std::vector<std::vector<std::size_t>> groups = {{0, 1, 2, 3}};
  SpeciesSwapGenerator gen{eng.structure(), groups, 42};
  for (std::size_t i = 0; i < eng.structure().size(); ++i) {
    Group g;
    g.name = "site_" + std::to_string(i);
    g.indices = {i};
    g.generator = MoveGenerator{gen};
    eng.add_group(std::move(g));
  }
  eng.set_selector(GroupSelector{OrderedSelector{}});
}

TEST_CASE("Engine - GreedySampler rejects all uphill swaps",
          "[sampler][engine]") {
  Engine eng{make_4site_alternating(), InfiniteBC{}};
  setup_uphill_engine(eng);
  eng.set_sampler(Sampler{GreedySampler{}}, 7);
  eng.run(40);
  REQUIRE(eng.stats().steps_accepted == 0);
}

TEST_CASE("Engine - high-T Metropolis accepts uphill swaps greedy rejects",
          "[sampler][engine]") {
  Engine eng{make_4site_alternating(), InfiniteBC{}};
  setup_uphill_engine(eng);
  // T huge ⇒ exp(-ΔE/T) ≈ 1 > any u01 ∈ [0,1) ⇒ accept nearly all uphill moves.
  eng.set_sampler(Sampler{MetropolisSampler{1e9}}, 7);
  eng.run(40);
  REQUIRE(eng.stats().steps_accepted > 0);
}
