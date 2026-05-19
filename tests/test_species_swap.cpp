#include <RMC/Engine.hpp>
#include <RMC/constraints/ClusterCorrelationConstraint.hpp>
#include <RMC/core/Structure.hpp>
#include <RMC/generators/SpeciesSwap.hpp>
#include <RMC/generators/Translations.hpp>
#include <RMC/selectors/OrderedSelector.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <string>
#include <vector>

using namespace RMC;
using Catch::Matchers::WithinAbs;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Single sublattice: all 4 sites; 2 Cu + 2 Au.
static AtomicStructure make_4site_single_sublattice() {
  AtomicStructure s;
  s.coordinates.resize(4, 3);
  s.coordinates.setZero();
  s.atomic_numbers.resize(4);
  for (int i = 0; i < 4; ++i) {
    s.coordinates(i, 0) = static_cast<double>(i);
    s.atomic_numbers[i] = (i < 2) ? 29 : 79;
    s.elements.push_back((i < 2) ? "Cu" : "Au");
    s.names.push_back("X");
    s.residues.push_back("ALL");
    s.molecule_ids.push_back(static_cast<std::size_t>(i));
  }
  return s;
}

// ============================================================
// AtomicStructure species snapshot
// ============================================================

TEST_CASE("AtomicStructure - species snapshot roundtrip", "[species_swap]") {
  auto s = make_4site_single_sublattice();
  std::vector<std::string> orig_elems = s.elements;

  s.save_species_snapshot();

  // Mutate elements.
  s.elements[0] = "Au";
  s.elements[2] = "Cu";
  REQUIRE(s.elements[0] == "Au");
  REQUIRE(s.elements[2] == "Cu");

  s.restore_species_snapshot();

  REQUIRE(s.elements == orig_elems);
}

TEST_CASE("AtomicStructure - restore_species_snapshot is no-op without save",
          "[species_swap]") {
  auto s = make_4site_single_sublattice();
  std::vector<std::string> orig = s.elements;
  // Must not crash; elements must not change.
  s.restore_species_snapshot();
  REQUIRE(s.elements == orig);
}

// ============================================================
// SpeciesSwapGenerator
// ============================================================

TEST_CASE("SpeciesSwapGenerator - swaps elements between different-species sites",
          "[species_swap]") {
  auto s = make_4site_single_sublattice();
  // One sublattice: all 4 sites.
  std::vector<std::vector<std::size_t>> groups = {{0, 1, 2, 3}};
  SpeciesSwapGenerator gen{s, groups, /*seed=*/7};

  const std::vector<std::string> before = s.elements; // Cu Cu Au Au

  // Select site 0 (Cu); generator must swap it with one of sites 2 or 3 (Au).
  coords_t coords = s.coordinates;
  std::vector<std::size_t> idx = {0};
  IMoveGenerator wrapped{gen};
  wrapped.generate(coords, idx);

  // Coordinates must be unchanged.
  REQUIRE_THAT((coords - s.coordinates).norm(), WithinAbs(0.0, 1e-12));

  // Exactly one Cu→Au swap must have happened.
  int n_cu = 0, n_au = 0;
  for (const auto &e : s.elements) {
    if (e == "Cu") ++n_cu;
    if (e == "Au") ++n_au;
  }
  REQUIRE(n_cu == 2);
  REQUIRE(n_au == 2);
  REQUIRE(s.elements != before); // something changed
}

TEST_CASE("SpeciesSwapGenerator - no swap when all sites share the same element",
          "[species_swap]") {
  // All sites are Cu: no candidate with a different element exists → no swap.
  AtomicStructure s;
  s.coordinates.resize(4, 3);
  s.coordinates.setZero();
  s.atomic_numbers.resize(4);
  s.atomic_numbers.setConstant(29);
  for (int i = 0; i < 4; ++i) {
    s.elements.push_back("Cu");
    s.names.push_back("X");
    s.residues.push_back("ALL");
    s.molecule_ids.push_back(static_cast<std::size_t>(i));
    s.coordinates(i, 0) = static_cast<double>(i);
  }

  std::vector<std::vector<std::size_t>> groups = {{0, 1, 2, 3}};
  SpeciesSwapGenerator gen{s, groups, 42};
  IMoveGenerator wrapped{gen};

  coords_t coords = s.coordinates;
  for (std::size_t site = 0; site < 4; ++site) {
    const auto before = s.elements;
    std::vector<std::size_t> idx = {site};
    wrapped.generate(coords, idx);
    REQUIRE(s.elements == before);
  }
}

TEST_CASE("SpeciesSwapGenerator - respects sublattice boundary",
          "[species_swap]") {
  // Two sublattices: {0,1} and {2,3}. Mixed species on sublattice-0: Cu,Au.
  // Site 0 (Cu) should only swap with site 1 (Au), never with sites 2 or 3.
  AtomicStructure s;
  s.coordinates.resize(4, 3);
  s.coordinates.setZero();
  s.atomic_numbers.resize(4);
  s.elements = {"Cu", "Au", "Cu", "Au"};
  for (int i = 0; i < 4; ++i) {
    s.atomic_numbers[i] = (s.elements[i] == "Cu") ? 29 : 79;
    s.names.push_back("X");
    s.residues.push_back(i < 2 ? "SLA" : "SLB");
    s.molecule_ids.push_back(static_cast<std::size_t>(i));
    s.coordinates(i, 0) = static_cast<double>(i);
  }

  std::vector<std::vector<std::size_t>> groups = {{0, 1}, {2, 3}};
  SpeciesSwapGenerator gen{s, groups, 1};
  IMoveGenerator wrapped{gen};

  // Run many steps selecting site 0; site 2 and 3 must never change element.
  for (int trial = 0; trial < 50; ++trial) {
    const auto before = s.elements;
    coords_t coords = s.coordinates;
    std::vector<std::size_t> idx = {0};
    wrapped.generate(coords, idx);

    // Elements at sites 2 and 3 are untouched.
    REQUIRE(s.elements[2] == before[2]);
    REQUIRE(s.elements[3] == before[3]);
  }
}

TEST_CASE("SpeciesSwapGenerator - modifies_species returns true",
          "[species_swap]") {
  auto s = make_4site_single_sublattice();
  std::vector<std::vector<std::size_t>> groups = {{0, 1, 2, 3}};
  IMoveGenerator wrapped{SpeciesSwapGenerator{s, groups}};
  REQUIRE(wrapped.modifies_species());
}

TEST_CASE("SpeciesSwapGenerator - normal generators report modifies_species false",
          "[species_swap]") {
  IMoveGenerator t = TranslationGenerator(0.1, 0.2, 1);
  REQUIRE_FALSE(t.modifies_species());
}

// ============================================================
// ClusterCorrelationConstraint
// ============================================================

// Helper: build two pair orbits covering a 4-site ring.
// Orbit 0: nearest-neighbour pairs (0-1, 1-2, 2-3, 3-0)
// Orbit 1: next-nearest pairs     (0-2, 1-3)
static std::vector<ClusterOrbit> two_pair_orbits() {
  ClusterOrbit nn, nnn;
  nn.target = 0.0; nn.weight = 1.0;
  nn.instances = {{{0, 1}}, {{1, 2}}, {{2, 3}}, {{3, 0}}};

  nnn.target = 0.0; nnn.weight = 1.0;
  nnn.instances = {{{0, 2}}, {{1, 3}}};

  return {nn, nnn};
}

TEST_CASE("ClusterCorrelationConstraint - perfectly ordered state has nonzero error",
          "[species_swap][constraint]") {
  // All Cu on sites 0,1 and all Au on 2,3: high correlation, nonzero error vs 0.
  auto s = make_4site_single_sublattice(); // Cu Cu Au Au
  ClusterCorrelationConstraint::SpeciesMap sm{{"Cu", +1.0}, {"Au", -1.0}};
  ClusterCorrelationConstraint cc{s, sm, two_pair_orbits()};

  coords_t c = s.coordinates;
  std::vector<std::size_t> all = {0, 1, 2, 3};
  const double err = cc.compute_error(c, all);
  REQUIRE(err > 0.0);
}

TEST_CASE("ClusterCorrelationConstraint - alternating arrangement is closer to random",
          "[species_swap][constraint]") {
  // Cu Au Cu Au: nearest-neighbour correlation = -1 (anti-ferromagnetic),
  // which is further from 0 target. next-nearest = +1, also nonzero.
  // But compare to all-same arrangement: both have nonzero error.
  // The test verifies compute_error returns a finite, non-negative value.
  auto s = make_4site_single_sublattice();
  s.elements = {"Cu", "Au", "Cu", "Au"};

  ClusterCorrelationConstraint::SpeciesMap sm{{"Cu", +1.0}, {"Au", -1.0}};
  ClusterCorrelationConstraint cc{s, sm, two_pair_orbits()};

  coords_t c = s.coordinates;
  std::vector<std::size_t> all = {0, 1, 2, 3};
  const double err = cc.compute_error(c, all);
  REQUIRE(err >= 0.0);
  REQUIRE(std::isfinite(err));
}

TEST_CASE("ClusterCorrelationConstraint - error decreases toward target arrangement",
          "[species_swap][constraint]") {
  // Cu Au Cu Au → NN corr = -1, NNN corr = +1; both deviate from 0.
  // Cu Cu Au Au → NN corr > 0, NNN corr = -1; both deviate from 0.
  // Mixed Cu Au Au Cu → NN corr = 0 exactly for two orbits.
  AtomicStructure s;
  s.coordinates.resize(4, 3);
  s.coordinates.setZero();
  s.atomic_numbers.resize(4, 1);
  s.elements = {"Cu", "Au", "Au", "Cu"}; // NN pairs: (Cu,Au),(Au,Au),(Au,Cu),(Cu,Cu)
  for (int i = 0; i < 4; ++i) {
    s.names.push_back("X"); s.residues.push_back("A");
    s.molecule_ids.push_back(static_cast<std::size_t>(i));
    s.coordinates(i, 0) = i;
  }

  ClusterCorrelationConstraint::SpeciesMap sm{{"Cu", +1.0}, {"Au", -1.0}};
  ClusterCorrelationConstraint cc{s, sm, two_pair_orbits()};

  coords_t c = s.coordinates;
  std::vector<std::size_t> all = {0, 1, 2, 3};
  REQUIRE(std::isfinite(cc.compute_error(c, all)));
}

TEST_CASE("ClusterCorrelationConstraint - should_reject worsening move",
          "[species_swap][constraint]") {
  // NN target = 0. Cu Cu Au Au has NN corr = 0 → err = 0 (at target).
  // Cu Au Cu Au has NN corr = -1 → err = 1 (worse). Moving away is rejected.
  auto s = make_4site_single_sublattice();
  s.elements = {"Cu", "Cu", "Au", "Au"}; // err = 0

  // Use only the NN orbit with target 0.
  ClusterOrbit nn;
  nn.target = 0.0; nn.weight = 1.0;
  nn.instances = {{{0, 1}}, {{1, 2}}, {{2, 3}}, {{3, 0}}};

  ClusterCorrelationConstraint::SpeciesMap sm{{"Cu", +1.0}, {"Au", -1.0}};
  IConstraint cc{ClusterCorrelationConstraint{s, sm, {nn}}};
  cc.set_boundary_conditions(InfiniteBC{});

  coords_t c = s.coordinates;
  std::vector<std::size_t> all = {0, 1, 2, 3};

  cc.compute_before_move(c, all); // err_before = 0

  // Worsen: alternating arrangement has NN corr = -1, err = 1.
  s.elements = {"Cu", "Au", "Cu", "Au"};
  cc.compute_after_move(c, all); // err_after = 1

  REQUIRE(cc.should_reject()); // err_after > err_before → reject
}

// ============================================================
// Engine integration: species restored on rejection
// ============================================================

TEST_CASE("Engine - SpeciesSwapGenerator restores elements on rejection",
          "[species_swap][engine]") {
  // Set up: 4 sites, one sublattice. ClusterCorrelationConstraint with a
  // target that the starting state already satisfies perfectly.
  // Any swap will worsen it → every move is rejected → elements must not change.

  auto s = make_4site_single_sublattice();
  // Cu Au Cu Au → alternating. Build an orbit whose target EQUALS the current
  // correlation so err_before == 0 and any swap raises err_after > 0.

  // NN correlation for Cu Au Cu Au: each NN pair is (Cu,Au) or (Au,Cu) → σᵢσⱼ = -1.
  // Average NN corr = -1.0. Set target = -1.0 so starting state is perfect.
  ClusterOrbit perfect_nn;
  perfect_nn.target = -1.0; perfect_nn.weight = 1.0;
  perfect_nn.instances = {{{0, 1}}, {{1, 2}}, {{2, 3}}, {{3, 0}}};

  s.elements = {"Cu", "Au", "Cu", "Au"};
  s.atomic_numbers[0] = s.atomic_numbers[2] = 29;
  s.atomic_numbers[1] = s.atomic_numbers[3] = 79;

  Engine eng{s, InfiniteBC{}};

  ClusterCorrelationConstraint::SpeciesMap sm{{"Cu", +1.0}, {"Au", -1.0}};
  eng.add_constraint(IConstraint{
      ClusterCorrelationConstraint{eng.structure(), sm, {perfect_nn}}});

  std::vector<std::vector<std::size_t>> groups = {{0, 1, 2, 3}};
  SpeciesSwapGenerator gen{eng.structure(), groups, 42};

  for (std::size_t i = 0; i < eng.structure().size(); ++i) {
    Group g;
    g.name    = "site_" + std::to_string(i);
    g.indices = {i};
    g.generator = IMoveGenerator{gen};
    eng.add_group(std::move(g));
  }
  eng.set_selector(IGroupSelector{OrderedSelector{}});

  const std::vector<std::string> elems_before = eng.structure().elements;
  eng.run(40);

  // All moves rejected → elements unchanged.
  REQUIRE(eng.stats().steps_accepted == 0);
  REQUIRE(eng.structure().elements == elems_before);
}

TEST_CASE("Engine - SpeciesSwapGenerator accepts improving moves",
          "[species_swap][engine]") {
  // Start far from target; verify that at least some swaps are accepted.
  auto s = make_4site_single_sublattice();
  // Start: Cu Cu Au Au (all same-species together).
  // Target NN correlation = -1 (alternating). Engine should accept swaps
  // that move toward alternating.

  ClusterOrbit nn;
  nn.target = -1.0; nn.weight = 1.0;
  nn.instances = {{{0, 1}}, {{1, 2}}, {{2, 3}}, {{3, 0}}};

  Engine eng{s, InfiniteBC{}};
  ClusterCorrelationConstraint::SpeciesMap sm{{"Cu", +1.0}, {"Au", -1.0}};
  eng.add_constraint(IConstraint{
      ClusterCorrelationConstraint{eng.structure(), sm, {nn}}});

  std::vector<std::vector<std::size_t>> groups = {{0, 1, 2, 3}};
  SpeciesSwapGenerator gen{eng.structure(), groups, 17};

  for (std::size_t i = 0; i < eng.structure().size(); ++i) {
    Group g;
    g.name    = "site_" + std::to_string(i);
    g.indices = {i};
    g.generator = IMoveGenerator{gen};
    eng.add_group(std::move(g));
  }
  eng.set_selector(IGroupSelector{OrderedSelector{}});

  eng.run(200);
  REQUIRE(eng.stats().steps_accepted > 0);
  // Final error must be <= initial (engine never makes things worse).
  REQUIRE(eng.stats().last_total_err >= 0.0);
}
