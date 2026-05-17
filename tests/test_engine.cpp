#include <RMC/Engine.hpp>
#include <RMC/constraints/BondConstraint.hpp>
#include <RMC/constraints/DistanceConstraint.hpp>
#include <RMC/selectors/OrderedSelector.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <numeric>

using namespace RMC;
using Catch::Matchers::WithinAbs;

static AtomicStructure make_linear_chain(int N, double spacing) {
  AtomicStructure s;
  s.coordinates.resize(N, 3);
  s.coordinates.setZero();
  s.atomic_numbers.resize(N);
  s.atomic_numbers.setOnes();
  for (int i = 0; i < N; ++i) {
    s.coordinates(i, 0) = i * spacing;
    s.names.push_back("Ar");
    s.elements.push_back("Ar");
    s.residues.push_back("ARG");
    s.molecule_ids.push_back(i);
  }
  return s;
}

TEST_CASE("Engine - constructs and runs without crash", "[engine]") {
  AtomicStructure s = make_linear_chain(5, 3.0);
  InfiniteBC bc(1000.0);
  Engine engine(std::move(s), bc);
  engine.build_atomic_groups(0.0, 0.1, 42);
  engine.run(100);

  auto st = engine.stats();
  REQUIRE(st.steps_total == 100);
  REQUIRE(st.steps_tried <= 100);
}

TEST_CASE("Engine - acceptance rate > 0 for unconstrained system", "[engine]") {
  AtomicStructure s = make_linear_chain(10, 3.0);
  Engine engine(std::move(s), InfiniteBC(1e6));
  engine.build_atomic_groups(0.05, 0.05, 1);
  engine.run(500);

  auto st = engine.stats();
  // No constraints → every move should be accepted.
  REQUIRE(st.steps_accepted == st.steps_tried);
}

TEST_CASE("Engine - hard distance constraint limits proximity", "[engine]") {
  // 3 atoms; add a pairwise minimum-distance constraint of 2 Å.
  AtomicStructure s = make_linear_chain(3, 4.0); // initial spacing 4 Å
  Engine engine(std::move(s), InfiniteBC(1e6));
  engine.build_atomic_groups(0.0, 0.5, 99);

  InterMolecularDistanceConstraint c;
  c.set_minimum_distance("Ar", "Ar", 2.0);
  c.set_structure(engine.structure().elements,
                  engine.structure().molecule_ids);
  engine.add_constraint(std::move(c));

  engine.run(2000);

  // Verify no pair is closer than 2 Å after the run.
  const coords_t &coords = engine.structure().coordinates;
  const int N = static_cast<int>(coords.rows());
  for (int i = 0; i < N; ++i)
    for (int j = i + 1; j < N; ++j) {
      double d = (coords.row(i) - coords.row(j)).norm();
      REQUIRE(d >= 2.0 - 1e-3);
    }
}

TEST_CASE("Engine - snapshot restore after rejection", "[engine]") {
  // Use BondConstraint with lo==hi==initial_distance so err_before==0 and
  // any non-trivial displacement raises err_after > 0, guaranteeing rejection.
  AtomicStructure s = make_linear_chain(2, 1.0); // atoms 1 Å apart
  Engine engine(std::move(s), InfiniteBC(1e6));
  engine.build_atomic_groups(0.5, 0.5, 7); // fixed 0.5 Å step

  BondConstraint c;
  c.add_bond(0, 1, 1.0, 1.0); // exact bond → any move worsens error
  engine.add_constraint(std::move(c));

  coords_t before = engine.structure().coordinates;
  engine.run(100);

  const coords_t &after = engine.structure().coordinates;
  REQUIRE_THAT((after - before).norm(), WithinAbs(0.0, 1e-10));
  REQUIRE(engine.stats().steps_accepted == 0);
}

TEST_CASE("Engine - PBC wrapping keeps atoms in box", "[engine]") {
  AtomicStructure s = make_linear_chain(4, 3.0);
  mat3_t box = mat3_t::Identity() * 12.0;
  Engine engine(std::move(s), PeriodicBC(box));
  engine.build_atomic_groups(0.0, 0.5, 55);
  engine.run(500);

  const coords_t &coords = engine.structure().coordinates;
  for (Eigen::Index i = 0; i < coords.rows(); ++i) {
    REQUIRE(coords(i, 0) >= 0.0 - 1e-6);
    REQUIRE(coords(i, 0) < 12.0 + 1e-6);
  }
}
