// Nine test fixtures mapped from the fullrmc small examples.
// Each fixture is self-contained: structures built inline, no PDB I/O.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <RMC/Engine.hpp>
#include <RMC/constraints/AngleConstraint.hpp>
#include <RMC/constraints/BondConstraint.hpp>
#include <RMC/constraints/CoordinationConstraint.hpp>
#include <RMC/constraints/DihedralAngleConstraint.hpp>
#include <RMC/constraints/DistanceConstraint.hpp>
#include <RMC/constraints/ImproperAngleConstraint.hpp>
#include <RMC/core/AtomsCollector.hpp>
#include <RMC/generators/Removes.hpp>
#include <RMC/generators/Rotations.hpp>
#include <RMC/generators/Translations.hpp>
#include <RMC/selectors/OrderedSelector.hpp>

#include <array>
#include <cmath>
#include <initializer_list>
#include <memory>
#include <numbers>
#include <vector>

using namespace RMC;
using Catch::Matchers::WithinAbs;

namespace {

AtomicStructure
make_structure(std::initializer_list<std::array<double, 3>> positions,
               std::initializer_list<const char *> elems,
               std::vector<std::size_t> mol_ids = {}) {
  AtomicStructure s;
  const auto N = static_cast<Eigen::Index>(positions.size());
  s.coordinates.resize(N, 3);
  s.atomic_numbers.resize(N);
  s.atomic_numbers.setOnes();

  Eigen::Index i = 0;
  for (auto &p : positions) {
    s.coordinates(i, 0) = p[0];
    s.coordinates(i, 1) = p[1];
    s.coordinates(i, 2) = p[2];
    ++i;
  }
  for (auto e : elems) {
    s.names.push_back(e);
    s.elements.push_back(e);
    s.residues.push_back("RES");
  }
  if (mol_ids.empty()) {
    for (std::size_t j = 0; j < static_cast<std::size_t>(N); ++j)
      s.molecule_ids.push_back(j);
  } else {
    s.molecule_ids = std::move(mol_ids);
  }
  return s;
}

// Helper: bond angle (radians) at vertex j for atoms i-j-k.
double bond_angle(const coords_t &c, int i, int j, int k) {
  vec3_t v1 = (c.row(i) - c.row(j)).transpose();
  vec3_t v2 = (c.row(k) - c.row(j)).transpose();
  double cos_a = v1.dot(v2) / (v1.norm() * v2.norm() + 1e-30);
  return std::acos(std::clamp(cos_a, -1.0, 1.0));
}

// Helper: improper dihedral angle (atan2 formula) for atoms i-j-k-l.
double improper_angle(const coords_t &c, int i, int j, int k, int l) {
  vec3_t b1 = c.row(j).transpose() - c.row(i).transpose();
  vec3_t b2 = c.row(k).transpose() - c.row(j).transpose();
  vec3_t b3 = c.row(l).transpose() - c.row(k).transpose();
  vec3_t n1 = b1.cross(b2);
  vec3_t n2 = b2.cross(b3);
  double x = n1.dot(n2);
  double y = n1.cross(n2).dot(b2.normalized());
  return std::atan2(y, x);
}

} // namespace

// ─── 1. translations
// ────────────────────────────────────────────────────────── fullrmc example:
// whole-molecule random translation, no constraints.
TEST_CASE("translations - molecule translates freely with 100% acceptance",
          "[example_translations]") {
  auto s =
      make_structure({{0, 0, 0}, {1, 0, 0}, {2, 0, 0}, {3, 0, 0}, {4, 0, 0}},
                     {"C", "C", "C", "C", "C"}, {0, 0, 0, 0, 0});
  coords_t initial = s.coordinates;

  Engine engine(std::move(s), InfiniteBC(1e6));
  Group g;
  g.name = "mol";
  g.indices = {0, 1, 2, 3, 4};
  g.generator.emplace(TranslationGenerator(0.05, 0.3, 42));
  engine.add_group(std::move(g));
  engine.set_selector(OrderedSelector{});

  engine.run(500);

  REQUIRE(engine.stats().steps_total == 500);
  REQUIRE(engine.stats().steps_accepted == engine.stats().steps_tried);
  REQUIRE((engine.structure().coordinates - initial).norm() > 0.01);
}

// ─── 2. rotations
// ───────────────────────────────────────────────────────────── fullrmc
// example: whole-molecule rotation — all pairwise distances preserved.
TEST_CASE("rotations - rigid-body rotation preserves all pairwise distances",
          "[example_rotations]") {
  auto s = make_structure({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
                          {"C", "C", "C", "C"}, {0, 0, 0, 0});

  Engine engine(std::move(s), InfiniteBC(1e6));
  Group g;
  g.name = "mol";
  g.indices = {0, 1, 2, 3};
  g.generator.emplace(RotationGenerator(0.05, 0.5, 42));
  engine.add_group(std::move(g));
  engine.set_selector(OrderedSelector{});

  const auto dist_ij = [&](int a, int b) {
    return (engine.structure().coordinates.row(a) -
            engine.structure().coordinates.row(b))
        .norm();
  };
  std::array<double, 6> d0;
  int idx = 0;
  for (int a = 0; a < 4; ++a)
    for (int b = a + 1; b < 4; ++b)
      d0[idx++] = dist_ij(a, b);

  engine.run(500);

  idx = 0;
  for (int a = 0; a < 4; ++a)
    for (int b = a + 1; b < 4; ++b)
      REQUIRE_THAT(dist_ij(a, b), WithinAbs(d0[idx++], 1e-6));
}

// ─── 3. agitations
// ──────────────────────────────────────────────────────────── fullrmc example:
// bond and angle constraints on a water-like molecule with per-atom move
// groups.  Initial geometry already satisfies both; constraints prevent any
// worsening move.
TEST_CASE(
    "agitations - bond and angle constraints maintained on water geometry",
    "[example_agitations]") {
  // O(0) H1(1) H2(2): O-H bonds = 1.0 Å in [0.8,1.1], H-O-H = 90° in [80°,120°]
  auto s = make_structure({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}, {"O", "H", "H"},
                          {0, 0, 0});

  Engine engine(std::move(s), InfiniteBC(1e6));

  BondConstraint bc;
  bc.add_bond(0, 1, 0.8, 1.1);
  bc.add_bond(0, 2, 0.8, 1.1);
  engine.add_constraint(std::move(bc));

  AngleConstraint ac;
  ac.add_angle(1, 0, 2, 80.0 * std::numbers::pi / 180.0,
               120.0 * std::numbers::pi / 180.0);
  engine.add_constraint(std::move(ac));

  engine.build_atomic_groups(0.0, 0.08, 42);
  engine.run(2000);

  const coords_t &c = engine.structure().coordinates;
  double d01 = (c.row(0) - c.row(1)).norm();
  double d02 = (c.row(0) - c.row(2)).norm();
  REQUIRE(d01 >= 0.8 - 1e-6);
  REQUIRE(d01 <= 1.1 + 1e-6);
  REQUIRE(d02 >= 0.8 - 1e-6);
  REQUIRE(d02 <= 1.1 + 1e-6);

  double angle = bond_angle(c, 1, 0, 2);
  REQUIRE(angle >= 80.0 * std::numbers::pi / 180.0 - 1e-6);
  REQUIRE(angle <= 120.0 * std::numbers::pi / 180.0 + 1e-6);
}

// ─── 4. bondsConstraint
// ─────────────────────────────────────────────────────── fullrmc example:
// BondConstraint holds structure near target bond lengths. Scenario A: normal
// target — bonds stay in range from the start. Scenario B: distorted target —
// engine moves atoms toward the new target.
TEST_CASE("bondsConstraint - normal bond target holds water bonds in range",
          "[example_bonds_constraint]") {
  auto s = make_structure({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}, {"O", "H", "H"},
                          {0, 0, 0});

  Engine engine(std::move(s), InfiniteBC(1e6));
  BondConstraint bc;
  bc.add_bond(0, 1, 0.8, 1.1);
  bc.add_bond(0, 2, 0.8, 1.1);
  engine.add_constraint(std::move(bc));
  engine.build_atomic_groups(0.0, 0.1, 7);

  engine.run(2000);

  const coords_t &c = engine.structure().coordinates;
  REQUIRE((c.row(0) - c.row(1)).norm() >= 0.8 - 1e-6);
  REQUIRE((c.row(0) - c.row(1)).norm() <= 1.1 + 1e-6);
  REQUIRE((c.row(0) - c.row(2)).norm() >= 0.8 - 1e-6);
  REQUIRE((c.row(0) - c.row(2)).norm() <= 1.1 + 1e-6);
}

TEST_CASE("bondsConstraint - distorted bond target produces accepted moves",
          "[example_bonds_constraint]") {
  // Initial bonds 1.0 Å; target [2.2,2.5] and [0.2,0.5] — engine must move
  // atoms.
  auto s = make_structure({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}, {"O", "H", "H"},
                          {0, 0, 0});

  Engine engine(std::move(s), InfiniteBC(1e6));
  BondConstraint bc;
  bc.add_bond(0, 1, 2.2, 2.5); // O-H1 should be elongated
  bc.add_bond(0, 2, 0.2, 0.5); // O-H2 should be compressed
  engine.add_constraint(std::move(bc));
  engine.build_atomic_groups(0.0, 0.1, 13);

  engine.run(2000);

  REQUIRE(engine.stats().steps_accepted > 0);
}

// ─── 5. anglesConstraint
// ────────────────────────────────────────────────────── fullrmc example:
// AngleConstraint steers the H-O-H angle.
TEST_CASE("anglesConstraint - normal angle target holds H-O-H in 80–120 deg",
          "[example_angles_constraint]") {
  auto s = make_structure({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}, {"O", "H", "H"},
                          {0, 0, 0});

  Engine engine(std::move(s), InfiniteBC(1e6));

  BondConstraint bc;
  bc.add_bond(0, 1, 0.8, 1.1);
  bc.add_bond(0, 2, 0.8, 1.1);
  engine.add_constraint(std::move(bc));

  AngleConstraint ac;
  ac.add_angle(1, 0, 2, 80.0 * std::numbers::pi / 180.0,
               120.0 * std::numbers::pi / 180.0);
  engine.add_constraint(std::move(ac));
  engine.build_atomic_groups(0.0, 0.08, 17);

  engine.run(2000);

  double angle = bond_angle(engine.structure().coordinates, 1, 0, 2);
  REQUIRE(angle >= 80.0 * std::numbers::pi / 180.0 - 1e-6);
  REQUIRE(angle <= 120.0 * std::numbers::pi / 180.0 + 1e-6);
}

TEST_CASE("anglesConstraint - squeezed angle target produces accepted moves",
          "[example_angles_constraint]") {
  // H-O-H starts at 90°; target [30°,40°] is violated → engine explores.
  auto s = make_structure({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}, {"O", "H", "H"},
                          {0, 0, 0});

  Engine engine(std::move(s), InfiniteBC(1e6));
  AngleConstraint ac;
  ac.add_angle(1, 0, 2, 30.0 * std::numbers::pi / 180.0,
               40.0 * std::numbers::pi / 180.0);
  engine.add_constraint(std::move(ac));
  engine.build_atomic_groups(0.0, 0.1, 23);

  engine.run(2000);

  REQUIRE(engine.stats().steps_accepted > 0);
}

// ─── 6. dihedralConstraint
// ──────────────────────────────────────────────────── fullrmc example:
// DihedralAngleConstraint drives a 4-atom chain between rotamer shells (like
// butane C1-C2-C3-C4).
TEST_CASE(
    "dihedralConstraint - permissive dihedral completes without rejection",
    "[example_dihedral_constraint]") {
  // Non-collinear chain so the dihedral is well-defined.
  auto s = make_structure({{0, 0, 0}, {1, 0, 0}, {2, 0, 1}, {3, 1, 1}},
                          {"C", "C", "C", "C"}, {0, 0, 0, 0});

  Engine engine(std::move(s), InfiniteBC(1e6));
  DihedralAngleConstraint dc;
  dc.add_dihedral(0, 1, 2, 3, -std::numbers::pi, std::numbers::pi);
  engine.add_constraint(std::move(dc));
  engine.build_atomic_groups(0.0, 0.05, 3);

  engine.run(1000);

  REQUIRE(engine.stats().steps_total == 1000);
  REQUIRE(engine.stats().steps_accepted == engine.stats().steps_tried);
}

TEST_CASE("dihedralConstraint - targeted rotamer shell produces accepted moves",
          "[example_dihedral_constraint]") {
  // Same chain; initial dihedral ≈ −2.19 rad; target [0.5, 1.0] is violated.
  auto s = make_structure({{0, 0, 0}, {1, 0, 0}, {2, 0, 1}, {3, 1, 1}},
                          {"C", "C", "C", "C"}, {0, 0, 0, 0});

  Engine engine(std::move(s), InfiniteBC(1e6));
  DihedralAngleConstraint dc;
  dc.add_dihedral(0, 1, 2, 3, 0.5, 1.0);
  engine.add_constraint(std::move(dc));
  engine.build_atomic_groups(0.0, 0.1, 5);

  engine.run(1000);

  REQUIRE(engine.stats().steps_accepted > 0);
}

// ─── 7. improperConstraint
// ──────────────────────────────────────────────────── fullrmc example:
// ImproperAngleConstraint enforces planarity of a molecular fragment
// (XeF5-like).
TEST_CASE("improperConstraint - tight planarity constraint holds near-planar "
          "geometry",
          "[example_improper_constraint]") {
  // 4 atoms: 3 in XY plane, 4th slightly out (improper ≈ 0.069 rad < 0.1).
  auto s = make_structure(
      {{0, 0, 0}, {1, 0, 0}, {0.5, 0.866, 0}, {0.5, 0.289, 0.02}},
      {"Xe", "F", "F", "F"}, {0, 0, 0, 0});

  Engine engine(std::move(s), InfiniteBC(1e6));
  ImproperAngleConstraint ic;
  ic.add_improper(0, 1, 2, 3, -0.1, 0.1);
  engine.add_constraint(std::move(ic));
  engine.build_atomic_groups(0.0, 0.05, 31);

  engine.run(1000);

  double phi = improper_angle(engine.structure().coordinates, 0, 1, 2, 3);
  REQUIRE(phi >= -0.1 - 1e-6);
  REQUIRE(phi <= 0.1 + 1e-6);
}

TEST_CASE("improperConstraint - out-of-plane start drives toward planarity",
          "[example_improper_constraint]") {
  // 4th atom strongly out of plane → large initial improper; engine accepts
  // moves that push the fragment back toward planarity.
  auto s =
      make_structure({{0, 0, 0}, {1, 0, 0}, {0.5, 0.866, 0}, {0.5, 0.289, 1.0}},
                     {"Xe", "F", "F", "F"}, {0, 0, 0, 0});

  Engine engine(std::move(s), InfiniteBC(1e6));
  ImproperAngleConstraint ic;
  ic.add_improper(0, 1, 2, 3, -0.1, 0.1);
  engine.add_constraint(std::move(ic));
  engine.build_atomic_groups(0.0, 0.1, 37);

  engine.run(1000);

  REQUIRE(engine.stats().steps_accepted > 0);
}

// ─── 8. coordNumConstraint
// ──────────────────────────────────────────────────── fullrmc example:
// AtomicCoordinationNumberConstraint locks the coordination number of a central
// atom (Al) to exactly 2 Cl neighbours in a shell.
TEST_CASE("coordNumConstraint - coordination number held at 2 after 2000 steps",
          "[example_coord_num_constraint]") {
  // Al(0) at origin; Cl(1,2) at ±2 Å → CN = 2 (in [1.5,2.5]); Cl(3,4) at ±10 Å
  // → outside.
  auto s = make_structure(
      {{0, 0, 0}, {2, 0, 0}, {-2, 0, 0}, {10, 0, 0}, {-10, 0, 0}},
      {"Al", "Cl", "Cl", "Cl", "Cl"}, {0, 1, 2, 3, 4});

  Engine engine(std::move(s), InfiniteBC(1e6));

  CoordinationConstraint cc;
  cc.add_shell(0, "Cl", 1.5, 2.5, 2, 2);
  cc.set_elements(engine.structure().elements);
  engine.add_constraint(std::move(cc));

  InterMolecularDistanceConstraint dc;
  dc.set_minimum_distance("Al", "Cl", 1.5);
  dc.set_structure(engine.structure().elements,
                   engine.structure().molecule_ids);
  engine.add_constraint(std::move(dc));

  // Move only Cl atoms (indices 1–4); Al stays fixed.
  for (std::size_t idx : {1u, 2u, 3u, 4u}) {
    Group g;
    g.name = "Cl_" + std::to_string(idx);
    g.indices = {idx};
    g.generator.emplace(
        TranslationGenerator(0.0, 0.1, 50u + static_cast<std::uint32_t>(idx)));
    engine.add_group(std::move(g));
  }

  engine.run(2000);

  const coords_t &c = engine.structure().coordinates;
  auto in_shell = [&](int j) {
    double d = (c.row(0) - c.row(j)).norm();
    return d >= 1.5 && d <= 2.5;
  };
  int cn = in_shell(1) + in_shell(2) + in_shell(3) + in_shell(4);
  REQUIRE(cn == 2);
}

// ─── 9. removes
// ─────────────────────────────────────────────────────────────── fullrmc
// example: AtomsRemoveGenerator stages atomic removals that the engine either
// commits (accepted) or rolls back (rejected).
TEST_CASE("removes - RemoveGenerator stages, commits, and rolls back correctly",
          "[example_removes]") {
  auto col = std::make_shared<AtomsCollector>();
  MoveGenerator gen = RemoveGenerator{col};

  coords_t c(6, 3);
  c.setZero();
  std::vector<index_t> group_indices = {1, 3};

  SECTION("commit makes atoms permanently removed") {
    gen.generate(c, group_indices);
    REQUIRE(col->pending().size() == 2);
    REQUIRE(col->n_active(6) == 6); // pending, not yet committed

    col->commit_removal();
    REQUIRE(col->n_removed() == 2);
    REQUIRE(col->is_removed(1));
    REQUIRE(col->is_removed(3));
    REQUIRE(!col->is_removed(0));
    REQUIRE(col->n_active(6) == 4);
  }

  SECTION("rollback clears pending without committing") {
    gen.generate(c, group_indices);
    REQUIRE(col->pending().size() == 2);

    col->rollback_removal();
    REQUIRE(col->pending().empty());
    REQUIRE(col->n_removed() == 0);
    REQUIRE(col->n_active(6) == 6);
  }

  SECTION("multiple removals accumulate in removed list") {
    std::vector<index_t> g1 = {0};
    std::vector<index_t> g2 = {4};
    gen.generate(c, g1);
    col->commit_removal();
    gen.generate(c, g2);
    col->commit_removal();

    REQUIRE(col->n_removed() == 2);
    REQUIRE(col->is_removed(0));
    REQUIRE(col->is_removed(4));
    REQUIRE(col->n_active(6) == 4);
  }
}
