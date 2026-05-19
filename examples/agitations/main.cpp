// agitations — fullrmc equivalent
// Single water molecule (O-H1-H2). Demonstrates DistanceAgitationGenerator
// (shiver bond lengths) and AngleAgitationGenerator (shiver H-O-H angle).
// Three phases:
//   Phase 1: distance agitation on O-H1 and O-H2
//   Phase 2: angle agitation on H1-O-H2
//   Phase 3: all three agitations combined via MoveGeneratorCollector
// BondConstraint and AngleConstraint keep the molecule chemically sane.
#include <RMC/Engine.hpp>
#include <RMC/constraints/AngleConstraint.hpp>
#include <RMC/constraints/BondConstraint.hpp>
#include <RMC/core/BoundaryConditions.hpp>
#include <RMC/core/Group.hpp>
#include <RMC/core/Structure.hpp>
#include <RMC/generators/Agitations.hpp>
#include <RMC/generators/Combined.hpp>
#include <RMC/selectors/OrderedSelector.hpp>

#include <cmath>
#include <iostream>
#include <numbers>

using namespace RMC;

// Water: O(0) H1(1) H2(2). O-H bond = 0.96 Å, H-O-H angle = 104.45°.
static AtomicStructure make_water() {
  AtomicStructure s;
  s.coordinates.resize(3, 3);
  constexpr double r_oh = 0.96;
  constexpr double half_angle = 104.45 * std::numbers::pi / 360.0;
  s.coordinates.row(0) << 0.0, 0.0, 0.0;                                // O
  s.coordinates.row(1) << r_oh * std::sin(half_angle), r_oh * std::cos(half_angle), 0.0;  // H1
  s.coordinates.row(2) << -r_oh * std::sin(half_angle), r_oh * std::cos(half_angle), 0.0; // H2
  s.atomic_numbers.resize(3);
  s.atomic_numbers << 8, 1, 1;
  for (auto [e, n] : std::initializer_list<std::pair<const char *, const char *>>{
           {"O", "O"}, {"H", "H1"}, {"H", "H2"}}) {
    s.elements.push_back(e);
    s.names.push_back(n);
    s.residues.push_back("WAT");
    s.molecule_ids.push_back(0);
  }
  return s;
}

static void add_water_constraints(Engine &eng) {
  BondConstraint bc;
  bc.add_bond(0, 1, 0.85, 1.10); // O-H1
  bc.add_bond(0, 2, 0.85, 1.10); // O-H2
  eng.add_constraint(std::move(bc));

  AngleConstraint ac;
  // H1-O-H2: [100°, 110°]
  ac.add_angle(1, 0, 2, 100.0 * std::numbers::pi / 180.0,
               110.0 * std::numbers::pi / 180.0);
  eng.add_constraint(std::move(ac));
}

static void report(const char *label, const Engine &eng) {
  std::cout << label << ": accepted " << eng.stats().steps_accepted << " / "
            << eng.stats().steps_tried << "  err "
            << eng.stats().last_total_err << "\n";
}

int main() {
  constexpr std::uint64_t N = 5000;

  // Phase 1 — distance agitation only.
  {
    auto s = make_water();
    Engine eng(std::move(s), InfiniteBC{});
    add_water_constraints(eng);

    // Group 0–1: agitate O-H1 bond.
    {
      Group g;
      g.name = "oh1";
      g.indices = {0, 1};
      g.generator.emplace(DistanceAgitationGenerator(0, 1, 0.0, 0.03, 1));
      eng.add_group(std::move(g));
    }
    // Group 0–2: agitate O-H2 bond.
    {
      Group g;
      g.name = "oh2";
      g.indices = {0, 2};
      g.generator.emplace(DistanceAgitationGenerator(0, 2, 0.0, 0.03, 2));
      eng.add_group(std::move(g));
    }
    eng.set_selector(IGroupSelector{OrderedSelector{}});
    eng.run(N);
    report("Phase 1 (distance agitation)", eng);
  }

  // Phase 2 — angle agitation only.
  {
    auto s = make_water();
    Engine eng(std::move(s), InfiniteBC{});
    add_water_constraints(eng);

    Group g;
    g.name = "hoh";
    g.indices = {0, 1, 2};
    g.generator.emplace(AngleAgitationGenerator(1, 0, 2, 0.0, 0.04, 3));
    eng.add_group(std::move(g));
    eng.set_selector(IGroupSelector{OrderedSelector{}});
    eng.run(N);
    report("Phase 2 (angle agitation)   ", eng);
  }

  // Phase 3 — all three combined via MoveGeneratorCollector.
  {
    auto s = make_water();
    Engine eng(std::move(s), InfiniteBC{});
    add_water_constraints(eng);

    MoveGeneratorCollector col(4);
    col.add(DistanceAgitationGenerator(0, 1, 0.0, 0.03, 5), 1.0);
    col.add(DistanceAgitationGenerator(0, 2, 0.0, 0.03, 6), 1.0);
    col.add(AngleAgitationGenerator(1, 0, 2, 0.0, 0.04, 7), 1.0);

    Group g;
    g.name = "all";
    g.indices = {0, 1, 2};
    g.generator = std::move(col);
    eng.add_group(std::move(g));
    eng.set_selector(IGroupSelector{OrderedSelector{}});
    eng.run(N);
    report("Phase 3 (collector)         ", eng);
  }
}
