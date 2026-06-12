// improper_constraint
// XeF4-like planar fragment; ImproperAngleConstraint enforces planarity.
// Phase 1: tight ±2° — starts near planar.
// Phase 2: ±30°      — loosened, more freedom.
// Phase 3: ±2°       — restored tight.
#include <RMC/Engine.hpp>
#include <RMC/constraints/BondConstraint.hpp>
#include <RMC/constraints/ImproperAngleConstraint.hpp>
#include <RMC/core/BoundaryConditions.hpp>
#include <RMC/core/Structure.hpp>

#include <cmath>
#include <iostream>
#include <numbers>

using namespace RMC;

// Xe(0) + 3 F in XY plane; 4th F slightly out.
static AtomicStructure make_xef4() {
  AtomicStructure s;
  s.coordinates.resize(4, 3);
  s.coordinates << 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0, 0.0, -2.0, 0.0,
      0.02; // tiny out-of-plane
  s.atomic_numbers.resize(4);
  s.atomic_numbers[0] = 54; // Xe
  s.atomic_numbers[1] = s.atomic_numbers[2] = s.atomic_numbers[3] = 9;
  s.elements = {"Xe", "F", "F", "F"};
  s.names = {"Xe", "F", "F", "F"};
  for (int i = 0; i < 4; ++i) {
    s.residues.push_back("XEF");
    s.molecule_ids.push_back(0);
  }
  return s;
}

int main() {
  using std::numbers::pi;
  struct Phase {
    const char *name;
    double tol_deg;
  };
  const Phase phases[] = {
      {"tight  ±2°", 2.0},
      {"loose  ±30°", 30.0},
      {"restore ±2°", 2.0},
  };

  for (auto &ph : phases) {
    auto s = make_xef4();
    Engine eng(std::move(s), InfiniteBC{});

    BondConstraint bc;
    bc.add_bond(0, 1, 1.8, 2.2);
    bc.add_bond(0, 2, 1.8, 2.2);
    bc.add_bond(0, 3, 1.8, 2.2);
    eng.add_constraint(std::move(bc));

    ImproperAngleConstraint ic;
    const double tol = ph.tol_deg * pi / 180.0;
    ic.add_improper(0, 1, 2, 3, -tol, tol);
    eng.add_constraint(std::move(ic));

    eng.build_atomic_groups(0.0, 0.1, 11);
    eng.run(1000);

    std::cout << ph.name << ": accepted " << eng.stats().steps_accepted
              << "  last_err " << eng.stats().last_total_err << "\n";
  }
}
