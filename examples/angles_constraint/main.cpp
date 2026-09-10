// angles_constraint
// Three water molecules; BondConstraint + AngleConstraint on H-O-H.
// Phase 1: [80°,120°] — initial 90° is satisfied.
// Phase 2: [30°,40°]  — engine must compress angle.
// Phase 3: [160°,170°] — engine must open angle.
#include <RMC/Engine.hpp>
#include <RMC/constraints/GeometricConstraints.hpp>
#include <RMC/core/BoundaryConditions.hpp>
#include <RMC/core/Structure.hpp>

#include <cmath>
#include <iostream>
#include <numbers>

using namespace RMC;

static AtomicStructure make_3_waters() {
  AtomicStructure s;
  s.coordinates.resize(9, 3);
  s.atomic_numbers.resize(9);
  for (int m = 0; m < 3; ++m) {
    double ox = static_cast<double>(m) * 5.0;
    s.coordinates.row(3 * m) << ox, 0.0, 0.0;
    s.coordinates.row(3 * m + 1) << ox + 1.0, 0.0, 0.0;
    s.coordinates.row(3 * m + 2) << ox, 1.0, 0.0;
    for (int j = 0; j < 3; ++j) {
      const char *e = (j == 0) ? "O" : "H";
      s.atomic_numbers[3 * m + j] = (j == 0) ? 8 : 1;
      s.elements.push_back(e);
      s.names.push_back(e);
      s.residues.push_back("WAT");
      s.molecule_ids.push_back(static_cast<std::size_t>(m));
    }
  }
  return s;
}

int main() {
  using std::numbers::pi;
  struct Phase {
    const char *name;
    double lo_deg, hi_deg;
  };
  const Phase phases[] = {
      {"[80,120] deg", 80.0, 120.0},
      {"[30, 40] deg", 30.0, 40.0},
      {"[160,170] deg", 160.0, 170.0},
  };

  for (auto &ph : phases) {
    auto s = make_3_waters();
    Engine eng(std::move(s), InfiniteBC{});

    BondConstraint bc;
    for (int m = 0; m < 3; ++m) {
      bc.add_bond(static_cast<std::size_t>(3 * m),
                  static_cast<std::size_t>(3 * m + 1), 0.8, 1.4);
      bc.add_bond(static_cast<std::size_t>(3 * m),
                  static_cast<std::size_t>(3 * m + 2), 0.8, 1.4);
    }
    eng.add_constraint(std::move(bc));

    AngleConstraint ac;
    for (int m = 0; m < 3; ++m) {
      ac.add_angle(static_cast<std::size_t>(3 * m + 1),
                   static_cast<std::size_t>(3 * m),
                   static_cast<std::size_t>(3 * m + 2), ph.lo_deg * pi / 180.0,
                   ph.hi_deg * pi / 180.0);
    }
    eng.add_constraint(std::move(ac));

    eng.build_atomic_groups(0.0, 0.1, 7);
    eng.run(2000);

    std::cout << ph.name << ": accepted " << eng.stats().steps_accepted
              << "  last_err " << eng.stats().last_total_err << "\n";
  }
}
