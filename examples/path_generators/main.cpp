// path_generators — fullrmc equivalent
// Single C atom loaded from data/atom.pdb; C4 tetrahedron from data/tetrahedron.pdb.
// Demonstrates TranslationAlongAxisPath and RotationAboutAxisPath:
// predefined move sequences applied cyclically.
//
// Scenario A — TranslationAlongAxisPath:
//   Single atom; a 3-step oscillation path along Z (+0.5, -0.3, +0.2 Å)
//   repeated for 9 steps. The cumulative Z displacement after each full cycle
//   should equal +0.4 Å.
//
// Scenario B — RotationAboutAxisPath:
//   Tetrahedral molecule; 4-step rotation path (+π/4, +π/4, +π/4, -3π/4)
//   that returns to the starting orientation after one full cycle.
//   Pairwise distances must be preserved at every step.
#include <RMC/Engine.hpp>
#include <RMC/core/BoundaryConditions.hpp>
#include <RMC/core/Group.hpp>
#include <RMC/core/Structure.hpp>
#include <RMC/generators/Path.hpp>
#include <RMC/io/PdbReader.hpp>
#include <RMC/selectors/OrderedSelector.hpp>

#include <cmath>
#include <iostream>
#include <numbers>

using namespace RMC;

static AtomicStructure load(const char *path) {
  auto r = io::read_pdb(path);
  if (!r) { std::cerr << "Cannot open " << path << "\n"; std::exit(1); }
  return std::move(*r);
}

int main() {
  const auto atom = load("data/atom.pdb");
  const auto tet  = load("data/tetrahedron.pdb");
  std::cout << "Loaded atom.pdb (" << atom.size() << " atom) at ("
            << atom.coordinates(0,0) << ", " << atom.coordinates(0,1) << ", "
            << atom.coordinates(0,2) << ")\n";
  std::cout << "Loaded tetrahedron.pdb (" << tet.size() << " atoms)\n\n";

  // ---- A: TranslationAlongAxisPath ----
  {
    std::cout << "=== A: TranslationAlongAxisPath ===\n";
    // Path sums per cycle: +0.5 - 0.3 + 0.2 = +0.4 Å
    std::vector<double> steps = {0.5, -0.3, 0.2};
    Engine eng(atom, InfiniteBC{});

    Group g;
    g.name = "atom";
    g.indices = {0};
    g.generator.emplace(TranslationAlongAxisPath({0, 0, 1}, steps));
    eng.add_group(std::move(g));
    eng.set_selector(IGroupSelector{OrderedSelector{}});

    double expected_z = atom.coordinates(0, 2);
    for (int step = 0; step < 9; ++step) {
      eng.run(1);
      expected_z += steps[step % 3];
      double z = eng.structure().coordinates(0, 2);
      std::cout << "  step " << step + 1 << ": z = " << z
                << "  expected = " << expected_z << "\n";
    }
  }

  // ---- B: RotationAboutAxisPath ----
  {
    std::cout << "\n=== B: RotationAboutAxisPath ===\n";
    constexpr double q = std::numbers::pi / 4.0;
    // 4 steps: +45°, +45°, +45°, -135° → net 0 per cycle.
    std::vector<double> angles = {q, q, q, -3.0 * q};

    const double d01 = (tet.coordinates.row(0) - tet.coordinates.row(1)).norm();
    const double d23 = (tet.coordinates.row(2) - tet.coordinates.row(3)).norm();

    Engine eng(tet, InfiniteBC{});
    Group g;
    g.name = "mol";
    g.indices = {0, 1, 2, 3};
    g.generator.emplace(RotationAboutAxisPath({0, 0, 1}, angles));
    eng.add_group(std::move(g));
    eng.set_selector(IGroupSelector{OrderedSelector{}});

    for (int step = 0; step < 8; ++step) {
      eng.run(1);
      const auto &c = eng.structure().coordinates;
      double e01 = std::abs((c.row(0) - c.row(1)).norm() - d01);
      double e23 = std::abs((c.row(2) - c.row(3)).norm() - d23);
      std::cout << "  step " << step + 1 << ": dist_err01=" << e01
                << " dist_err23=" << e23 << "\n";
    }
  }
}
