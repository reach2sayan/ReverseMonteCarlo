// rotations
// Rigid-body rotation about various axes on a 4-atom tetrahedral molecule.
// No constraints: all pairwise distances must be preserved (rigid body).
#include <RMC/Engine.hpp>
#include <RMC/core/BoundaryConditions.hpp>
#include <RMC/core/Group.hpp>
#include <RMC/core/Structure.hpp>
#include <RMC/generators/Rotations.hpp>
#include <RMC/selectors/OrderedSelector.hpp>

#include <array>
#include <cmath>
#include <iostream>

using namespace RMC;

int main() {
  // Tetrahedron-ish 4-atom molecule.
  auto make_mol = [] {
    AtomicStructure s;
    s.coordinates.resize(4, 3);
    s.coordinates << 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0;
    s.atomic_numbers.resize(4);
    s.atomic_numbers.setConstant(6);
    for (int i = 0; i < 4; ++i) {
      s.elements.push_back("C");
      s.names.push_back("C");
      s.residues.push_back("MOL");
      s.molecule_ids.push_back(0);
    }
    return s;
  };

  struct Case {
    const char *label;
    MoveGenerator gen;
  };
  std::vector<Case> cases;
  cases.push_back({"about X axis", MoveGenerator{RotationAboutAxisGenerator(
                                       {1, 0, 0}, 0.0, 0.5, 1)}});
  cases.push_back({"about Y axis", MoveGenerator{RotationAboutAxisGenerator(
                                       {0, 1, 0}, 0.0, 0.5, 2)}});
  cases.push_back({"about Z axis", MoveGenerator{RotationAboutAxisGenerator(
                                       {0, 0, 1}, 0.0, 0.5, 3)}});
  cases.push_back(
      {"about (1,1,1) axis",
       MoveGenerator{RotationAboutAxisGenerator({1, 1, 1}, 0.0, 0.5, 4)}});
  cases.push_back(
      {"random axis", MoveGenerator{RotationGenerator(0.0, 0.5, 5)}});

  for (auto &[label, gen] : cases) {
    auto s = make_mol();
    // Record initial pairwise distances.
    std::array<double, 6> d0;
    int idx = 0;
    for (int a = 0; a < 4; ++a)
      for (int b = a + 1; b < 4; ++b)
        d0[idx++] = (s.coordinates.row(a) - s.coordinates.row(b)).norm();

    Engine eng(std::move(s), InfiniteBC{});
    Group g;
    g.name = "mol";
    g.indices = {0, 1, 2, 3};
    g.generator = std::move(gen);
    eng.add_group(std::move(g));
    eng.set_selector(GroupSelector{OrderedSelector{}});
    eng.run(500);

    // Verify distances preserved.
    const auto &c = eng.structure().coordinates;
    idx = 0;
    double max_err = 0.0;
    for (int a = 0; a < 4; ++a)
      for (int b = a + 1; b < 4; ++b) {
        double d = (c.row(a) - c.row(b)).norm();
        max_err = std::max(max_err, std::abs(d - d0[idx++]));
      }
    std::cout << label << ": accepted " << eng.stats().steps_accepted
              << "  max dist error " << max_err << "\n";
  }
}
