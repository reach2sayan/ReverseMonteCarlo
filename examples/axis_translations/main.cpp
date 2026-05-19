// axis_translations — fullrmc equivalent
// Demonstrates TranslationTowardsAxisGenerator, TranslationAlongSymmetryAxisGenerator,
// TranslationTowardsSymmetryAxisGenerator, and RotationAboutSymmetryAxisGenerator
// on simple structures. No constraints: all moves accepted.
//
// Scenario A: single atom initially at (3,4,0); move it toward the Z-axis.
// Scenario B: single atom; translate along each of X, Y, Z symmetry axes.
// Scenario C: single atom at (3,4,7); move it toward the Z symmetry axis.
// Scenario D: tetrahedral molecule; rotate about each Cartesian symmetry axis.
#include <RMC/Engine.hpp>
#include <RMC/core/BoundaryConditions.hpp>
#include <RMC/core/Group.hpp>
#include <RMC/core/Structure.hpp>
#include <RMC/generators/Rotations.hpp>
#include <RMC/generators/Translations.hpp>
#include <RMC/selectors/OrderedSelector.hpp>

#include <cmath>
#include <iostream>

using namespace RMC;

static AtomicStructure single_atom(double x, double y, double z) {
  AtomicStructure s;
  s.coordinates.resize(1, 3);
  s.coordinates.row(0) << x, y, z;
  s.atomic_numbers.resize(1);
  s.atomic_numbers[0] = 6;
  s.elements.push_back("C");
  s.names.push_back("C");
  s.residues.push_back("MOL");
  s.molecule_ids.push_back(0);
  return s;
}

static AtomicStructure tetrahedron() {
  AtomicStructure s;
  s.coordinates.resize(4, 3);
  s.coordinates << 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
      1.0;
  s.atomic_numbers.resize(4);
  s.atomic_numbers.setConstant(6);
  for (int i = 0; i < 4; ++i) {
    s.elements.push_back("C");
    s.names.push_back("C");
    s.residues.push_back("MOL");
    s.molecule_ids.push_back(0);
  }
  return s;
}

static void run_single(const char *label, IMoveGenerator gen, double x,
                       double y, double z, std::uint64_t n = 200) {
  auto s = single_atom(x, y, z);
  Engine eng(std::move(s), InfiniteBC{});
  Group g;
  g.name = "a";
  g.indices = {0};
  g.generator = std::move(gen);
  eng.add_group(std::move(g));
  eng.set_selector(IGroupSelector{OrderedSelector{}});
  eng.run(n);
  const auto &c = eng.structure().coordinates;
  std::cout << label << ": final (" << c(0, 0) << ", " << c(0, 1) << ", "
            << c(0, 2) << ")\n";
}

static void run_rotation(const char *label, IMoveGenerator gen) {
  auto s = tetrahedron();
  const auto &orig = s.coordinates;
  double d01 = (orig.row(0) - orig.row(1)).norm();
  double d23 = (orig.row(2) - orig.row(3)).norm();
  Engine eng(std::move(s), InfiniteBC{});
  Group g;
  g.name = "mol";
  g.indices = {0, 1, 2, 3};
  g.generator = std::move(gen);
  eng.add_group(std::move(g));
  eng.set_selector(IGroupSelector{OrderedSelector{}});
  eng.run(500);
  const auto &c = eng.structure().coordinates;
  double max_err = std::max(std::abs((c.row(0) - c.row(1)).norm() - d01),
                            std::abs((c.row(2) - c.row(3)).norm() - d23));
  std::cout << label << ": accepted " << eng.stats().steps_accepted
            << "  max_dist_err " << max_err << "\n";
}

int main() {
  std::cout << "=== A: TranslationTowardsAxisGenerator (toward Z-axis) ===\n";
  run_single("towards Z-axis  ",
             TranslationTowardsAxisGenerator({0, 0, 0}, {0, 0, 1}, 0.0, 0.5,
                                             1),
             3.0, 4.0, 0.0);

  std::cout << "\n=== B: TranslationAlongSymmetryAxisGenerator ===\n";
  run_single("along X", TranslationAlongSymmetryAxisGenerator(SymmetryAxis::X, 0.0, 0.3, 2), 0, 0, 0);
  run_single("along Y", TranslationAlongSymmetryAxisGenerator(SymmetryAxis::Y, 0.0, 0.3, 3), 0, 0, 0);
  run_single("along Z", TranslationAlongSymmetryAxisGenerator(SymmetryAxis::Z, 0.0, 0.3, 4), 0, 0, 0);

  std::cout << "\n=== C: TranslationTowardsSymmetryAxisGenerator ===\n";
  run_single("toward X-axis",
             TranslationTowardsSymmetryAxisGenerator(SymmetryAxis::X, 0.0, 0.5, 5),
             0.0, 4.0, 3.0);
  run_single("toward Y-axis",
             TranslationTowardsSymmetryAxisGenerator(SymmetryAxis::Y, 0.0, 0.5, 6),
             4.0, 0.0, 3.0);
  run_single("toward Z-axis",
             TranslationTowardsSymmetryAxisGenerator(SymmetryAxis::Z, 0.0, 0.5, 7),
             3.0, 4.0, 0.0);

  std::cout << "\n=== D: RotationAboutSymmetryAxisGenerator ===\n";
  run_rotation("about X", RotationAboutSymmetryAxisGenerator(SymmetryAxis::X, 0.0, 0.5, 8));
  run_rotation("about Y", RotationAboutSymmetryAxisGenerator(SymmetryAxis::Y, 0.0, 0.5, 9));
  run_rotation("about Z", RotationAboutSymmetryAxisGenerator(SymmetryAxis::Z, 0.0, 0.5, 10));
}
