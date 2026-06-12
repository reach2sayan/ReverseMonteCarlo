// axis_translations
// Loads structures from data/atom.pdb (single C) and data/tetrahedron.pdb (C4).
// Demonstrates:
//   A) TranslationTowardsAxisGenerator  — single atom moves toward the Z-axis
//   B) TranslationAlongSymmetryAxisGenerator — atom slides along X, Y, or Z
//   C) TranslationTowardsSymmetryAxisGenerator — atom moves toward a Cartesian axis
//   D) RotationAboutSymmetryAxisGenerator — tetrahedron rotates about X, Y, or Z
#include <RMC/Engine.hpp>
#include <RMC/core/BoundaryConditions.hpp>
#include <RMC/core/Group.hpp>
#include <RMC/core/Structure.hpp>
#include <RMC/generators/Rotations.hpp>
#include <RMC/generators/Translations.hpp>
#include <RMC/io/PdbReader.hpp>
#include <RMC/selectors/OrderedSelector.hpp>

#include <cmath>
#include <iostream>

using namespace RMC;

static AtomicStructure load(const char *path) {
  auto r = io::read_pdb(path);
  if (!r) { std::cerr << "Cannot open " << path << "\n"; std::exit(1); }
  return std::move(*r);
}

static void run_single(const char *label, MoveGenerator gen,
                       const AtomicStructure &tmpl, std::uint64_t n = 200) {
  Engine eng(tmpl, InfiniteBC{});
  Group g;
  g.name = "a";
  g.indices = {0};
  g.generator = std::move(gen);
  eng.add_group(std::move(g));
  eng.set_selector(GroupSelector{OrderedSelector{}});
  eng.run(n);
  const auto &c = eng.structure().coordinates;
  std::cout << label << ": final (" << c(0,0) << ", " << c(0,1) << ", "
            << c(0,2) << ")\n";
}

static void run_rotation(const char *label, MoveGenerator gen,
                         const AtomicStructure &tmpl) {
  double d01 = (tmpl.coordinates.row(0) - tmpl.coordinates.row(1)).norm();
  double d23 = (tmpl.coordinates.row(2) - tmpl.coordinates.row(3)).norm();
  Engine eng(tmpl, InfiniteBC{});
  Group g;
  g.name = "mol";
  g.indices = {0, 1, 2, 3};
  g.generator = std::move(gen);
  eng.add_group(std::move(g));
  eng.set_selector(GroupSelector{OrderedSelector{}});
  eng.run(500);
  const auto &c = eng.structure().coordinates;
  double max_err = std::max(
      std::abs((c.row(0)-c.row(1)).norm() - d01),
      std::abs((c.row(2)-c.row(3)).norm() - d23));
  std::cout << label << ": accepted " << eng.stats().steps_accepted
            << "  max_dist_err " << max_err << "\n";
}

int main() {
  const auto atom = load("data/atom.pdb");
  const auto tet  = load("data/tetrahedron.pdb");
  std::cout << "Loaded atom.pdb (" << atom.size() << " atom) at ("
            << atom.coordinates(0,0) << ", " << atom.coordinates(0,1) << ", "
            << atom.coordinates(0,2) << ")\n";
  std::cout << "Loaded tetrahedron.pdb (" << tet.size() << " atoms)\n\n";

  std::cout << "=== A: TranslationTowardsAxisGenerator (toward Z-axis) ===\n";
  run_single("towards Z-axis",
             TranslationTowardsAxisGenerator({0,0,0}, {0,0,1}, 0.0, 0.5, 1),
             atom);

  std::cout << "\n=== B: TranslationAlongSymmetryAxisGenerator ===\n";
  for (auto [name, ax] : {std::pair{"along X", SymmetryAxis::X},
                           std::pair{"along Y", SymmetryAxis::Y},
                           std::pair{"along Z", SymmetryAxis::Z}})
    run_single(name,
               TranslationAlongSymmetryAxisGenerator(ax, 0.0, 0.3,
                   static_cast<std::uint32_t>(ax == SymmetryAxis::X ? 2
                   : ax == SymmetryAxis::Y ? 3 : 4)),
               atom);

  std::cout << "\n=== C: TranslationTowardsSymmetryAxisGenerator ===\n";
  for (auto [name, ax] : {std::pair{"toward X-axis", SymmetryAxis::X},
                           std::pair{"toward Y-axis", SymmetryAxis::Y},
                           std::pair{"toward Z-axis", SymmetryAxis::Z}})
    run_single(name,
               TranslationTowardsSymmetryAxisGenerator(ax, 0.0, 0.5,
                   static_cast<std::uint32_t>(ax == SymmetryAxis::X ? 5
                   : ax == SymmetryAxis::Y ? 6 : 7)),
               atom);

  std::cout << "\n=== D: RotationAboutSymmetryAxisGenerator ===\n";
  for (auto [name, ax] : {std::pair{"about X", SymmetryAxis::X},
                           std::pair{"about Y", SymmetryAxis::Y},
                           std::pair{"about Z", SymmetryAxis::Z}})
    run_rotation(name,
                 RotationAboutSymmetryAxisGenerator(ax, 0.0, 0.5,
                     static_cast<std::uint32_t>(ax == SymmetryAxis::X ? 8
                     : ax == SymmetryAxis::Y ? 9 : 10)),
                 tet);
}
