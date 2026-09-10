// generator_collector
// 6 THF-like rings loaded from data/thf.pdb (O,C1,C2,C3,C4 pentagon).
// MoveGeneratorCollector randomly picks one generator per step (weighted):
//   translation (w=3), rotation (w=2), angle agitation on O-C1-C2 (w=1).
// Compared against CombinedMoveGenerator which applies all three every step.
// BondConstraint, AngleConstraint, and InterMolecularDistanceConstraint active.
#include <RMC/Engine.hpp>
#include <RMC/constraints/GeometricConstraints.hpp>
#include <RMC/constraints/DistanceConstraint.hpp>
#include <RMC/core/BoundaryConditions.hpp>
#include <RMC/core/Group.hpp>
#include <RMC/core/Structure.hpp>
#include <RMC/generators/Agitations.hpp>
#include <RMC/generators/Combined.hpp>
#include <RMC/generators/Rotations.hpp>
#include <RMC/generators/Translations.hpp>
#include <RMC/io/PdbReader.hpp>
#include <RMC/selectors/SmartRandomSelector.hpp>

#include <iostream>
#include <map>
#include <numbers>
#include <vector>

using namespace RMC;

static AtomicStructure load(const char *path) {
  auto r = io::read_pdb(path);
  if (!r) {
    std::cerr << "Cannot open " << path << "\n";
    std::exit(1);
  }
  return std::move(*r);
}

static std::map<std::size_t, std::vector<std::size_t>>
molecules(const AtomicStructure &s) {
  std::map<std::size_t, std::vector<std::size_t>> m;
  for (std::size_t i = 0; i < s.size(); ++i)
    m[s.molecule_ids[i]].push_back(i);
  return m;
}

static void
add_constraints(Engine &eng,
                const std::map<std::size_t, std::vector<std::size_t>> &mols) {
  BondConstraint bc;
  AngleConstraint ac;
  for (auto &[mid, a] : mols) {
    // Ring bonds: O-C1, C1-C2, C2-C3, C3-C4, C4-O
    for (int j = 0; j < 5; ++j) {
      int next = (j + 1) % 5;
      bool oc = (j == 0 || j == 4);
      bc.add_bond(a[static_cast<std::size_t>(j)],
                  a[static_cast<std::size_t>(next)], oc ? 1.3 : 1.4,
                  oc ? 1.6 : 1.7);
    }
    // Ring angles at each vertex.
    for (int j = 0; j < 5; ++j) {
      int prev = (j + 4) % 5, next = (j + 1) % 5;
      ac.add_angle(
          a[static_cast<std::size_t>(prev)], a[static_cast<std::size_t>(j)],
          a[static_cast<std::size_t>(next)], 90.0 * std::numbers::pi / 180.0,
          115.0 * std::numbers::pi / 180.0);
    }
  }
  eng.add_constraint(std::move(bc));
  eng.add_constraint(std::move(ac));

  InterMolecularDistanceConstraint dc;
  dc.set_minimum_distance("O", "O", 1.5);
  dc.set_minimum_distance("C", "C", 1.5);
  dc.set_minimum_distance("O", "C", 1.5);
  dc.set_structure(eng.structure().elements, eng.structure().molecule_ids);
  eng.add_constraint(std::move(dc));
}

static void run(const char *label, bool use_collector,
                const AtomicStructure &tmpl) {
  const auto mols = molecules(tmpl);
  Engine eng(tmpl, InfiniteBC{});
  add_constraints(eng, mols);

  std::uint32_t seed = 0;
  for (auto &[mid, a] : mols) {
    Group g;
    g.name = "mol_" + std::to_string(mid);
    g.indices = a;
    // O-C1-C2 angle agitation uses ring atoms 0,1,2 of each molecule.
    std::size_t iO = a[0], iC1 = a[1], iC2 = a[2];

    if (use_collector) {
      MoveGeneratorCollector col(seed + 100);
      col.add(TranslationGenerator(0.0, 0.15, seed + 1), 3.0);
      col.add(RotationGenerator(0.0, 0.08, seed + 2), 2.0);
      col.add(AngleAgitationGenerator(iO, iC1, iC2, 0.0, 0.04, seed + 3), 1.0);
      g.generator = std::move(col);
    } else {
      g.generator = CombinedMoveGenerator{
          TranslationGenerator(0.0, 0.15, seed + 1),
          RotationGenerator(0.0, 0.08, seed + 2),
          AngleAgitationGenerator(iO, iC1, iC2, 0.0, 0.04, seed + 3)};
    }
    eng.add_group(std::move(g));
    seed += 10;
  }

  eng.set_selector(GroupSelector{SmartRandomSelector{1.1, 13}});
  eng.run(15000);
  std::cout << label << ": accepted " << eng.stats().steps_accepted << " / "
            << eng.stats().steps_tried << "  err " << eng.stats().last_total_err
            << "\n";
}

int main() {
  const auto tmpl = load("data/thf.pdb");
  const auto mols = molecules(tmpl);
  std::cout << "Loaded " << tmpl.size() << " atoms (" << mols.size()
            << " THF rings) from thf.pdb\n\n";

  run("MoveGeneratorCollector (select one)", true, tmpl);
  run("CombinedMoveGenerator  (apply all) ", false, tmpl);
}
