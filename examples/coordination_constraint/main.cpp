// coordination_constraint
// Al(0) surrounded by 4 Cl atoms; CoordinationConstraint requires exactly
// 2 Cl neighbours in [1.5,2.5] Å shell. Engine repositions Cl atoms.
#include <RMC/Engine.hpp>
#include <RMC/constraints/CoordinationConstraint.hpp>
#include <RMC/constraints/DistanceConstraint.hpp>
#include <RMC/core/BoundaryConditions.hpp>
#include <RMC/core/Group.hpp>
#include <RMC/core/Structure.hpp>
#include <RMC/generators/Translations.hpp>

#include <iostream>

using namespace RMC;

int main() {
  // Al at origin; Cl1/Cl2 near (CN=2 in shell), Cl3/Cl4 far (CN outside).
  AtomicStructure s;
  s.coordinates.resize(5, 3);
  s.coordinates << 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, -2.0, 0.0, 0.0, 10.0, 0.0, 0.0,
      -10.0, 0.0, 0.0;
  s.atomic_numbers.resize(5);
  s.atomic_numbers[0] = 13; // Al
  s.atomic_numbers[1] = s.atomic_numbers[2] = s.atomic_numbers[3] =
      s.atomic_numbers[4] = 17; // Cl
  s.elements = {"Al", "Cl", "Cl", "Cl", "Cl"};
  s.names = s.elements;
  for (int i = 0; i < 5; ++i) {
    s.residues.push_back(i == 0 ? "AL" : "CL");
    s.molecule_ids.push_back(static_cast<std::size_t>(i));
  }

  Engine eng(std::move(s), InfiniteBC{});

  CoordinationConstraint cc;
  cc.add_shell(0, "Cl", 1.5, 2.5, 2, 2);
  cc.set_elements(eng.structure().elements);
  eng.add_constraint(std::move(cc));

  InterMolecularDistanceConstraint dc;
  dc.set_minimum_distance("Al", "Cl", 1.5);
  dc.set_structure(eng.structure().elements, eng.structure().molecule_ids);
  eng.add_constraint(std::move(dc));

  // Only Cl atoms move; Al is fixed.
  for (std::size_t i = 1; i < 5; ++i) {
    Group g;
    g.name = "Cl_" + std::to_string(i);
    g.indices = {i};
    g.generator.emplace(
        TranslationGenerator(0.0, 0.15, static_cast<std::uint32_t>(50 + i)));
    eng.add_group(std::move(g));
  }

  eng.run(10000);

  const auto &c = eng.structure().coordinates;
  int cn = 0;
  for (int j = 1; j < 5; ++j) {
    double d = (c.row(0) - c.row(j)).norm();
    if (d >= 1.5 && d <= 2.5)
      ++cn;
  }
  std::cout << "Accepted: " << eng.stats().steps_accepted
            << "  Al-Cl CN in [1.5,2.5]: " << cn << " (target 2)\n";
}
