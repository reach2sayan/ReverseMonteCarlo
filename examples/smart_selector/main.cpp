// smart_selector — fullrmc equivalent
// Compare RandomSelector vs SmartRandomSelector on 10 independent molecule
// groups with an InterMolecularDistanceConstraint.
// SmartRandomSelector adapts group weights based on move acceptance history.
#include <RMC/Engine.hpp>
#include <RMC/constraints/DistanceConstraint.hpp>
#include <RMC/core/BoundaryConditions.hpp>
#include <RMC/core/Group.hpp>
#include <RMC/core/Structure.hpp>
#include <RMC/generators/Translations.hpp>
#include <RMC/selectors/RandomSelector.hpp>
#include <RMC/selectors/SmartRandomSelector.hpp>

#include <iostream>
#include <string>

using namespace RMC;

// 10 single-atom "molecules" spaced along X.
static AtomicStructure make_10_atoms() {
  AtomicStructure s;
  s.coordinates.resize(10, 3);
  s.atomic_numbers.resize(10);
  s.atomic_numbers.setConstant(6);
  for (int i = 0; i < 10; ++i) {
    s.coordinates.row(i) << static_cast<double>(i) * 3.0, 0.0, 0.0;
    s.elements.push_back("C");
    s.names.push_back("C");
    s.residues.push_back("MOL");
    s.molecule_ids.push_back(static_cast<std::size_t>(i));
  }
  return s;
}

static void run_with(const std::string &label, IGroupSelector selector) {
  auto s = make_10_atoms();
  Engine eng(std::move(s), InfiniteBC{});

  InterMolecularDistanceConstraint dc;
  dc.set_minimum_distance("C", "C", 1.0);
  dc.set_structure(eng.structure().elements, eng.structure().molecule_ids);
  eng.add_constraint(std::move(dc));

  for (std::size_t i = 0; i < 10; ++i) {
    Group g;
    g.name = "mol_" + std::to_string(i);
    g.indices = {i};
    g.generator.emplace(
        TranslationGenerator(0.0, 0.3, static_cast<std::uint32_t>(i + 1)));
    eng.add_group(std::move(g));
  }

  eng.set_selector(std::move(selector));
  eng.run(20000);

  double accept_rate = eng.stats().steps_tried > 0
                           ? static_cast<double>(eng.stats().steps_accepted) /
                                 static_cast<double>(eng.stats().steps_tried)
                           : 0.0;
  std::cout << label << ": accepted " << eng.stats().steps_accepted << " / "
            << eng.stats().steps_tried << "  rate " << accept_rate << "\n";
}

int main() {
  run_with("RandomSelector     ", IGroupSelector{RandomSelector{42}});
  run_with("SmartRandomSelector", IGroupSelector{SmartRandomSelector{1.1, 42}});
}
