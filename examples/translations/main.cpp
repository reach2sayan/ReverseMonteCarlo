// translations
// Demonstrates random, axis-aligned, and combined translations on a single
// atom. No constraints: every proposed move is accepted.
#include <RMC/Engine.hpp>
#include <RMC/core/BoundaryConditions.hpp>
#include <RMC/core/Group.hpp>
#include <RMC/core/Structure.hpp>
#include <RMC/generators/Translations.hpp>
#include <RMC/selectors/OrderedSelector.hpp>

#include <iostream>
#include <numbers>

using namespace RMC;

static AtomicStructure make_single_atom() {
  AtomicStructure s;
  s.coordinates.resize(1, 3);
  s.coordinates.setZero();
  s.atomic_numbers.resize(1);
  s.atomic_numbers[0] = 6;
  s.elements.push_back("C");
  s.names.push_back("C");
  s.residues.push_back("MOL");
  s.molecule_ids.push_back(0);
  return s;
}

int main() {
  constexpr std::uint64_t N = 500;

  struct Case {
    const char *label;
    MoveGenerator gen;
  };
  std::vector<Case> cases;
  cases.push_back(
      {"random direction", MoveGenerator{TranslationGenerator(0.0, 0.3, 1)}});
  cases.push_back({"along X axis", MoveGenerator{TranslationAlongAxisGenerator(
                                       {1, 0, 0}, 0.0, 0.3, 2)}});
  cases.push_back({"along Y axis", MoveGenerator{TranslationAlongAxisGenerator(
                                       {0, 1, 0}, 0.0, 0.3, 3)}});
  cases.push_back({"along Z axis", MoveGenerator{TranslationAlongAxisGenerator(
                                       {0, 0, 1}, 0.0, 0.3, 4)}});

  for (auto &[label, gen] : cases) {
    auto s = make_single_atom();
    Engine eng(std::move(s), InfiniteBC{});

    Group g;
    g.name = "atom";
    g.indices = {0};
    g.generator = std::move(gen);
    eng.add_group(std::move(g));
    eng.set_selector(GroupSelector{OrderedSelector{}});
    eng.run(N);

    std::cout << label << ": accepted " << eng.stats().steps_accepted << " / "
              << eng.stats().steps_tried << "\n";
  }
}
