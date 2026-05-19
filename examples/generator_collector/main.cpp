// generator_collector — fullrmc equivalent
// 6 THF-like ring molecules. For each molecule a MoveGeneratorCollector
// randomly selects one of three generators per step (weighted):
//   - translation (weight 3): large-amplitude coarse move
//   - rotation (weight 2): rigid-body rotation
//   - agitation (weight 1): angle agitation of one ring bond
// Compares against a CombinedMoveGenerator baseline (applies all three every step).
// InterMolecularDistanceConstraint prevents overlap.
#include <RMC/Engine.hpp>
#include <RMC/constraints/AngleConstraint.hpp>
#include <RMC/constraints/BondConstraint.hpp>
#include <RMC/constraints/DistanceConstraint.hpp>
#include <RMC/core/BoundaryConditions.hpp>
#include <RMC/core/Group.hpp>
#include <RMC/core/Structure.hpp>
#include <RMC/generators/Agitations.hpp>
#include <RMC/generators/Combined.hpp>
#include <RMC/generators/Rotations.hpp>
#include <RMC/generators/Translations.hpp>
#include <RMC/selectors/SmartRandomSelector.hpp>

#include <cmath>
#include <iostream>
#include <numbers>

using namespace RMC;

static void add_thf(AtomicStructure &s, int mol_id, double cx, double cy) {
  constexpr int M = 5;
  const char *syms[M] = {"O", "C", "C", "C", "C"};
  const int ans[M] = {8, 6, 6, 6, 6};
  for (int j = 0; j < M; ++j) {
    double angle = j * 2.0 * std::numbers::pi / M;
    double r = 0.73;
    s.coordinates.conservativeResize(s.coordinates.rows() + 1, 3);
    Eigen::Index row = s.coordinates.rows() - 1;
    s.coordinates(row, 0) = cx + r * std::cos(angle);
    s.coordinates(row, 1) = cy + r * std::sin(angle);
    s.coordinates(row, 2) = 0.0;
    s.atomic_numbers.conservativeResize(s.atomic_numbers.size() + 1);
    s.atomic_numbers(s.atomic_numbers.size() - 1) = ans[j];
    s.elements.push_back(syms[j]);
    s.names.push_back(syms[j]);
    s.residues.push_back("THF");
    s.molecule_ids.push_back(static_cast<std::size_t>(mol_id));
  }
}

static AtomicStructure make_thf(int n_mol = 6) {
  AtomicStructure s;
  s.coordinates.resize(0, 3);
  s.atomic_numbers.resize(0);
  for (int m = 0; m < n_mol; ++m)
    add_thf(s, m, static_cast<double>(m) * 4.0, 0.0);
  return s;
}

static void add_constraints(Engine &eng, int n_mol) {
  constexpr int M = 5;
  BondConstraint bc;
  AngleConstraint ac;
  for (int m = 0; m < n_mol; ++m) {
    int b = m * M;
    for (int j = 0; j < M; ++j) {
      int next = b + (j + 1) % M;
      bool oc = (j == 0 || j == 4);
      bc.add_bond(static_cast<std::size_t>(b + j),
                  static_cast<std::size_t>(next), oc ? 1.3 : 1.4,
                  oc ? 1.6 : 1.7);
    }
    for (int j = 0; j < M; ++j) {
      int prev = b + (j + 4) % M;
      int next = b + (j + 1) % M;
      ac.add_angle(static_cast<std::size_t>(prev),
                   static_cast<std::size_t>(b + j),
                   static_cast<std::size_t>(next),
                   90.0 * std::numbers::pi / 180.0,
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

static void run(const char *label, bool use_collector, int n_mol = 6) {
  constexpr int M = 5;
  auto s = make_thf(n_mol);
  Engine eng(std::move(s), InfiniteBC{});
  add_constraints(eng, n_mol);

  for (int m = 0; m < n_mol; ++m) {
    std::vector<std::size_t> idx;
    for (int j = 0; j < M; ++j)
      idx.push_back(static_cast<std::size_t>(m * M + j));

    Group g;
    g.name = "mol_" + std::to_string(m);
    g.indices = idx;

    auto seed = static_cast<std::uint32_t>(m);
    std::size_t b = static_cast<std::size_t>(m * M);

    if (use_collector) {
      MoveGeneratorCollector col(seed + 100);
      col.add(TranslationGenerator(0.0, 0.15, seed + 1), 3.0);
      col.add(RotationGenerator(0.0, 0.08, seed + 2), 2.0);
      col.add(AngleAgitationGenerator(b, b + 1, b + 2, 0.0, 0.04, seed + 3),
              1.0);
      g.generator = std::move(col);
    } else {
      g.generator = CombinedMoveGenerator{
          TranslationGenerator(0.0, 0.15, seed + 1),
          RotationGenerator(0.0, 0.08, seed + 2),
          AngleAgitationGenerator(b, b + 1, b + 2, 0.0, 0.04, seed + 3)};
    }
    eng.add_group(std::move(g));
  }

  eng.set_selector(IGroupSelector{SmartRandomSelector{1.1, 13}});
  eng.run(15000);

  std::cout << label << ": accepted " << eng.stats().steps_accepted << " / "
            << eng.stats().steps_tried << "  err "
            << eng.stats().last_total_err << "\n";
}

int main() {
  run("MoveGeneratorCollector (select one)", true);
  run("CombinedMoveGenerator (apply all) ", false);
}
