// molecular_thf
// 6 THF-like ring molecules (5 heavy atoms each: O + 4 C), 30 atoms total.
// Constraints: BondConstraint, AngleConstraint, InterMolecular distance.
// Per-atom fine moves + per-molecule coarse translation & rotation.
#include <RMC/Engine.hpp>
#include <RMC/constraints/AngleConstraint.hpp>
#include <RMC/constraints/BondConstraint.hpp>
#include <RMC/constraints/DistanceConstraint.hpp>
#include <RMC/core/BoundaryConditions.hpp>
#include <RMC/core/Group.hpp>
#include <RMC/core/Structure.hpp>
#include <RMC/generators/Rotations.hpp>
#include <RMC/generators/Translations.hpp>
#include <RMC/selectors/SmartRandomSelector.hpp>

#include <cmath>
#include <iostream>
#include <numbers>

using namespace RMC;

// Single THF ring: O(0) C1(1) C2(2) C3(3) C4(4) in a pentagon.
// Radius ~0.73 Å for C-C ≈ 1.54 Å, C-O ≈ 1.43 Å.
static void add_thf(AtomicStructure &s, int mol_id, double cx, double cy,
                    double cz) {
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
    s.coordinates(row, 2) = cz;
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
    add_thf(s, m, static_cast<double>(m) * 4.0, 0.0, 0.0);
  return s;
}

int main() {
  constexpr int n_mol = 6;
  constexpr int atoms_per_mol = 5;
  auto s = make_thf(n_mol);
  Engine eng(std::move(s), InfiniteBC{});

  BondConstraint bc;
  AngleConstraint ac;
  for (int m = 0; m < n_mol; ++m) {
    int b = m * atoms_per_mol;
    // Ring bonds: O-C1, C1-C2, C2-C3, C3-C4, C4-O.
    int ring[5] = {b, b + 1, b + 2, b + 3, b + 4};
    for (int j = 0; j < 5; ++j) {
      int next = ring[(j + 1) % 5];
      bool oc = (j == 0 || j == 4);
      bc.add_bond(static_cast<std::size_t>(ring[j]),
                  static_cast<std::size_t>(next), oc ? 1.3 : 1.4,
                  oc ? 1.6 : 1.7);
    }
    // Ring angles at each vertex.
    for (int j = 0; j < 5; ++j) {
      int prev = ring[(j + 4) % 5];
      int next = ring[(j + 1) % 5];
      ac.add_angle(
          static_cast<std::size_t>(prev), static_cast<std::size_t>(ring[j]),
          static_cast<std::size_t>(next), 90.0 * std::numbers::pi / 180.0,
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

  // Per-atom fine translation.
  for (std::size_t i = 0; i < eng.structure().size(); ++i) {
    Group g;
    g.name = "a" + std::to_string(i);
    g.indices = {i};
    g.generator.emplace(
        TranslationGenerator(0.0, 0.1, static_cast<std::uint32_t>(i + 1)));
    eng.add_group(std::move(g));
  }

  // Per-molecule coarse translation + rotation.
  for (int m = 0; m < n_mol; ++m) {
    std::vector<std::size_t> idx;
    for (int j = 0; j < atoms_per_mol; ++j)
      idx.push_back(static_cast<std::size_t>(m * atoms_per_mol + j));

    Group gt;
    gt.name = "mol_t" + std::to_string(m);
    gt.indices = idx;
    gt.generator.emplace(
        TranslationGenerator(0.0, 0.2, static_cast<std::uint32_t>(m + 200)));
    eng.add_group(std::move(gt));

    Group gr;
    gr.name = "mol_r" + std::to_string(m);
    gr.indices = idx;
    gr.generator.emplace(
        RotationGenerator(0.0, 0.05, static_cast<std::uint32_t>(m + 300)));
    eng.add_group(std::move(gr));
  }

  eng.set_selector(GroupSelector{SmartRandomSelector{1.1, 13}});
  eng.run(20000);

  std::cout << "THF (" << n_mol << " mol): accepted "
            << eng.stats().steps_accepted << " / " << eng.stats().steps_tried
            << "  err " << eng.stats().last_total_err << "\n";
}
