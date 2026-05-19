// swap_centers — fullrmc equivalent
// 8 diatomic molecules (A-B, bond ~1.5 Å) arranged in a 2×4 grid.
// SwapCentersGenerator translates each molecule so its centroid coincides
// with the centroid of a randomly chosen partner molecule.
// InterMolecularDistanceConstraint prevents atomic overlap.
// After running, molecular centroids should be more evenly redistributed.
#include <RMC/Engine.hpp>
#include <RMC/constraints/DistanceConstraint.hpp>
#include <RMC/core/BoundaryConditions.hpp>
#include <RMC/core/Group.hpp>
#include <RMC/core/Structure.hpp>
#include <RMC/generators/Swaps.hpp>
#include <RMC/selectors/SmartRandomSelector.hpp>

#include <cmath>
#include <iostream>

using namespace RMC;

// n_mol diatomic A-B molecules on a 2D grid; atom 2m = A, atom 2m+1 = B.
static AtomicStructure make_dimers(int n_mol = 8) {
  AtomicStructure s;
  s.coordinates.resize(2 * n_mol, 3);
  s.atomic_numbers.resize(2 * n_mol);
  for (int m = 0; m < n_mol; ++m) {
    double cx = static_cast<double>(m % 4) * 5.0;
    double cy = static_cast<double>(m / 4) * 5.0;
    s.coordinates.row(2 * m) << cx - 0.75, cy, 0.0;
    s.coordinates.row(2 * m + 1) << cx + 0.75, cy, 0.0;
    s.atomic_numbers[2 * m] = 6;
    s.atomic_numbers[2 * m + 1] = 7;
    s.elements.push_back("C");
    s.elements.push_back("N");
    s.names.push_back("C");
    s.names.push_back("N");
    s.residues.push_back("DIM");
    s.residues.push_back("DIM");
    s.molecule_ids.push_back(static_cast<std::size_t>(m));
    s.molecule_ids.push_back(static_cast<std::size_t>(m));
  }
  return s;
}

static double mean_centroid_spread(const Engine &eng, int n_mol) {
  const auto &c = eng.structure().coordinates;
  double cx = 0, cy = 0;
  for (int m = 0; m < n_mol; ++m) {
    cx += 0.5 * (c(2 * m, 0) + c(2 * m + 1, 0));
    cy += 0.5 * (c(2 * m, 1) + c(2 * m + 1, 1));
  }
  cx /= n_mol;
  cy /= n_mol;
  double var = 0;
  for (int m = 0; m < n_mol; ++m) {
    double mx = 0.5 * (c(2 * m, 0) + c(2 * m + 1, 0)) - cx;
    double my = 0.5 * (c(2 * m, 1) + c(2 * m + 1, 1)) - cy;
    var += mx * mx + my * my;
  }
  return std::sqrt(var / n_mol);
}

int main() {
  constexpr int n_mol = 8;
  const auto tmpl = make_dimers(n_mol);

  std::cout << "Initial centroid spread: "
            << mean_centroid_spread(Engine(tmpl, InfiniteBC{}), n_mol) << "\n";

  Engine eng(tmpl, InfiniteBC{});

  InterMolecularDistanceConstraint dc;
  dc.set_minimum_distance("C", "C", 1.0);
  dc.set_minimum_distance("C", "N", 1.0);
  dc.set_minimum_distance("N", "N", 1.0);
  dc.set_structure(eng.structure().elements, eng.structure().molecule_ids);
  eng.add_constraint(std::move(dc));

  // Build candidate pool: all other molecules' atom lists.
  for (int m = 0; m < n_mol; ++m) {
    std::vector<std::vector<std::size_t>> cands;
    for (int other = 0; other < n_mol; ++other) {
      if (other == m)
        continue;
      cands.push_back({static_cast<std::size_t>(2 * other),
                       static_cast<std::size_t>(2 * other + 1)});
    }
    Group g;
    g.name = "mol_" + std::to_string(m);
    g.indices = {static_cast<std::size_t>(2 * m),
                 static_cast<std::size_t>(2 * m + 1)};
    g.generator.emplace(
        SwapCentersGenerator(std::move(cands),
                             static_cast<std::uint32_t>(m + 1)));
    eng.add_group(std::move(g));
  }

  eng.set_selector(IGroupSelector{SmartRandomSelector{1.1, 42}});
  eng.run(10000);

  std::cout << "Final centroid spread:   "
            << mean_centroid_spread(eng, n_mol) << "\n";
  std::cout << "Accepted " << eng.stats().steps_accepted << " / "
            << eng.stats().steps_tried << "\n";
}
