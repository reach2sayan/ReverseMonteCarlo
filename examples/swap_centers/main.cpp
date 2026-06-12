// swap_centers
// 8 C-N dimers loaded from data/dimers.pdb (2×4 grid, bond ~1.5 Å).
// SwapCentersGenerator translates each molecule so its centroid coincides
// with the centroid of a randomly chosen partner molecule.
// InterMolecularDistanceConstraint prevents atomic overlap.
#include <RMC/Engine.hpp>
#include <RMC/constraints/DistanceConstraint.hpp>
#include <RMC/core/BoundaryConditions.hpp>
#include <RMC/core/Group.hpp>
#include <RMC/core/Structure.hpp>
#include <RMC/generators/Swaps.hpp>
#include <RMC/io/PdbReader.hpp>
#include <RMC/selectors/SmartRandomSelector.hpp>

#include <cmath>
#include <iostream>
#include <map>
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

static double
centroid_spread(const Engine &eng,
                const std::map<std::size_t, std::vector<std::size_t>> &mols) {
  const auto &c = eng.structure().coordinates;
  double cx = 0, cy = 0;
  for (auto &[mid, atoms] : mols) {
    for (auto i : atoms) {
      cx += c(static_cast<Eigen::Index>(i), 0);
      cy += c(static_cast<Eigen::Index>(i), 1);
    }
  }
  std::size_t N = mols.size();
  cx /= static_cast<double>(N);
  cy /= static_cast<double>(N);
  double var = 0;
  for (auto &[mid, atoms] : mols) {
    double mx = 0, my = 0;
    for (auto i : atoms) {
      mx += c(static_cast<Eigen::Index>(i), 0);
      my += c(static_cast<Eigen::Index>(i), 1);
    }
    mx = mx / static_cast<double>(atoms.size()) - cx;
    my = my / static_cast<double>(atoms.size()) - cy;
    var += mx * mx + my * my;
  }
  return std::sqrt(var / static_cast<double>(N));
}

int main() {
  const auto tmpl = load("data/dimers.pdb");
  const auto mols = molecules(tmpl);
  std::cout << "Loaded " << tmpl.size() << " atoms (" << mols.size()
            << " dimers) from dimers.pdb\n";

  {
    Engine tmp(tmpl, InfiniteBC{});
    std::cout << "Initial centroid spread: " << centroid_spread(tmp, mols)
              << "\n\n";
  }

  Engine eng(tmpl, InfiniteBC{});

  InterMolecularDistanceConstraint dc;
  dc.set_minimum_distance("C", "C", 1.0);
  dc.set_minimum_distance("C", "N", 1.0);
  dc.set_minimum_distance("N", "N", 1.0);
  dc.set_structure(eng.structure().elements, eng.structure().molecule_ids);
  eng.add_constraint(std::move(dc));

  std::uint32_t seed = 1;
  for (auto &[mid, atoms] : mols) {
    std::vector<std::vector<std::size_t>> cands;
    for (auto &[omid, oatoms] : mols)
      if (omid != mid)
        cands.push_back(oatoms);
    Group g;
    g.name = "mol_" + std::to_string(mid);
    g.indices = atoms;
    g.generator.emplace(SwapCentersGenerator(std::move(cands), seed++));
    eng.add_group(std::move(g));
  }

  eng.set_selector(GroupSelector{SmartRandomSelector{1.1, 42}});
  eng.run(10000);

  std::cout << "Final centroid spread:   " << centroid_spread(eng, mols)
            << "\n";
  std::cout << "Accepted " << eng.stats().steps_accepted << " / "
            << eng.stats().steps_tried << "\n";
}
