// orientation — fullrmc equivalent
// 6 CO2 molecules loaded from data/co2.pdb (O1-C-O2 along X, spaced along Y).
// OrientationGenerator aligns the molecular axis toward Z.
// BondConstraint keeps C-O bond lengths; AngleConstraint enforces linearity.
// Reports the mean angle between each molecule's axis and the Z-axis.
#include <RMC/Engine.hpp>
#include <RMC/constraints/AngleConstraint.hpp>
#include <RMC/constraints/BondConstraint.hpp>
#include <RMC/constraints/DistanceConstraint.hpp>
#include <RMC/core/BoundaryConditions.hpp>
#include <RMC/core/Group.hpp>
#include <RMC/core/Structure.hpp>
#include <RMC/generators/Rotations.hpp>
#include <RMC/io/PdbReader.hpp>
#include <RMC/selectors/SmartRandomSelector.hpp>

#include <cmath>
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

// Mean angle (degrees) between each CO2 principal axis and the Z-axis.
static double
mean_angle_deg(const Engine &eng,
               const std::map<std::size_t, std::vector<std::size_t>> &mols) {
  const auto &c = eng.structure().coordinates;
  vec3_t z{0, 0, 1};
  double sum = 0;
  for (auto &[mid, atoms] : mols) {
    // Principal axis: first atom → last atom in molecule.
    vec3_t ax = (c.row(static_cast<Eigen::Index>(atoms.back())) -
                 c.row(static_cast<Eigen::Index>(atoms.front())))
                    .transpose()
                    .normalized();
    double cos_a = std::abs(ax.dot(z));
    sum += std::acos(std::clamp(cos_a, 0.0, 1.0)) * 180.0 / std::numbers::pi;
  }
  return sum / static_cast<double>(mols.size());
}

int main() {
  const auto tmpl = load("data/co2.pdb");
  const auto mols = molecules(tmpl);
  std::cout << "Loaded " << tmpl.size() << " atoms (" << mols.size()
            << " CO2 molecules) from co2.pdb\n";

  {
    Engine tmp(tmpl, InfiniteBC{});
    std::cout << "Initial mean axis angle from Z: " << mean_angle_deg(tmp, mols)
              << "°\n\n";
  }

  Engine eng(tmpl, InfiniteBC{});

  BondConstraint bc;
  AngleConstraint ac;
  for (auto &[mid, atoms] : mols) {
    // O1-C and O2-C bonds [1.0, 1.3] Å.
    bc.add_bond(atoms[0], atoms[1], 1.0, 1.3);
    bc.add_bond(atoms[2], atoms[1], 1.0, 1.3);
    // O1-C-O2 linearity [170°, 180°].
    ac.add_angle(atoms[0], atoms[1], atoms[2], 170.0 * std::numbers::pi / 180.0,
                 std::numbers::pi);
  }
  eng.add_constraint(std::move(bc));
  eng.add_constraint(std::move(ac));

  InterMolecularDistanceConstraint dc;
  dc.set_minimum_distance("O", "O", 1.4);
  dc.set_minimum_distance("O", "C", 1.4);
  dc.set_minimum_distance("C", "C", 1.4);
  dc.set_structure(eng.structure().elements, eng.structure().molecule_ids);
  eng.add_constraint(std::move(dc));

  vec3_t target{0, 0, 1};
  std::uint32_t seed = 10;
  for (auto &[mid, atoms] : mols) {
    Group g;
    g.name = "mol_" + std::to_string(mid);
    g.indices = atoms;
    g.generator.emplace(OrientationGenerator(target, 0.15, seed++));
    eng.add_group(std::move(g));
  }

  eng.set_selector(GroupSelector{SmartRandomSelector{1.1, 7}});
  eng.run(20000);

  std::cout << "Final mean axis angle from Z:   " << mean_angle_deg(eng, mols)
            << "°\n";
  std::cout << "Accepted " << eng.stats().steps_accepted << " / "
            << eng.stats().steps_tried << "  err " << eng.stats().last_total_err
            << "\n";
}
