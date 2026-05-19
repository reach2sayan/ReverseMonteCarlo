// orientation — fullrmc equivalent
// 6 CO2-like linear triatomic molecules (O-C-O), each initially placed along
// the X-axis. OrientationGenerator aligns the molecular axis toward Z.
// BondConstraint keeps C-O bond lengths. AngleConstraint keeps linearity.
// SmartRandomSelector with per-molecule groups.
#include <RMC/Engine.hpp>
#include <RMC/constraints/AngleConstraint.hpp>
#include <RMC/constraints/BondConstraint.hpp>
#include <RMC/constraints/DistanceConstraint.hpp>
#include <RMC/core/BoundaryConditions.hpp>
#include <RMC/core/Group.hpp>
#include <RMC/core/Structure.hpp>
#include <RMC/generators/Rotations.hpp>
#include <RMC/selectors/SmartRandomSelector.hpp>

#include <cmath>
#include <iostream>
#include <numbers>

using namespace RMC;

// n_mol CO2 molecules, all aligned along X, spaced 6 Å apart on Y.
// O(3m) C(3m+1) O(3m+2).
static AtomicStructure make_co2(int n_mol = 6) {
  AtomicStructure s;
  const int N = 3 * n_mol;
  s.coordinates.resize(N, 3);
  s.atomic_numbers.resize(N);
  for (int m = 0; m < n_mol; ++m) {
    double cy = static_cast<double>(m) * 6.0;
    s.coordinates.row(3 * m) << -1.16, cy, 0.0;
    s.coordinates.row(3 * m + 1) << 0.0, cy, 0.0;
    s.coordinates.row(3 * m + 2) << 1.16, cy, 0.0;
    s.atomic_numbers[3 * m] = s.atomic_numbers[3 * m + 2] = 8;
    s.atomic_numbers[3 * m + 1] = 6;
    s.elements.push_back("O");
    s.elements.push_back("C");
    s.elements.push_back("O");
    s.names.push_back("O");
    s.names.push_back("C");
    s.names.push_back("O");
    for (int j = 0; j < 3; ++j) {
      s.residues.push_back("CO2");
      s.molecule_ids.push_back(static_cast<std::size_t>(m));
    }
  }
  return s;
}

// Angle between the molecular axis and the Z-axis (degrees), averaged over all
// molecules.
static double mean_axis_angle_deg(const Engine &eng, int n_mol) {
  const auto &c = eng.structure().coordinates;
  double sum = 0.0;
  vec3_t z{0, 0, 1};
  for (int m = 0; m < n_mol; ++m) {
    vec3_t ax = (c.row(3 * m + 2) - c.row(3 * m)).transpose().normalized();
    double cos_a = std::abs(ax.dot(z));
    sum += std::acos(std::clamp(cos_a, 0.0, 1.0)) * 180.0 / std::numbers::pi;
  }
  return sum / n_mol;
}

int main() {
  constexpr int n_mol = 6;
  auto s = make_co2(n_mol);

  std::cout << "Initial mean axis angle from Z: "
            << mean_axis_angle_deg(Engine(s, InfiniteBC{}), n_mol) << "°\n";

  Engine eng(std::move(s), InfiniteBC{});

  // C-O bond [1.0, 1.3] Å.
  BondConstraint bc;
  for (int m = 0; m < n_mol; ++m) {
    bc.add_bond(static_cast<std::size_t>(3 * m),
                static_cast<std::size_t>(3 * m + 1), 1.0, 1.3);
    bc.add_bond(static_cast<std::size_t>(3 * m + 2),
                static_cast<std::size_t>(3 * m + 1), 1.0, 1.3);
  }
  eng.add_constraint(std::move(bc));

  // O-C-O linearity [170°, 180°].
  AngleConstraint ac;
  for (int m = 0; m < n_mol; ++m) {
    ac.add_angle(static_cast<std::size_t>(3 * m),
                 static_cast<std::size_t>(3 * m + 1),
                 static_cast<std::size_t>(3 * m + 2),
                 170.0 * std::numbers::pi / 180.0, std::numbers::pi);
  }
  eng.add_constraint(std::move(ac));

  // Intermolecular distance.
  InterMolecularDistanceConstraint dc;
  dc.set_minimum_distance("O", "O", 1.4);
  dc.set_minimum_distance("O", "C", 1.4);
  dc.set_minimum_distance("C", "C", 1.4);
  dc.set_structure(eng.structure().elements, eng.structure().molecule_ids);
  eng.add_constraint(std::move(dc));

  vec3_t target_axis{0, 0, 1}; // align CO2 molecules with Z

  for (int m = 0; m < n_mol; ++m) {
    std::vector<std::size_t> idx = {static_cast<std::size_t>(3 * m),
                                    static_cast<std::size_t>(3 * m + 1),
                                    static_cast<std::size_t>(3 * m + 2)};
    Group g;
    g.name = "mol_" + std::to_string(m);
    g.indices = idx;
    // max_offset = 0.15 rad ≈ 9°: allows small random perturbation.
    g.generator.emplace(
        OrientationGenerator(target_axis, 0.15,
                             static_cast<std::uint32_t>(m + 10)));
    eng.add_group(std::move(g));
  }

  eng.set_selector(IGroupSelector{SmartRandomSelector{1.1, 7}});
  eng.run(20000);

  std::cout << "Final mean axis angle from Z:   "
            << mean_axis_angle_deg(eng, n_mol) << "°\n";
  std::cout << "Accepted " << eng.stats().steps_accepted << " / "
            << eng.stats().steps_tried << "  err "
            << eng.stats().last_total_err << "\n";
}
