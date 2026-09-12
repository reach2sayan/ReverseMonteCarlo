// agitations
// 33 TIP water molecules loaded from data/waterBox.pdb (10×10×10 Å box).
// Demonstrates DistanceAgitationGenerator (shiver O-H bond lengths) and
// AngleAgitationGenerator (shiver H-O-H angles).
// Three phases:
//   Phase 1: distance agitation on both O-H bonds per molecule
//   Phase 2: angle agitation on H-O-H per molecule
//   Phase 3: all three agitations combined via MoveGeneratorCollector
// BondConstraint and AngleConstraint keep the molecules chemically sane.
#include <RMC/Engine.hpp>
#include <RMC/constraints/GeometricConstraints.hpp>
#include <RMC/core/BoundaryConditions.hpp>
#include <RMC/core/Group.hpp>
#include <RMC/core/Structure.hpp>
#include <RMC/generators/Agitations.hpp>
#include <RMC/generators/Combined.hpp>
#include <RMC/io/PdbReader.hpp>
#include <RMC/selectors/SmartRandomSelector.hpp>

#include <initializer_list>
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

// Group atoms by molecule_id → {O_idx, H1_idx, H2_idx} in PDB record order.
static std::map<std::size_t, std::vector<std::size_t>>
molecules(const AtomicStructure &s) {
  std::map<std::size_t, std::vector<std::size_t>> m;
  for (std::size_t i = 0; i < s.size(); ++i)
    m[s.molecule_ids[i]].push_back(i);
  return m;
}

static void run_phase(const char *label, bool dist_agit, bool angle_agit) {
  auto s = load("data/waterBox.pdb");
  const auto mols = molecules(s);

  Engine eng(std::move(s), InfiniteBC{});

  // O-H bonds [0.85, 1.10] Å; H-O-H angle [100°, 110°].
  BondConstraint bc;
  AngleConstraint ac;
  for (auto &[mid, atoms] : mols) {
    std::size_t iO = atoms[0], iH1 = atoms[1], iH2 = atoms[2];
    bc.add_bond(iO, iH1, 0.85, 1.10);
    bc.add_bond(iO, iH2, 0.85, 1.10);
    ac.add_angle(iH1, iO, iH2, 100.0 * std::numbers::pi / 180.0,
                 110.0 * std::numbers::pi / 180.0);
  }
  eng.add_constraint(std::move(bc));
  eng.add_constraint(std::move(ac));

  std::uint32_t seed = 1;
  for (auto &[mid, atoms] : mols) {
    std::size_t iO = atoms[0], iH1 = atoms[1], iH2 = atoms[2];

    if (dist_agit && angle_agit) {
      // Phase 3: collector picks one agitation per step.
      MoveGeneratorCollector col(seed + 200);
      col.add(DistanceAgitationGenerator(iO, iH1, 0.0, 0.03, seed), 1.0);
      col.add(DistanceAgitationGenerator(iO, iH2, 0.0, 0.03, seed + 1), 1.0);
      col.add(AngleAgitationGenerator(iH1, iO, iH2, 0.0, 0.04, seed + 2), 1.0);
      Group g;
      g.name = "w" + std::to_string(mid);
      g.indices = {iO, iH1, iH2};
      g.generator = std::move(col);
      eng.add_group(std::move(g));
    } else if (dist_agit) {
      for (auto [iH, s2] :
           std::initializer_list<std::pair<std::size_t, std::uint32_t>>{
               {iH1, seed}, {iH2, seed + 1}}) {
        Group g;
        g.name = "d" + std::to_string(mid) + "_" + std::to_string(iH);
        g.indices = {iO, iH};
        g.generator.emplace(DistanceAgitationGenerator(iO, iH, 0.0, 0.03, s2));
        eng.add_group(std::move(g));
      }
    } else {
      Group g;
      g.name = "a" + std::to_string(mid);
      g.indices = {iO, iH1, iH2};
      g.generator.emplace(
          AngleAgitationGenerator(iH1, iO, iH2, 0.0, 0.04, seed));
      eng.add_group(std::move(g));
    }
    seed += 10;
  }

  eng.set_selector(GroupSelector{SmartRandomSelector{1.1, 42}});
  eng.run(5000);
  std::cout << label << ": accepted " << eng.stats().steps_accepted << " / "
            << eng.stats().steps_tried << "  err " << eng.stats().last_total_err
            << "\n";
}

int main() {
  auto s = load("data/waterBox.pdb");
  std::cout << "Loaded " << s.size() << " atoms (" << molecules(s).size()
            << " water molecules) from waterBox.pdb\n\n";

  run_phase("Phase 1 (distance agitation)", true, false);
  run_phase("Phase 2 (angle agitation)   ", false, true);
  run_phase("Phase 3 (collector)         ", true, true);
}
