// reduced_sq — fullrmc equivalent
// NiTi-like binary alloy loaded from data/system.pdb (32-atom 4×4×2 supercell).
// ReducedStructureFactorConstraint fits F(Q) = Q·(S(Q)−1) data loaded
// from data/experimental.fq at runtime — mirrors fullrmc's atomicNiTi example
// which loads experimental.fq committed alongside the code.
// Three phases with decreasing step amplitude.
#include <RMC/Engine.hpp>
#include <RMC/constraints/ReducedStructureFactorConstraint.hpp>
#include <RMC/core/BoundaryConditions.hpp>
#include <RMC/core/Group.hpp>
#include <RMC/core/Structure.hpp>
#include <RMC/generators/Translations.hpp>
#include <RMC/io/DataReader.hpp>
#include <RMC/io/PdbReader.hpp>
#include <RMC/selectors/SmartRandomSelector.hpp>

#include <cmath>
#include <iostream>

using namespace RMC;

static AtomicStructure load_pdb(const char *path) {
  auto r = io::read_pdb(path);
  if (!r) {
    std::cerr << "Cannot open " << path << "\n";
    std::exit(1);
  }
  return std::move(*r);
}

static mat_t load_fq(const char *path) {
  auto r = io::read_xy_data(path);
  if (!r) {
    std::cerr << "Cannot open " << path << "\n";
    std::exit(1);
  }
  return std::move(*r);
}

static Engine build(const AtomicStructure &tmpl, double a,
                    const mat_t &fq_data) {
  mat3_t box = mat3_t::Zero();
  box.diagonal() << 4 * a, 4 * a, 2 * a;
  Engine eng(tmpl, PeriodicBC{box});

  ReducedStructureFactorConstraint rfq;
  rfq.set_experimental_data(fq_data);
  rfq.set_number_density(static_cast<double>(tmpl.size()) /
                         (4 * a * 4 * a * 2 * a));
  rfq.set_elements(eng.structure().elements);
  rfq.initialise();
  eng.add_constraint(std::move(rfq));
  return eng;
}

static void run_phase(const char *label, double amp,
                      const AtomicStructure &tmpl, double a,
                      const mat_t &fq_data, std::uint64_t n_steps) {
  auto eng = build(tmpl, a, fq_data);
  eng.build_atomic_groups(0.0, amp, 42);
  eng.set_selector(IGroupSelector{SmartRandomSelector{1.1, 42}});
  eng.run(n_steps);
  std::cout << label << " (amp=" << amp << "): accepted "
            << eng.stats().steps_accepted << "  err "
            << eng.stats().last_total_err << "\n";
}

int main() {
  constexpr double a = 2.87;

  const auto tmpl = load_pdb("data/system.pdb");
  const auto fq_data = load_fq("data/experimental.fq");

  std::cout << "Loaded " << tmpl.size() << " atoms from system.pdb\n";
  std::cout << "Loaded " << fq_data.rows()
            << " Q points from experimental.fq\n\n";

  run_phase("Phase 1", 0.10, tmpl, a, fq_data, 5000);
  run_phase("Phase 2", 0.05, tmpl, a, fq_data, 5000);
  run_phase("Phase 3", 0.02, tmpl, a, fq_data, 5000);
}
