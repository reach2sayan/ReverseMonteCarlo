// reduced_sq — fullrmc equivalent
// NiTi-like binary alloy (4×4×2 simple cubic, a=2.87 Å, 32 atoms).
// ReducedStructureFactorConstraint fits F(Q) = Q·(S(Q)−1) data loaded
// from a file. Three phases with decreasing step amplitude.
//
// The example writes a synthetic F(Q) file to /tmp/niti_fq.dat on startup,
// then reads it back — matching the fullrmc pattern of loading experimental
// data from disk at runtime.
#include <RMC/Engine.hpp>
#include <RMC/constraints/ReducedStructureFactorConstraint.hpp>
#include <RMC/core/BoundaryConditions.hpp>
#include <RMC/core/Group.hpp>
#include <RMC/core/Structure.hpp>
#include <RMC/generators/Translations.hpp>
#include <RMC/io/DataReader.hpp>
#include <RMC/selectors/SmartRandomSelector.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numbers>

using namespace RMC;

// 4×4×2 NiTi-like checkerboard on a simple cubic lattice.
static AtomicStructure make_niti(double a = 2.87) {
  AtomicStructure s;
  constexpr int Nx = 4, Ny = 4, Nz = 2;
  constexpr int N = Nx * Ny * Nz;
  s.coordinates.resize(N, 3);
  s.atomic_numbers.resize(N);
  int idx = 0;
  for (int iz = 0; iz < Nz; ++iz)
    for (int iy = 0; iy < Ny; ++iy)
      for (int ix = 0; ix < Nx; ++ix) {
        s.coordinates.row(idx) << ix * a, iy * a, iz * a;
        bool ni = ((ix + iy + iz) % 2 == 0);
        s.atomic_numbers[idx] = ni ? 28 : 22;
        s.elements.push_back(ni ? "Ni" : "Ti");
        s.names.push_back(ni ? "Ni" : "Ti");
        s.residues.push_back("ALL");
        s.molecule_ids.push_back(static_cast<std::size_t>(idx));
        ++idx;
      }
  return s;
}

// Synthetic F(Q) for a liquid-like model:
//   F(Q) = A * sin(Q*r0) * exp(-b*Q) where r0 ≈ nearest-neighbour distance.
static void write_fq(const std::filesystem::path &path, double r0, int nQ = 60,
                     double Q_min = 0.5, double Q_max = 15.0) {
  std::ofstream f(path);
  f << "# Q[1/A]  F(Q)\n";
  for (int i = 0; i < nQ; ++i) {
    double Q = Q_min + (Q_max - Q_min) * i / (nQ - 1);
    double fq = 3.0 * std::sin(Q * r0) * std::exp(-0.1 * Q);
    f << Q << "  " << fq << "\n";
  }
}

static Engine build(const AtomicStructure &tmpl, double a,
                    const std::filesystem::path &fq_path) {
  mat3_t box = mat3_t::Zero();
  box.diagonal() << 4 * a, 4 * a, 2 * a;
  Engine eng(tmpl, PeriodicBC{box});

  auto data_or_err = io::read_xy_data(fq_path);
  if (!data_or_err) {
    std::cerr << "Failed to read F(Q) file: " << fq_path << "\n";
    std::exit(1);
  }
  const mat_t &data = *data_or_err;

  ReducedStructureFactorConstraint rfq;
  rfq.set_experimental_data(data);
  rfq.set_number_density(static_cast<double>(tmpl.size()) /
                         (4 * a * 4 * a * 2 * a));
  rfq.set_elements(eng.structure().elements);
  rfq.initialise();
  eng.add_constraint(std::move(rfq));
  return eng;
}

static void run_phase(const char *label, double amp,
                      const AtomicStructure &tmpl, double a,
                      const std::filesystem::path &fq_path,
                      std::uint64_t n_steps) {
  auto eng = build(tmpl, a, fq_path);
  eng.build_atomic_groups(0.0, amp, 42);
  eng.set_selector(IGroupSelector{SmartRandomSelector{1.1, 42}});
  eng.run(n_steps);
  std::cout << label << " (amp=" << amp << "): accepted "
            << eng.stats().steps_accepted << "  err "
            << eng.stats().last_total_err << "\n";
}

int main() {
  constexpr double a = 2.87;
  const auto tmpl = make_niti(a);

  // Write synthetic F(Q) to a temp file, then load it — mirrors fullrmc's
  // pattern of loading experimental data at runtime.
  const std::filesystem::path fq_path = std::filesystem::temp_directory_path() /
                                         "niti_fq.dat";
  write_fq(fq_path, a); // nearest-neighbour distance = a
  std::cout << "Wrote F(Q) to " << fq_path << "\n";

  run_phase("Phase 1", 0.10, tmpl, a, fq_path, 5000);
  run_phase("Phase 2", 0.05, tmpl, a, fq_path, 5000);
  run_phase("Phase 3", 0.02, tmpl, a, fq_path, 5000);
}
