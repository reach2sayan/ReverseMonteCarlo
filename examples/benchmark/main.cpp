// benchmark — fullrmc equivalent
// 20 THF molecules (13 atoms each: O + C1..C4 + 8H, 260 atoms total) loaded
// from data/system.pdb. G(r) experimental data from data/experimental.gr.
//
// Two benchmark sweeps:
//   A) benchmark_constraints — 7 constraint subsets × group sizes 1‥29
//      Measures time/step, tried, accepted. Saves 3 .dat files.
//   B) benchmark_nsteps — all constraints, group size 13 (one molecule),
//      step counts 5000‥100000. Saves 3 .dat files.
//
// Output mirrors fullrmc's benchmark/run.py.
#include <RMC/Engine.hpp>
#include <RMC/constraints/AngleConstraint.hpp>
#include <RMC/constraints/BondConstraint.hpp>
#include <RMC/constraints/DistanceConstraint.hpp>
#include <RMC/constraints/ImproperAngleConstraint.hpp>
#include <RMC/constraints/PairDistributionConstraint.hpp>
#include <RMC/core/BoundaryConditions.hpp>
#include <RMC/core/Group.hpp>
#include <RMC/core/Structure.hpp>
#include <RMC/generators/Translations.hpp>
#include <RMC/io/DataReader.hpp>
#include <RMC/io/PdbReader.hpp>
#include <RMC/selectors/SmartRandomSelector.hpp>

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numbers>
#include <string>
#include <vector>

using namespace RMC;
using Clock = std::chrono::steady_clock;

// ---------------------------------------------------------------------------
// Data loading
// ---------------------------------------------------------------------------
static AtomicStructure load_pdb(const char *path) {
  auto r = io::read_pdb(path);
  if (!r) {
    std::cerr << "Cannot open " << path << "\n";
    std::exit(1);
  }
  return std::move(*r);
}

static mat_t load_gr(const char *path) {
  auto r = io::read_xy_data(path);
  if (!r) {
    std::cerr << "Cannot open " << path << "\n";
    std::exit(1);
  }
  return std::move(*r);
}

// ---------------------------------------------------------------------------
// Constraint builders — add THF bonds/angles/impropers per molecule.
// Atom order per molecule: O(0) C1(1) C2(2) C3(3) C4(4)
//                          H11(5) H12(6) H21(7) H22(8) H31(9) H32(10) H41(11)
//                          H42(12)
// ---------------------------------------------------------------------------
static constexpr int ATOMS_PER_MOL = 13;

static void add_bonds(BondConstraint &bc, std::size_t base) {
  // Ring bonds
  bc.add_bond(base + 0, base + 1, 1.22, 1.70); // O-C1
  bc.add_bond(base + 0, base + 4, 1.22, 1.70); // O-C4
  bc.add_bond(base + 1, base + 2, 1.25, 1.90); // C1-C2
  bc.add_bond(base + 2, base + 3, 1.25, 1.90); // C2-C3
  bc.add_bond(base + 3, base + 4, 1.25, 1.90); // C3-C4
  // C-H bonds
  bc.add_bond(base + 1, base + 5, 0.58, 1.22);  // C1-H11
  bc.add_bond(base + 1, base + 6, 0.58, 1.22);  // C1-H12
  bc.add_bond(base + 2, base + 7, 0.58, 1.22);  // C2-H21
  bc.add_bond(base + 2, base + 8, 0.58, 1.22);  // C2-H22
  bc.add_bond(base + 3, base + 9, 0.58, 1.22);  // C3-H31
  bc.add_bond(base + 3, base + 10, 0.58, 1.22); // C3-H32
  bc.add_bond(base + 4, base + 11, 0.58, 1.22); // C4-H41
  bc.add_bond(base + 4, base + 12, 0.58, 1.22); // C4-H42
}

static void add_angles(AngleConstraint &ac, std::size_t b) {
  const double lo = 95.0 * std::numbers::pi / 180.0;
  const double hi = 125.0 * std::numbers::pi / 180.0;
  const double lH = 98.0 * std::numbers::pi / 180.0;
  const double hH = 123.0 * std::numbers::pi / 180.0;
  // Ring angles
  ac.add_angle(b + 1, b + 0, b + 4, lo, hi); // C1-O-C4
  ac.add_angle(b + 0, b + 1, b + 2, lo, hi); // O-C1-C2
  ac.add_angle(b + 1, b + 2, b + 3, lo, hi); // C1-C2-C3
  ac.add_angle(b + 2, b + 3, b + 4, lo, hi); // C2-C3-C4
  ac.add_angle(b + 3, b + 4, b + 0, lo, hi); // C3-C4-O
  // H-C-H angles
  ac.add_angle(b + 5, b + 1, b + 6, lH, hH);
  ac.add_angle(b + 7, b + 2, b + 8, lH, hH);
  ac.add_angle(b + 9, b + 3, b + 10, lH, hH);
  ac.add_angle(b + 11, b + 4, b + 12, lH, hH);
  // H-C-C / H-C-O angles
  ac.add_angle(b + 5, b + 1, b + 2, lH, hH); // H11-C1-C2
  ac.add_angle(b + 6, b + 1, b + 2, lH, hH); // H12-C1-C2
  ac.add_angle(b + 7, b + 2, b + 1, lH, hH); // H21-C2-C1
  ac.add_angle(b + 8, b + 2, b + 1, lH, hH);
  ac.add_angle(b + 7, b + 2, b + 3, lH, hH);
  ac.add_angle(b + 8, b + 2, b + 3, lH, hH);
  ac.add_angle(b + 9, b + 3, b + 2, lH, hH);
  ac.add_angle(b + 10, b + 3, b + 2, lH, hH);
  ac.add_angle(b + 9, b + 3, b + 4, lH, hH);
  ac.add_angle(b + 10, b + 3, b + 4, lH, hH);
  ac.add_angle(b + 11, b + 4, b + 3, lH, hH);
  ac.add_angle(b + 12, b + 4, b + 3, lH, hH);
}

static void add_impropers(ImproperAngleConstraint &ia, std::size_t b) {
  const double lo = -15.0 * std::numbers::pi / 180.0;
  const double hi = 15.0 * std::numbers::pi / 180.0;
  ia.add_improper(b + 2, b + 0, b + 1, b + 4, lo, hi); // C2,O,C1,C4
  ia.add_improper(b + 3, b + 0, b + 1, b + 4, lo, hi); // C3,O,C1,C4
}

// ---------------------------------------------------------------------------
// Engine factory
// ---------------------------------------------------------------------------
struct Flags {
  bool pdf, vdw, bond, angle, improper;
};

static Engine build_engine(const AtomicStructure &tmpl, const mat_t &gr_data,
                           Flags f, int group_size) {
  const std::size_t N = tmpl.size();
  const int n_mol = static_cast<int>(N) / ATOMS_PER_MOL;
  Engine eng(tmpl, InfiniteBC{});

  if (f.pdf) {
    PairDistributionConstraint pdc;
    pdc.set_experimental_data(gr_data);
    const double vol = 20.0 * 8.0 * 20.0; // approximate box volume Å³
    pdc.set_number_density(static_cast<double>(N) / vol);
    pdc.set_elements(eng.structure().elements);
    pdc.set_molecule_ids(eng.structure().molecule_ids);
    pdc.set_exclude_intra(true);
    pdc.initialise();
    eng.add_constraint(std::move(pdc));
  }

  if (f.vdw) {
    InterMolecularDistanceConstraint dc;
    dc.set_minimum_distance("O", "O", 1.5);
    dc.set_minimum_distance("O", "C", 1.5);
    dc.set_minimum_distance("O", "H", 1.2);
    dc.set_minimum_distance("C", "C", 1.5);
    dc.set_minimum_distance("C", "H", 1.2);
    dc.set_minimum_distance("H", "H", 1.0);
    dc.set_structure(eng.structure().elements, eng.structure().molecule_ids);
    eng.add_constraint(std::move(dc));
  }

  if (f.bond) {
    BondConstraint bc;
    for (int m = 0; m < n_mol; ++m)
      add_bonds(bc, static_cast<std::size_t>(m * ATOMS_PER_MOL));
    eng.add_constraint(std::move(bc));
  }

  if (f.angle) {
    AngleConstraint ac;
    for (int m = 0; m < n_mol; ++m)
      add_angles(ac, static_cast<std::size_t>(m * ATOMS_PER_MOL));
    eng.add_constraint(std::move(ac));
  }

  if (f.improper) {
    ImproperAngleConstraint ia;
    for (int m = 0; m < n_mol; ++m)
      add_impropers(ia, static_cast<std::size_t>(m * ATOMS_PER_MOL));
    eng.add_constraint(std::move(ia));
  }

  // Groups: contiguous blocks of `group_size` atoms.
  std::uint32_t seed = 1;
  for (std::size_t base = 0; base < N;
       base += static_cast<std::size_t>(group_size)) {
    std::size_t end = std::min(base + static_cast<std::size_t>(group_size), N);
    Group g;
    g.name = "g" + std::to_string(base);
    for (std::size_t i = base; i < end; ++i)
      g.indices.push_back(i);
    g.generator.emplace(TranslationGenerator(0.0, 0.15, seed++));
    eng.add_group(std::move(g));
  }

  eng.set_selector(GroupSelector{SmartRandomSelector{1.1, 42}});
  return eng;
}

// ---------------------------------------------------------------------------
// Single timed run
// ---------------------------------------------------------------------------
struct RunResult {
  double sec_per_step;
  std::uint64_t tried, accepted;
};

static RunResult timed_run(const AtomicStructure &tmpl, const mat_t &gr_data,
                           Flags f, int group_size, int nsteps) {
  auto eng = build_engine(tmpl, gr_data, f, group_size);
  auto t0 = Clock::now();
  eng.run(static_cast<std::uint64_t>(nsteps));
  double elapsed = std::chrono::duration<double>(Clock::now() - t0).count();
  const auto &st = eng.stats();
  return {elapsed / nsteps, st.steps_tried, st.steps_accepted};
}

// ---------------------------------------------------------------------------
// A) benchmark_constraints
// ---------------------------------------------------------------------------
static void benchmark_constraints(const AtomicStructure &tmpl,
                                  const mat_t &gr_data, int nsteps,
                                  int max_group) {
  std::cout << "================ Benchmark constraints ================\n";

  struct Combo {
    const char *name;
    Flags f;
  };
  const Combo combos[] = {
      {"none", {false, false, false, false, false}},
      {"pdf", {true, false, false, false, false}},
      {"vdw", {false, true, false, false, false}},
      {"bond", {false, false, true, false, false}},
      {"angle", {false, false, false, true, false}},
      {"improper", {false, false, false, false, true}},
      {"all", {true, true, true, true, true}},
  };
  constexpr int NC = static_cast<int>(std::size(combos));

  // Storage: time[GN][combo], tried[GN][combo], accepted[GN][combo]
  std::vector<int> group_sizes;
  for (int gn = 1; gn <= max_group; ++gn)
    group_sizes.push_back(gn);
  const int NG = static_cast<int>(group_sizes.size());

  std::vector<std::vector<double>> time_mat(NG, std::vector<double>(NC));
  std::vector<std::vector<std::uint64_t>> tried_mat(
      NG, std::vector<std::uint64_t>(NC));
  std::vector<std::vector<std::uint64_t>> acc_mat(
      NG, std::vector<std::uint64_t>(NC));

  for (int gi = 0; gi < NG; ++gi) {
    int gn = group_sizes[gi];
    std::cout << "++++++++ " << gn << " atoms per group\n";
    for (int ci = 0; ci < NC; ++ci) {
      std::cout << "---- " << combos[ci].name << "\n";
      auto r = timed_run(tmpl, gr_data, combos[ci].f, gn, nsteps);
      time_mat[gi][ci] = r.sec_per_step;
      tried_mat[gi][ci] = r.tried;
      acc_mat[gi][ci] = r.accepted;
    }
  }

  // Save .dat files
  auto save_int = [&](const char *fname,
                      const std::vector<std::vector<std::uint64_t>> &mat_i) {
    std::ofstream f(fname);
    f << "# groups";
    for (auto &c : combos)
      f << " " << c.name;
    f << "\n";
    for (int gi = 0; gi < NG; ++gi) {
      f << std::setw(6) << group_sizes[gi];
      for (int ci = 0; ci < NC; ++ci)
        f << "    " << mat_i[gi][ci];
      f << "\n";
    }
  };
  auto save_dbl = [&](const char *fname,
                      const std::vector<std::vector<double>> &mat_d) {
    std::ofstream f(fname);
    f << "# groups";
    for (auto &c : combos)
      f << " " << c.name;
    f << "\n";
    for (int gi = 0; gi < NG; ++gi) {
      f << std::setw(6) << group_sizes[gi];
      for (int ci = 0; ci < NC; ++ci)
        f << "    " << std::fixed << std::setprecision(8) << mat_d[gi][ci];
      f << "\n";
    }
  };
  save_dbl("benchmark_constraints_time.dat", time_mat);
  save_int("benchmark_constraints_tried.dat", tried_mat);
  save_int("benchmark_constraints_accepted.dat", acc_mat);
  std::cout << "Saved benchmark_constraints_*.dat\n\n";
}

// ---------------------------------------------------------------------------
// B) benchmark_nsteps
// ---------------------------------------------------------------------------
static void benchmark_nsteps(const AtomicStructure &tmpl, const mat_t &gr_data,
                             int group_size) {
  std::cout << "================ Benchmark number of steps ================\n";
  Flags all{true, true, true, true, true};

  std::vector<int> steps_list;
  for (int n = 5000; n <= 100000; n += 5000)
    steps_list.push_back(n);
  const int NS = static_cast<int>(steps_list.size());

  std::vector<double> times(NS);
  std::vector<std::uint64_t> tried(NS), accepted(NS);

  for (int si = 0; si < NS; ++si) {
    int ns = steps_list[si];
    std::cout << "---- " << ns << " steps\n";
    auto r = timed_run(tmpl, gr_data, all, group_size, ns);
    times[si] = r.sec_per_step;
    tried[si] = r.tried;
    accepted[si] = r.accepted;
  }

  const std::string tag =
      "all_Steps_" + std::to_string(group_size) + "GroupSize";
  auto save_steps = [&](const char *suffix, bool is_double, bool use_tried) {
    std::string fname = "benchmark_" + tag + suffix;
    std::ofstream f(fname);
    f << "# steps timePerStep(s)\n";
    for (int si = 0; si < NS; ++si) {
      f << std::setw(8) << steps_list[si] << "    ";
      if (is_double)
        f << std::fixed << std::setprecision(8) << times[si];
      else if (use_tried)
        f << tried[si];
      else
        f << accepted[si];
      f << "\n";
    }
  };
  save_steps("_time.dat", true, false);
  save_steps("_tried.dat", false, true);
  save_steps("_accepted.dat", false, false);
  std::cout << "Saved benchmark_" << tag << "_*.dat\n";
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
  const auto tmpl = load_pdb("data/system.pdb");
  const auto gr_data = load_gr("data/experimental.gr");
  const int n_mol = static_cast<int>(tmpl.size()) / ATOMS_PER_MOL;

  std::cout << "Loaded " << tmpl.size() << " atoms (" << n_mol
            << " THF molecules) from system.pdb\n";
  std::cout << "Loaded " << gr_data.rows()
            << " G(r) points from experimental.gr\n\n";

  constexpr int NSTEPS = 10;
  benchmark_constraints(tmpl, gr_data, NSTEPS, 29);
  benchmark_nsteps(tmpl, gr_data, ATOMS_PER_MOL);
}
