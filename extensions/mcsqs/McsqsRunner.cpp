#include "McsqsRunner.hpp"

#include <RMC/Engine.hpp>
#include <RMC/Ensemble.hpp>
#include <RMC/constraints/ClusterCorrelationConstraint.hpp>
#include <RMC/core/BoundaryConditions.hpp>
#include <RMC/core/Group.hpp>
#include <RMC/core/Structure.hpp>
#include <RMC/generators/SpeciesSwap.hpp>
#include <RMC/sampling/AnnealingSampler.hpp>
#include <RMC/sampling/GreedySampler.hpp>
#include <RMC/sampling/MetropolisSampler.hpp>
#include <RMC/sampling/Sampler.hpp>
#include <RMC/selectors/SmartRandomSelector.hpp>

#include "AtatFormats.hpp"
#include "AtatRunner.hpp"
#include "ClusterEnumerator.hpp"

#include <boost/program_options.hpp>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace po = boost::program_options;

namespace RMC {

namespace {

// Build sublattice groups: one sublattice per distinct residue name. Falls back
// to a single sublattice containing all sites when residues are absent.
std::vector<std::vector<std::size_t>>
build_sublattices(const AtomicStructure &str) {
  std::unordered_map<std::string, std::vector<std::size_t>> by_residue;
  for (std::size_t i = 0; i < str.size(); ++i) {
    by_residue[str.residues.empty() ? "all" : str.residues[i]].push_back(i);
  }

  std::vector<std::vector<std::size_t>> result;
  result.reserve(by_residue.size());
  std::ranges::transform(by_residue, std::back_inserter(result),
                         [](auto &kv) { return std::move(kv.second); });

  return result;
}

// Write a minimal PDB with the current element / coords.
void write_sqs_pdb(const AtomicStructure &str, const std::string &path) {
  std::ofstream f(path);
  if (!f)
    throw std::runtime_error("Cannot write: " + path);
  for (std::size_t i = 0; i < str.size(); ++i) {
    const auto &e = str.elements.empty() ? "X" : str.elements[i];
    f << std::left << std::setw(6) << "ATOM" << std::right << std::setw(5)
      << (i + 1) << "  " << std::left << std::setw(4) << e << std::setw(4)
      << (str.residues.empty() ? "LIG" : str.residues[i]) << "  "
      << std::setw(4) << (i + 1) << "    " << std::fixed << std::setprecision(3)
      << std::setw(8) << str.coordinates(static_cast<Eigen::Index>(i), 0)
      << std::setw(8) << str.coordinates(static_cast<Eigen::Index>(i), 1)
      << std::setw(8) << str.coordinates(static_cast<Eigen::Index>(i), 2)
      << "\n";
  }
  f << "END\n";
}

// Parse "--supercell" ("n" | "nx ny nz" | nine ints) into an integer matrix.
Eigen::Matrix3i build_sc_matrix(const std::string &s) {
  std::istringstream ss(s);
  std::vector<int> v;
  int x;
  while (ss >> x) {
    v.push_back(x);
  }
  Eigen::Matrix3i m = Eigen::Matrix3i::Zero();
  if (v.size() == 1) {
    m(0, 0) = m(1, 1) = m(2, 2) = v[0];
  } else if (v.size() == 3) {
    m(0, 0) = v[0];
    m(1, 1) = v[1];
    m(2, 2) = v[2];
  } else if (v.size() == 9) {
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        m(i, j) = v[static_cast<std::size_t>(i * 3 + j)];
      }
    }
  } else {
    throw std::runtime_error("--supercell: expected 1, 3, or 9 integers");
  }
  if (m.determinant() == 0) {
    throw std::runtime_error("--supercell: singular matrix");
  }
  return m;
}

// corrdump path: --corrdump > compile-time vendored path > "corrdump" on PATH.
std::string resolve_corrdump(const po::variables_map &vm) {
  const std::string opt = vm["corrdump"].as<std::string>();
  if (!opt.empty()) {
    return opt;
  }
#ifdef RMC_CORRDUMP_PATH
  return RMC_CORRDUMP_PATH;
#else
  return "corrdump";
#endif
}

// The acceptance-policy knobs, validated from the CLI.
struct SamplerSettings {
  std::string kind; // greedy | metropolis | anneal
  double t0;
  double cooling;
  std::uint64_t cool_interval;
};

// Validate --sampler and capture its tuning knobs. Throws on an unknown policy.
SamplerSettings parse_sampler_settings(const po::variables_map &vm,
                                       std::uint64_t n_steps) {
  const std::string kind = vm["sampler"].as<std::string>();
  if (kind != "greedy" && kind != "metropolis" && kind != "anneal") {
    throw std::runtime_error("unknown --sampler '" + kind +
                             "' (use greedy | metropolis | anneal)");
  }
  std::uint64_t cool_interval = vm["cool-interval"].as<std::uint64_t>();
  if (cool_interval == 0) {
    cool_interval = std::max<std::uint64_t>(1, n_steps / 20);
  }
  return {kind, vm["T0"].as<double>(), vm["cooling"].as<double>(),
          cool_interval};
}

// Build a fresh Sampler for the configured policy (one per engine).
Sampler make_sampler(const SamplerSettings &s) {
  if (s.kind == "greedy") {
    return Sampler{GreedySampler{}};
  }
  if (s.kind == "metropolis") {
    return Sampler{MetropolisSampler{s.t0}};
  }
  return Sampler{AnnealingSampler{AnnealingSampler::Schedule{
      .t0 = s.t0, .cooling = s.cooling, .interval = s.cool_interval}}};
}

// The resolved SQS problem from the ATAT corrdump pipeline: the supercell to
// refine, its sublattice grouping, and the enumerated cluster orbits. The
// constraint reads this object's own members, so add_constraint() keeps the
// backing data local to a long-lived McsqsProblem.
struct McsqsProblem {
  AtomicStructure structure;
  std::vector<std::vector<std::size_t>> sublattices;
  atat::EnumeratedSqs enumerated;
  std::string lattice_path;
  std::filesystem::path clusters_out_path;

  // Register the cluster-correlation constraint on a freshly built engine.
  void add_constraint(Engine &eng) const {
    eng.add_constraint(Constraint{ClusterCorrelationConstraint{
        eng.structure(), enumerated.occ_index, enumerated.table,
        enumerated.orbits}});
  }
};

// Resolve the CLI into an McsqsProblem via corrdump + symmetry enumeration.
// Throws std::runtime_error on bad input.
McsqsProblem resolve_problem(const po::variables_map &vm, std::uint32_t seed) {
  McsqsProblem prob;

  if (vm.count("lattice") == 0) {
    throw std::runtime_error("--lattice (rndstr.in primitive lattice) is "
                             "required");
  }
  prob.lattice_path = vm["lattice"].as<std::string>();
  const double d2 = vm["d2"].as<double>();
  if (d2 <= 0.0) {
    throw std::runtime_error(
        "--d2 (max pair diameter) is required with --lattice");
  }
  std::map<int, double> diam{{2, d2}};
  if (vm["d3"].as<double>() > 0.0) {
    diam[3] = vm["d3"].as<double>();
  }
  if (vm["d4"].as<double>() > 0.0) {
    diam[4] = vm["d4"].as<double>();
  }

  const Eigen::Matrix3i sc = build_sc_matrix(vm["supercell"].as<std::string>());
  const auto lat = atat::parse_lattice(prob.lattice_path);
  const auto workdir =
      std::filesystem::temp_directory_path() / "mcsqs_rmc_clusters";
  const auto cl = atat::corrdump_generate_clusters(
      resolve_corrdump(vm), prob.lattice_path, diam, workdir);
  prob.clusters_out_path = cl.clusters_out;
  const auto sym = atat::parse_sym(cl.sym_out);
  const auto raw = atat::parse_clusters(cl.clusters_out);
  prob.enumerated = atat::enumerate(lat, sym, raw, sc, seed);
  prob.structure = prob.enumerated.structure;
  prob.sublattices = build_sublattices(prob.structure);
  std::cout << "Lattice sites: " << lat.sites.size()
            << "  Species: " << lat.labels.size()
            << "  Supercell atoms: " << prob.structure.size()
            << "  Orbits: " << prob.enumerated.orbits.size() << "\n";

  return prob;
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

// Configure an engine IN PLACE. The cluster constraint and SpeciesSwap
// generator hold references into eng.structure(), so the engine must NOT be
// moved afterwards — see run_search's heap-stability note.
void configure_engine(Engine &eng, const McsqsProblem &prob,
                      const SamplerSettings &sampler, std::uint32_t rseed,
                      std::uint64_t log_every) {
  prob.add_constraint(eng);

  // One group per site; generator performs sublattice-aware species swap.
  SpeciesSwapGenerator gen{eng.structure(), prob.sublattices, rseed};
  for (std::size_t i = 0; i < eng.structure().size(); ++i) {
    Group g;
    g.name = "site_" + std::to_string(i);
    g.indices = {i};
    g.generator = MoveGenerator{gen};
    eng.add_group(std::move(g));
  }

  eng.set_selector(
      GroupSelector{SmartRandomSelector{static_cast<double>(rseed) + 1.0}});
  eng.set_sampler(make_sampler(sampler), rseed);
  eng.set_track_best(true); // annealing's final state isn't the minimum
  eng.set_step_callback(
      [log_every](std::uint64_t total, std::uint64_t accepted,
                  std::uint64_t /*tried*/, double err,
                  const AtomicStructure & /*structure*/) {
        if (total % log_every == 0)
          std::cout << "  step " << total << "  accepted " << accepted
                    << "  err " << std::scientific << err << "\n";
      },
      log_every);
}

// Run the search: an island-model ensemble (best error wins) when
// --replicas > 1, otherwise a single annealed run.
Engine run_search(const McsqsProblem &prob, const SamplerSettings &sampler,
                  std::uint32_t seed, std::size_t n_replicas,
                  std::uint64_t n_steps, std::uint64_t log_every) {
  auto make_engine = [&](std::uint32_t rseed) {
    Engine eng{prob.structure, InfiniteBC{}};
    configure_engine(eng, prob, sampler, rseed, log_every);
    return eng; // move-safe: structure is heap-stable, refs survive the move
  };

  if (n_replicas > 1) {
    return run_ensemble(
        [&](std::size_t ri) {
          return make_engine(seed + static_cast<std::uint32_t>(ri * 17u));
        },
        n_replicas, n_steps);
  }
  Engine eng = make_engine(seed);
  eng.run(n_steps);
  return eng;
}

// Write the best structure and also emit bestsqs.out (str.out) plus cross-check
// its corrdump-recomputed correlations against the search — a direct oracle.
void report_result(const Engine &best_engine, const McsqsProblem &prob,
                   const po::variables_map &vm, const std::string &out_path) {
  const auto &best = best_engine.best_structure();
  write_sqs_pdb(best, out_path);
  std::cout << "Best SQS written to " << out_path
            << "  best error: " << best_engine.best_error() << "\n";

  try {
    const std::filesystem::path bestsqs =
        std::filesystem::path(out_path).replace_extension(".out");
    atat::write_str_out(bestsqs, prob.enumerated.axes, prob.enumerated.supercell,
                        prob.enumerated.frac_positions, best.elements);
    std::cout << "bestsqs (ATAT str.out): " << bestsqs << "\n";
    const auto bestcorr = atat::corrdump_correlations(
        resolve_corrdump(vm), prob.lattice_path, bestsqs,
        prob.clusters_out_path,
        std::filesystem::temp_directory_path() / "mcsqs_rmc_oracle");
    std::cout << "corrdump bestcorr:";
    for (const double c : bestcorr) {
      std::cout << " " << c;
    }
    std::cout << "\n";
  } catch (const std::exception &e) {
    std::cerr << "(bestcorr oracle skipped: " << e.what() << ")\n";
  }
}

po::options_description make_options_description() {
  po::options_description desc("mcsqs_rmc — advanced SQS search");
  // clang-format off
  desc.add_options()
    ("help,h", "Show this help")
    ("lattice,L", po::value<std::string>(),
       "ATAT rndstr.in primitive lattice — enables the corrdump pipeline")
    ("supercell", po::value<std::string>()->default_value("2 2 2"),
       "Supercell for --lattice: 'n' (cubic) or 'nx ny nz'")
    ("d2", po::value<double>()->default_value(0.0), "max pair diameter (corrdump -2)")
    ("d3", po::value<double>()->default_value(0.0), "max triplet diameter (-3)")
    ("d4", po::value<double>()->default_value(0.0), "max quadruplet diameter (-4)")
    ("corrdump", po::value<std::string>()->default_value(""),
       "Path to corrdump (default: the vendored build)")
    ("steps,n", po::value<std::uint64_t>()->default_value(500000),
       "MC steps per replica")
    ("replicas,r", po::value<std::size_t>()->default_value(1),
       "Parallel island-model replicas (>1 enables ensemble)")
    ("seed", po::value<std::uint32_t>()->default_value(42), "RNG seed")
    ("out,o", po::value<std::string>()->default_value("bestsqs.pdb"),
       "Output SQS PDB")
    ("log-every,l", po::value<std::uint64_t>()->default_value(10000),
       "Print progress every N steps")
    ("sampler", po::value<std::string>()->default_value("anneal"),
       "Acceptance policy: greedy | metropolis | anneal (default: anneal, "
       "mirrors ATAT mcsqs simulated annealing)")
    ("T0", po::value<double>()->default_value(1.0),
       "Temperature (metropolis: fixed T; anneal: initial T)")
    ("cooling", po::value<double>()->default_value(0.9),
       "Annealing geometric cooling factor in (0,1)")
    ("cool-interval", po::value<std::uint64_t>()->default_value(0),
       "Steps between annealing cooling updates (0 = steps/20)");
  // clang-format on
  return desc;
}

} // namespace

McsqsRunner::McsqsRunner() : options_(make_options_description()) {}

int McsqsRunner::run(int argc, char **argv) {
  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, options_), vm);
    if (vm.count("help")) {
      std::cout << options_ << "\n";
      return 0;
    }
    po::notify(vm);
  } catch (const po::error &e) {
    std::cerr << "Error: " << e.what() << "\n" << options_ << "\n";
    return 1;
  }

  const std::uint64_t n_steps = vm["steps"].as<std::uint64_t>();
  const std::size_t n_replicas = vm["replicas"].as<std::size_t>();
  const std::uint32_t seed = vm["seed"].as<std::uint32_t>();
  const auto out_path = vm["out"].as<std::string>();
  const std::uint64_t log_every = vm["log-every"].as<std::uint64_t>();

  try {
    const SamplerSettings sampler = parse_sampler_settings(vm, n_steps);
    const McsqsProblem problem = resolve_problem(vm, seed);

    std::cout << "Sites: " << problem.structure.size()
              << "  Sublattices: " << problem.sublattices.size()
              << "  Replicas: " << n_replicas << "  Sampler: " << sampler.kind
              << "\n";

    const Engine best_engine =
        run_search(problem, sampler, seed, n_replicas, n_steps, log_every);
    report_result(best_engine, problem, vm, out_path);
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}

} // namespace RMC
