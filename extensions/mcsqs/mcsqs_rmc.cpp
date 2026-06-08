// mcsqs_rmc — Advanced SQS search using the RMC engine.
//
// Replaces ATAT's mcsqs MC loop with:
//   - Pluggable acceptance policy (greedy | metropolis | anneal); the default
//     AnnealingSampler mirrors ATAT's simulated annealing so the search can
//     escape ordered local minima
//   - Adaptive site selection (SmartRandomSelector)
//   - Batch-buffered MT19937 RNG (RngBuffer)
//   - Optional island-model ensemble parallelism
//
// Workflow:
//   1. Generate cluster definitions with ATAT's corrdump:
//        corrdump -ro -noe -nop -cf=clusters.in < rndstr.in > clusters.out
//   2. Run this tool:
//        mcsqs_rmc --structure rndstr.pdb --clusters clusters.out
//                  --species Cu:+1,Au:-1 --replicas 8
//
// Cluster file format (--clusters):
//   One orbit per block, blank-line separated.
//   First line:  <target_corr> <weight> <n_points>
//   Subsequent:  space-separated site indices (0-based), one instance per line.
//
// Example clusters.txt for a 4-site cell with 2 pair orbits:
//   0.0 1.0 2
//   0 1
//   2 3
//
//   0.0 1.0 2
//   0 2
//   1 3
#include <RMC/Engine.hpp>
#include <RMC/Ensemble.hpp>
#include <RMC/constraints/ClusterCorrelationConstraint.hpp>
#include <RMC/core/BoundaryConditions.hpp>
#include <RMC/core/Group.hpp>
#include <RMC/core/Structure.hpp>
#include <RMC/generators/SpeciesSwap.hpp>
#include <RMC/io/PdbReader.hpp>
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
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace po = boost::program_options;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Parse "Cu:+1.0,Au:-1.0" into a SpeciesMap.
static RMC::ClusterCorrelationConstraint::SpeciesMap
parse_species(const std::string &s) {
  RMC::ClusterCorrelationConstraint::SpeciesMap m;
  std::istringstream ss(s);
  std::string token;
  while (std::getline(ss, token, ',')) {
    const auto colon = token.find(':');
    if (colon == std::string::npos)
      throw std::runtime_error("species: expected 'Elem:value', got: " + token);
    const std::string elem = token.substr(0, colon);
    const double val = std::stod(token.substr(colon + 1));
    m[elem] = val;
  }
  return m;
}

// Parse the cluster file into a vector of ClusterOrbit.
static std::vector<RMC::ClusterOrbit> load_clusters(const std::string &path) {
  std::ifstream f(path);
  if (!f)
    throw std::runtime_error("Cannot open cluster file: " + path);

  std::vector<RMC::ClusterOrbit> orbits;
  std::string line;
  std::optional<RMC::ClusterOrbit> cur;

  auto flush = [&] {
    if (cur && cur->instance_count() != 0)
      orbits.push_back(std::move(*cur));
    cur.reset();
  };

  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') {
      flush();
      continue;
    }
    std::istringstream ls(line);
    // Try reading as header: <target> <weight> <n_points>
    double target, weight;
    int n_pts;
    if (!cur && (ls >> target >> weight >> n_pts)) {
      cur = RMC::ClusterOrbit{};
      cur->target = target;
      cur->weight = weight;
      continue;
    }
    // Otherwise it's a cluster instance: space-separated site indices.
    if (!cur)
      continue;
    ls.clear();
    ls.str(line);
    std::vector<std::size_t> sites;
    std::size_t idx;
    while (ls >> idx)
      sites.push_back(idx);
    if (!sites.empty())
      cur->add_instance(sites);
  }
  flush();
  return orbits;
}

// Build sublattice groups: one sublattice per distinct residue name.
// Falls back to a single sublattice containing all sites.
static std::vector<std::vector<std::size_t>>
build_sublattices(const RMC::AtomicStructure &str) {
  std::unordered_map<std::string, std::vector<std::size_t>> by_residue;
  for (std::size_t i = 0; i < str.size(); ++i)
    by_residue[str.residues.empty() ? "all" : str.residues[i]].push_back(i);
  std::vector<std::vector<std::size_t>> result;
  for (auto &[_, sites] : by_residue)
    result.push_back(std::move(sites));
  return result;
}

// Write a minimal PDB with current element/coords.
static void write_pdb(const RMC::AtomicStructure &str,
                      const std::string &path) {
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
static Eigen::Matrix3i build_sc_matrix(const std::string &s) {
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

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char *argv[]) {
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
    ("structure,s", po::value<std::string>(),
       "[legacy] Input structure PDB file (fixed lattice sites)")
    ("clusters,c", po::value<std::string>(),
       "[legacy] Cluster orbit file (see header for format)")
    ("species,S", po::value<std::string>(),
       "[legacy] Species occupation map, e.g. Cu:+1,Au:-1")
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

  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
      std::cout << desc << "\n";
      return 0;
    }
    po::notify(vm);
  } catch (const po::error &e) {
    std::cerr << "Error: " << e.what() << "\n" << desc << "\n";
    return 1;
  }

  const std::uint64_t n_steps = vm["steps"].as<std::uint64_t>();
  const std::size_t n_replicas = vm["replicas"].as<std::size_t>();
  const std::uint32_t seed = vm["seed"].as<std::uint32_t>();
  const auto out_path = vm["out"].as<std::string>();
  const std::uint64_t log_ev = vm["log-every"].as<std::uint64_t>();

  // Acceptance policy (sampler).
  const std::string sampler_kind = vm["sampler"].as<std::string>();
  if (sampler_kind != "greedy" && sampler_kind != "metropolis" &&
      sampler_kind != "anneal") {
    std::cerr << "Error: unknown --sampler '" << sampler_kind
              << "' (use greedy | metropolis | anneal)\n";
    return 1;
  }
  const double t0 = vm["T0"].as<double>();
  const double cooling = vm["cooling"].as<double>();
  std::uint64_t cool_interval = vm["cool-interval"].as<std::uint64_t>();
  if (cool_interval == 0) {
    cool_interval = std::max<std::uint64_t>(1, n_steps / 20);
  }

  auto make_sampler = [&]() -> RMC::Sampler {
    if (sampler_kind == "greedy") {
      return RMC::Sampler{RMC::GreedySampler{}};
    }
    if (sampler_kind == "metropolis") {
      return RMC::Sampler{RMC::MetropolisSampler{t0}};
    }
    return RMC::Sampler{RMC::AnnealingSampler{RMC::AnnealingSampler::Schedule{
        .t0 = t0, .cooling = cooling, .interval = cool_interval}}};
  };

  // corrdump path: --corrdump > compile-time vendored path > "corrdump" on
  // PATH.
  const auto resolve_corrdump = [&]() -> std::string {
    const std::string opt = vm["corrdump"].as<std::string>();
    if (!opt.empty()) {
      return opt;
    }
#ifdef RMC_CORRDUMP_PATH
    return RMC_CORRDUMP_PATH;
#else
    return "corrdump";
#endif
  };

  // Resolve inputs: ATAT pipeline (--lattice) or legacy (--structure).
  RMC::AtomicStructure structure;
  std::vector<std::vector<std::size_t>> sublattices;
  std::function<void(RMC::Engine &)> add_constraint;
  std::optional<RMC::atat::EnumeratedSqs> enumerated;
  std::string lattice_path;
  std::filesystem::path clusters_out_path;
  // Storage kept alive for the legacy branch's by-reference constraint factory.
  RMC::ClusterCorrelationConstraint::SpeciesMap legacy_species;
  std::vector<RMC::ClusterOrbit> legacy_orbits;

  try {
    if (vm.count("lattice")) {
      lattice_path = vm["lattice"].as<std::string>();
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

      const Eigen::Matrix3i sc =
          build_sc_matrix(vm["supercell"].as<std::string>());
      const auto lat = RMC::atat::parse_lattice(lattice_path);
      const auto workdir =
          std::filesystem::temp_directory_path() / "mcsqs_rmc_clusters";
      const auto cl = RMC::atat::corrdump_generate_clusters(
          resolve_corrdump(), lattice_path, diam, workdir);
      clusters_out_path = cl.clusters_out;
      const auto sym = RMC::atat::parse_sym(cl.sym_out);
      const auto raw = RMC::atat::parse_clusters(cl.clusters_out);
      enumerated = RMC::atat::enumerate(lat, sym, raw, sc, seed);
      structure = enumerated->structure;
      sublattices = build_sublattices(structure);
      add_constraint = [&](RMC::Engine &eng) {
        eng.add_constraint(RMC::Constraint{RMC::ClusterCorrelationConstraint{
            eng.structure(), enumerated->occ_index, enumerated->table,
            enumerated->orbits}});
      };
      std::cout << "Lattice sites: " << lat.sites.size()
                << "  Species: " << lat.labels.size()
                << "  Supercell atoms: " << structure.size()
                << "  Orbits: " << enumerated->orbits.size() << "\n";
    } else {
      if (vm.count("structure") == 0 || vm.count("clusters") == 0 ||
          vm.count("species") == 0) {
        throw std::runtime_error(
            "provide --lattice (ATAT pipeline) OR --structure + --clusters + "
            "--species (legacy)");
      }
      auto r = RMC::io::read_pdb(vm["structure"].as<std::string>());
      if (!r) {
        throw std::runtime_error("Failed to read PDB: " +
                                 vm["structure"].as<std::string>());
      }
      structure = std::move(*r);
      legacy_species = parse_species(vm["species"].as<std::string>());
      legacy_orbits = load_clusters(vm["clusters"].as<std::string>());
      sublattices = build_sublattices(structure);
      add_constraint = [&](RMC::Engine &eng) {
        eng.add_constraint(RMC::Constraint{RMC::ClusterCorrelationConstraint{
            eng.structure(), legacy_species, legacy_orbits}});
      };
    }
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

  std::cout << "Sites: " << structure.size()
            << "  Sublattices: " << sublattices.size()
            << "  Replicas: " << n_replicas << "  Sampler: " << sampler_kind
            << "\n";

  // Configure an engine IN PLACE. The cluster constraint and SpeciesSwap
  // generator hold references into eng.structure(), so the engine must NOT be
  // moved afterwards — hence the in-place ensemble below (a return-by-value
  // factory would move the engine and dangle those references).
  auto configure_engine = [&](RMC::Engine &eng, std::uint32_t rseed) {
    add_constraint(eng);

    // One group per site; generator performs sublattice-aware species swap.
    RMC::SpeciesSwapGenerator gen{eng.structure(), sublattices, rseed};
    for (std::size_t i = 0; i < eng.structure().size(); ++i) {
      RMC::Group g;
      g.name = "site_" + std::to_string(i);
      g.indices = {i};
      g.generator = RMC::MoveGenerator{gen};
      eng.add_group(std::move(g));
    }

    eng.set_selector(RMC::GroupSelector{
        RMC::SmartRandomSelector{static_cast<double>(rseed) + 1.0}});
    eng.set_sampler(make_sampler(), rseed);
    eng.set_track_best(true); // annealing's final state isn't the minimum
    eng.set_step_callback(
        [log_ev](std::uint64_t total, std::uint64_t accepted,
                 std::uint64_t /*tried*/, double err,
                 const RMC::AtomicStructure & /*structure*/) {
          if (total % log_ev == 0)
            std::cout << "  step " << total << "  accepted " << accepted
                      << "  err " << std::scientific << err << "\n";
        },
        log_ev);
  };

  auto make_engine = [&](std::uint32_t rseed) {
    RMC::Engine eng{structure, RMC::InfiniteBC{}};
    configure_engine(eng, rseed);
    return eng; // move-safe: structure is heap-stable, refs survive the move
  };

  RMC::Engine best_engine =
      (n_replicas > 1) ? RMC::run_ensemble(
                             [&](std::size_t ri) {
                               return make_engine(
                                   seed + static_cast<std::uint32_t>(ri * 17u));
                             },
                             n_replicas, n_steps)
                       : [&] {
                           auto eng = make_engine(seed);
                           eng.run(n_steps);
                           return eng;
                         }();

  const auto &best = best_engine.best_structure();
  write_pdb(best, out_path);
  std::cout << "Best SQS written to " << out_path
            << "  best error: " << best_engine.best_error() << "\n";

  // ATAT pipeline: also emit bestsqs.out (str.out) and report
  // corrdump-recomputed correlations of the result — a direct cross-check
  // against the search.
  if (enumerated) {
    try {
      const std::filesystem::path bestsqs =
          std::filesystem::path(out_path).replace_extension(".out");
      RMC::atat::write_str_out(bestsqs, enumerated->axes, enumerated->supercell,
                               enumerated->frac_positions, best.elements);
      std::cout << "bestsqs (ATAT str.out): " << bestsqs << "\n";
      const auto bestcorr = RMC::atat::corrdump_correlations(
          resolve_corrdump(), lattice_path, bestsqs, clusters_out_path,
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
  return 0;
}
