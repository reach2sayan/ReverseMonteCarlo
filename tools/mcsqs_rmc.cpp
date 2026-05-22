// mcsqs_rmc — Advanced SQS search using the RMC engine.
//
// Replaces ATAT's mcsqs MC loop with:
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
#include <RMC/selectors/SmartRandomSelector.hpp>

#include <boost/program_options.hpp>

#include <fstream>
#include <iomanip>
#include <iostream>
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
    if (cur && !cur->instances.empty())
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
    RMC::ClusterInstance inst;
    ls.clear();
    ls.str(line);
    std::size_t idx;
    while (ls >> idx)
      inst.sites.push_back(idx);
    if (!inst.sites.empty())
      cur->instances.push_back(std::move(inst));
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

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char *argv[]) {
  po::options_description desc("mcsqs_rmc — advanced SQS search");
  // clang-format off
  desc.add_options()
    ("help,h", "Show this help")
    ("structure,s", po::value<std::string>()->required(),
       "Input structure PDB file (fixed lattice sites)")
    ("clusters,c", po::value<std::string>()->required(),
       "Cluster orbit file (see header for format)")
    ("species,S", po::value<std::string>()->required(),
       "Species occupation map, e.g. Cu:+1,Au:-1")
    ("steps,n", po::value<std::uint64_t>()->default_value(500000),
       "MC steps per replica")
    ("replicas,r", po::value<std::size_t>()->default_value(1),
       "Parallel island-model replicas (>1 enables ensemble)")
    ("tol,t", po::value<double>()->default_value(1e-4),
       "Stop when total correlation error < tol")
    ("seed", po::value<std::uint32_t>()->default_value(42), "RNG seed")
    ("out,o", po::value<std::string>()->default_value("bestsqs.pdb"),
       "Output SQS PDB")
    ("log-every,l", po::value<std::uint64_t>()->default_value(10000),
       "Print progress every N steps");
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

  const auto struct_path = vm["structure"].as<std::string>();
  const auto clus_path = vm["clusters"].as<std::string>();
  const auto species_str = vm["species"].as<std::string>();
  const std::uint64_t n_steps = vm["steps"].as<std::uint64_t>();
  const std::size_t n_replicas = vm["replicas"].as<std::size_t>();
  const std::uint32_t seed = vm["seed"].as<std::uint32_t>();
  const auto out_path = vm["out"].as<std::string>();
  const std::uint64_t log_ev = vm["log-every"].as<std::uint64_t>();

  // Load inputs.
  auto structure = [&] {
    auto r = RMC::io::read_pdb(struct_path);
    if (!r)
      throw std::runtime_error("Failed to read PDB: " + struct_path);
    return std::move(*r);
  }();

  const auto species_map = parse_species(species_str);
  const auto orbits = load_clusters(clus_path);
  const auto sublattices = build_sublattices(structure);

  std::cout << "Sites: " << structure.size()
            << "  Cluster orbits: " << orbits.size()
            << "  Sublattices: " << sublattices.size()
            << "  Replicas: " << n_replicas << "\n";

  // Factory: builds one Engine instance (called once per replica).
  auto make_engine = [&](std::uint32_t rseed) {
    RMC::Engine eng{structure, RMC::InfiniteBC{}};

    // Add the cluster correlation constraint.
    eng.add_constraint(RMC::IConstraint{RMC::ClusterCorrelationConstraint{
        eng.structure(), species_map, orbits}});

    // One group per site; generator performs sublattice-aware species swap.
    RMC::SpeciesSwapGenerator gen{eng.structure(), sublattices, rseed};
    for (std::size_t i = 0; i < eng.structure().size(); ++i) {
      RMC::Group g;
      g.name = "site_" + std::to_string(i);
      g.indices = {i};
      g.generator = RMC::MoveGenerator{gen}; // each group shares same gen
      eng.add_group(std::move(g));
    }

    eng.set_selector(RMC::GroupSelector{
        RMC::SmartRandomSelector{static_cast<double>(rseed) + 1.0}});
    eng.set_step_callback(
        [log_ev](std::uint64_t total, std::uint64_t accepted,
                 std::uint64_t /*tried*/, double err,
                 const RMC::AtomicStructure &) {
          if (total % log_ev == 0)
            std::cout << "  step " << total << "  accepted " << accepted
                      << "  err " << std::scientific << err << "\n";
        },
        log_ev);

    return eng;
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

  const auto &best = best_engine.structure();
  write_pdb(best, out_path);
  std::cout << "Best SQS written to " << out_path
            << "  final error: " << best_engine.stats().last_total_err << "\n";
  return 0;
}
