#include "McsqsRunner.hpp"

#include "AtatFormats.hpp"
#include "SeitzClusters.hpp"

#include <RMC/Engine.hpp>
#include <RMC/Ensemble.hpp>
#include <RMC/RMCRunner.hpp>
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

#include <boost/describe/enum.hpp>
#include <boost/describe/enum_from_string.hpp>
#include <boost/describe/enum_to_string.hpp>
#include <boost/parser/parser.hpp>
#include <boost/program_options.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
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

// Parse "--supercell": 'n' (cubic), 'nx ny nz' (diagonal) or nine integers
// (row-major matrix).
Eigen::Matrix3i build_sc_matrix(const std::string &s) {
  namespace bp = boost::parser;
  const auto v = bp::parse(s, +bp::int_, bp::ws).value_or(std::vector<int>{});
  Eigen::Matrix3i m;
  switch (v.size()) {
  case 1:
    m = Eigen::Vector3i::Constant(v[0]).asDiagonal();
    break;
  case 3:
    m = Eigen::Vector3i::Map(v.data()).asDiagonal();
    break;
  case 9:
    m = Eigen::Map<const Eigen::Matrix<int, 3, 3, Eigen::RowMajor>>(v.data());
    break;
  default:
    throw std::runtime_error("--supercell: expected 1, 3, or 9 integers");
  }
  if (m.determinant() == 0) {
    throw std::runtime_error("--supercell: singular matrix");
  }
  return m;
}

// Acceptance policy, named as on the command line.
enum class SamplerKind { greedy, metropolis, anneal };
BOOST_DESCRIBE_ENUM(SamplerKind, greedy, metropolis, anneal)

// The acceptance-policy knobs, validated from the CLI.
struct SamplerSettings {
  SamplerKind kind;
  double t0;
  double cooling;
  std::uint64_t cool_interval;
};

// Validate --sampler and capture its tuning knobs. Throws on an unknown policy.
SamplerSettings parse_sampler_settings(const po::variables_map &vm,
                                       std::uint64_t n_steps) {
  const std::string name = vm["sampler"].as<std::string>();
  SamplerKind kind{};
  if (!boost::describe::enum_from_string(name.c_str(), kind)) {
    throw std::runtime_error("unknown --sampler '" + name +
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
  switch (s.kind) {
  case SamplerKind::greedy:
    return Sampler{GreedySampler{}};
  case SamplerKind::metropolis:
    return Sampler{MetropolisSampler{s.t0}};
  case SamplerKind::anneal:
    break;
  }
  return Sampler{AnnealingSampler{AnnealingSampler::Schedule{
      .t0 = s.t0, .cooling = s.cooling, .interval = s.cool_interval}}};
}

// The resolved SQS problem, held alive for the constraint's references.
struct McsqsProblem {
  atat::EnumeratedSqs sqs;
  std::vector<std::vector<std::size_t>> sublattices;

  void add_constraint(Engine &eng) const {
    eng.add_constraint(Constraint{ClusterCorrelationConstraint{
        eng.structure(), sqs.occ_index, sqs.table, sqs.orbits}});
  }
};

// --dN options: the largest cluster diameter per body order.
constexpr std::array<std::pair<int, const char *>, 3> kDiameterOptions{
    {{2, "d2"}, {3, "d3"}, {4, "d4"}}};

// Enumerate the SQS problem for --lattice on seitz.
McsqsProblem resolve_problem(const po::variables_map &vm, std::uint32_t seed) {
  if (vm.count("lattice") == 0 || vm["d2"].as<double>() <= 0.0) {
    throw std::runtime_error(
        "--lattice (ATAT rndstr.in) and --d2 (max pair diameter) are required");
  }
  atat::Diameters diameters;
  for (const auto &[body, option] : kDiameterOptions) {
    if (const double d = vm[option].as<double>(); d > 0.0) {
      diameters[body] = d;
    }
  }
  const auto lat = atat::parse_lattice(vm["lattice"].as<std::string>());
  McsqsProblem prob{
      atat::enumerate(lat, build_sc_matrix(vm["supercell"].as<std::string>()),
                      diameters, seed),
      {}};
  prob.sublattices = build_sublattices(prob.sqs.structure);
  std::cout << "Lattice sites: " << lat.sites.size()
            << "  Species: " << lat.labels.size()
            << "  Supercell atoms: " << prob.sqs.structure.size()
            << "  Orbits: " << prob.sqs.orbits.size() << "\n";
  return prob;
}

// Search

// Configure engine in place. Constraint/generator hold refs into eng.structure(),
// which stays heap-stable across the later move.
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

  // Default bias factor; seeded apart from the swap generator's rseed stream.
  eng.set_selector(GroupSelector{SmartRandomSelector{1.1, rseed + 1}});
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
    Engine eng{prob.sqs.structure, InfiniteBC{}};
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

// Write the best SQS as PDB plus <out>.out in ATAT's str.out format.
void report_result(const Engine &best_engine, const McsqsProblem &prob,
                   const std::string &out_path) {
  const auto &best = best_engine.best_structure();
  if (!io::write_pdb(best, out_path)) {
    throw std::runtime_error("Cannot write: " + out_path);
  }
  const auto bestsqs = std::filesystem::path(out_path).replace_extension(".out");
  atat::write_str_out(bestsqs, prob.sqs.axes, prob.sqs.supercell,
                      prob.sqs.frac_positions, best.elements);
  std::cout << "Best SQS written to " << out_path << " and "
            << bestsqs.string() << "  best error: " << best_engine.best_error()
            << "\n";
}

po::options_description make_options_description() {
  po::options_description desc("mcsqs_rmc — advanced SQS search");
  // clang-format off
  desc.add_options()
    ("help,h", "Show this help")
    ("lattice,L", po::value<std::string>(),
       "ATAT rndstr.in primitive lattice (required)")
    ("supercell", po::value<std::string>()->default_value("2 2 2"),
       "Supercell of the primitive cell: 'n', 'nx ny nz' or nine integers")
    ("d2", po::value<double>()->default_value(0.0),
       "max pair diameter (Å, required)")
    ("d3", po::value<double>()->default_value(0.0), "max triplet diameter (Å)")
    ("d4", po::value<double>()->default_value(0.0), "max quadruplet diameter (Å)")
    ("steps,n", po::value<std::uint64_t>()->default_value(500000),
       "MC steps per replica")
    ("replicas,r", po::value<std::size_t>()->default_value(1),
       "Parallel island-model replicas (>1 enables ensemble)")
    ("seed", po::value<std::uint32_t>()->default_value(42), "RNG seed")
    ("out,o", po::value<std::string>()->default_value("bestsqs.pdb"),
       "Output SQS PDB (bestsqs.out, ATAT str.out, is written beside it)")
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
  return run_cli(argc, argv, options_, [](const po::variables_map &vm) {
    const std::uint64_t n_steps = vm["steps"].as<std::uint64_t>();
    const std::size_t n_replicas = vm["replicas"].as<std::size_t>();
    const std::uint32_t seed = vm["seed"].as<std::uint32_t>();
    try {
      const SamplerSettings sampler = parse_sampler_settings(vm, n_steps);
      const McsqsProblem problem = resolve_problem(vm, seed);

      std::cout << "Sites: " << problem.sqs.structure.size()
                << "  Sublattices: " << problem.sublattices.size()
                << "  Replicas: " << n_replicas << "  Sampler: "
                << boost::describe::enum_to_string(sampler.kind, "?") << "\n";

      const Engine best_engine =
          run_search(problem, sampler, seed, n_replicas, n_steps,
                     vm["log-every"].as<std::uint64_t>());
      report_result(best_engine, problem, vm["out"].as<std::string>());
    } catch (const std::exception &e) {
      std::cerr << "Error: " << e.what() << "\n";
      return 1;
    }
    return 0;
  });
}

} // namespace RMC
