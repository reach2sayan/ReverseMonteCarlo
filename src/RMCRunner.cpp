#include <RMC/RMCRunner.hpp>

#include <RMC/Ensemble.hpp>
#include <RMC/analysis/AngularDistribution.hpp>
#include <RMC/analysis/RadialDistribution.hpp>
#include <RMC/constraints/AngularDistributionConstraint.hpp>
#include <RMC/constraints/PairDistributionConstraint.hpp>
#include <RMC/constraints/StructureFactorConstraint.hpp>
#include <RMC/core/RandomStructure.hpp>
#include <RMC/io/DataReader.hpp>
#include <RMC/io/LammpsReader.hpp>
#include <RMC/io/PdbReader.hpp>
#include <RMC/io/VaspReader.hpp>
#include <RMC/selectors/SmartRandomSelector.hpp>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/describe/enumerators.hpp>
#include <boost/leaf.hpp>
#include <boost/mp11/algorithm.hpp>

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <filesystem>
#include <iostream>
#include <print>
#include <ranges>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace po = boost::program_options;
namespace leaf = boost::leaf;

namespace RMC {

// Program options reads a MoveGenKind by enumerator name, case-insensitively
// ("langevin"). Found by ADL, so it lives in RMC, not the anonymous namespace.
std::istream &operator>>(std::istream &is, MoveGenKind &kind) {
  std::string name;
  is >> name;
  bool found = false;
  boost::mp11::mp_for_each<boost::describe::describe_enumerators<MoveGenKind>>(
      [&](auto e) {
        if (boost::algorithm::iequals(name, e.name)) {
          kind = e.value;
          found = true;
        }
      });
  if (!found) {
    is.setstate(std::ios::failbit);
  }
  return is;
}

namespace {

// The input-structure options: whichever one is supplied fixes the reader.
constexpr std::array<std::pair<std::string RMCConfig::*, io::StructFormat>, 3>
    kInputs{{
        {&RMCConfig::pdb_path, io::StructFormat::Pdb},
        {&RMCConfig::lammps_path, io::StructFormat::Lammps},
        {&RMCConfig::vasp_path, io::StructFormat::Vasp},
    }};

// Split a whitespace-separated string into tokens of type T (e.g. the
// "Zr Cu Ag" element legends and "50 50" count lists from the CLI).
template <typename T> std::vector<T> parse_tokens(const std::string &text) {
  std::istringstream is(text);
  return std::ranges::istream_view<T>(is) | std::ranges::to<std::vector>();
}

// --box overrides whatever cell the input implied (incl. the LAMMPS/VASP cell).
Result<void> apply_box_override(const std::string &box, BoundaryConditions &bc) {
  if (box.empty()) {
    return {};
  }
  if (box == "inf") {
    bc = InfiniteBC(1.0);
    spdlog::info("Box: infinite (non-periodic)");
    return {};
  }
  const auto abc = parse_tokens<double>(box);
  if (abc.size() != 3) {
    return leaf::new_error(std::string{"--box expects 'a b c' or 'inf'"});
  }
  const mat3_t cell = vec3_t(abc[0], abc[1], abc[2]).asDiagonal();
  bc = PeriodicBC(cell);
  spdlog::info("Periodic box: {} x {} x {}", abc[0], abc[1], abc[2]);
  return {};
}

// One experimental target: label, config field naming its file, reader, and
// destination field. S(Q)/G(r) are two-column; ADF is multi-column.
struct ExperimentalTarget {
  std::string_view label;
  std::string RMCConfig::*path;
  Result<mat_t> (*read)(const std::filesystem::path &);
  std::optional<mat_t> ExperimentalData::*field;
};

// The experimental-target table — the single place to register a new target.
constexpr std::array<ExperimentalTarget, 3> kExperimentalTargets{{
    {"PairDistribution", &RMCConfig::pdf_path, &io::read_xy_data,
     &ExperimentalData::pdf},
    {"StructureFactor", &RMCConfig::sq_path, &io::read_xy_data,
     &ExperimentalData::sq},
    {"AngularDistribution", &RMCConfig::adf_path, &io::read_columns,
     &ExperimentalData::adf},
}};

template <CConstraint Constraint, typename Configure>
void attach_constraint(Engine &engine, const mat_t &experimental,
                       Configure &&configure) {
  Constraint c;
  c.set_experimental_data(experimental);
  configure(c);
  c.set_elements(engine.structure().elements);
  c.initialise();
  engine.add_constraint(std::move(c));
}

} // namespace

Result<LoadedStructure> load_structure(const RMCConfig &cfg) {
  auto given = kInputs | std::views::filter([&](const auto &in) {
                 return !(cfg.*in.first).empty();
               });
  if (std::ranges::distance(given) != 1) {
    return leaf::new_error(std::string{
        "Provide exactly one input structure (pdb, lammps or vasp)"});
  }
  const auto &[path, fmt] = *given.begin();
  BOOST_LEAF_AUTO(loaded, io::read_structure(cfg.*path, fmt, InfiniteBC(1.0),
                                             cfg.lammps_types));
  spdlog::info("Loaded {} atoms from {}", loaded.structure.size(), cfg.*path);
  BOOST_LEAF_CHECK(apply_box_override(cfg.box_override, loaded.bc));
  return loaded;
}

Result<ExperimentalData> load_experimental_data(const RMCConfig &cfg) {
  ExperimentalData d;
  for (const ExperimentalTarget &t : kExperimentalTargets) {
    if (const std::string &path = cfg.*t.path; !path.empty()) {
      BOOST_LEAF_AUTO(x, t.read(path));
      d.*t.field = std::move(x);
      spdlog::info("Loaded {} data", t.label);
    }
  }
  return d;
}

void attach_constraints(Engine &engine, const ExperimentalData &data,
                        const RMCConfig &cfg) {
  if (data.pdf) {
    attach_constraint<PairDistributionConstraint>(
        engine, *data.pdf, [&](auto &c) { c.set_number_density(cfg.rho0); });
  }
  if (data.sq) {
    attach_constraint<StructureFactorConstraint>(
        engine, *data.sq, [&](auto &c) { c.set_number_density(cfg.rho0); });
  }
  if (data.adf) {
    attach_constraint<AngularDistributionConstraint>(
        engine, *data.adf, [&](auto &c) {
          c.set_cutoff(cfg.adf_cutoff);
          c.set_smoothing(cfg.adf_smooth);
        });
  }
}

Engine build_engine(const LoadedStructure &loaded, const ExperimentalData &data,
                    const RMCConfig &cfg) {
  Engine engine(loaded.structure, loaded.bc);
  engine.build_atomic_groups(cfg.group_min_amp, cfg.group_max_amp, cfg.seed);
  if (cfg.use_smart) {
    engine.set_selector(SmartRandomSelector{1.1, cfg.seed});
  }
  attach_constraints(engine, data, cfg);
  return engine;
}

Result<Engine> build_engine(const RMCConfig &cfg) {
  BOOST_LEAF_AUTO(loaded, load_structure(cfg));
  BOOST_LEAF_AUTO(data, load_experimental_data(cfg));
  return build_engine(loaded, data, cfg);
}

void apply_move_generator(Engine &engine, const RMCConfig &cfg) {
  switch (cfg.move_gen) {
  case MoveGenKind::Langevin:
    engine.build_langevin_groups(cfg.move_step, cfg.seed);
    break;
  case MoveGenKind::Leapfrog:
    engine.build_leapfrog_groups(cfg.move_step, /*n_steps=*/10, cfg.seed);
    break;
  case MoveGenKind::Random:
    break; // build_engine already created the random-walk groups
  }
}

mat3_t periodic_box_or_zero(const BoundaryConditions &bc) {
  return bc.periodic() ? bc.box() : mat3_t::Zero().eval();
}

// Write a structure, picking the format from the output path's extension
// (.vasp/.poscar → VASP, .lammps/.lmp/.data → LAMMPS, else PDB).
Result<void> write_structure_by_ext(const AtomicStructure &s, const mat3_t &box,
                                    const std::string &path) {
  // Unknown extensions fall back to PDB.
  const auto fmt =
      io::classify_structure_format(path).value_or(io::StructFormat::Pdb);
  if (fmt == io::StructFormat::Vasp) {
    return io::write_vasp(s, box, path);
  }
  if (fmt == io::StructFormat::Lammps) {
    // Re-center into [-L/2, L/2) and wrap atoms (the engine wraps into [0,L) and
    // PeriodicBC drops the origin). Orthogonal cells; any tilt passes through.
    AtomicStructure centered = s;
    const vec3_t half(0.5 * box(0, 0), 0.5 * box(1, 1), 0.5 * box(2, 2));
    // Per-axis wrap into [-L/2, L/2): x -= L·floor((x + L/2)/L).
    for (int d = 0; d < 3; ++d) {
      const double L = box(d, d);
      if (L > 0.0) {
        auto col = centered.coordinates.col(d).array();
        col -= L * ((col + half[d]) / L).floor();
      }
    }
    return io::write_lammps_data(centered, box, path, -half);
  }
  return io::write_pdb(s, path);
}

int run_cli(int argc, char **argv, const po::options_description &options,
            const std::function<int(const po::variables_map &)> &body) {
  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, options), vm);
    if (vm.count("help")) {
      std::cout << options << "\n";
      return 0;
    }
    po::notify(vm);
  } catch (const po::error &e) {
    std::cerr << "Error: " << e.what() << "\n" << options << "\n";
    return 1;
  }
  return body(vm);
}

// =====================================================================
// Command-line driver (RMC_run).
// =====================================================================

namespace {

void configure_logging(bool verbose) {
  // One thread-safe (_mt) colour sink as the default logger, shared by
  // parallel replicas; registered once per process.
  if (!spdlog::get("rmc")) {
    spdlog::set_default_logger(spdlog::stdout_color_mt("rmc"));
    spdlog::set_pattern("[%^%l%$] %v");
  }
  spdlog::set_level(verbose ? spdlog::level::debug : spdlog::level::info);
}

// State threaded through every command. The structure is loaded lazily on first
// use (and cached), so structure-free commands never read one.
class RMCContext {
public:
  RMCContext(const po::variables_map &vm, const RMCConfig &cfg)
      : vm_(vm), cfg_(cfg) {}
  const po::variables_map &options() const { return vm_; }
  const RMCConfig &config() const { return cfg_; }
  // Load-on-first-use; the pointer stays valid for the RMCContext's lifetime.
  Result<LoadedStructure *> structure() {
    if (!loaded_) {
      BOOST_LEAF_AUTO(ls, load_structure(cfg_));
      loaded_ = std::move(ls);
    }
    return &*loaded_;
  }

private:
  const po::variables_map &vm_;
  const RMCConfig &cfg_;
  std::optional<LoadedStructure> loaded_;
};

// --gen-random: build a random amorphous structure, write it to --out, exit.
Result<int> cmd_gen_random(RMCContext &ctx) {
  const auto &vm = ctx.options();
  const RMCConfig &cfg = ctx.config();
  if (!vm.count("elements") || !vm.count("counts")) {
    return leaf::new_error(
        std::string{"--gen-random requires --elements and --counts"});
  }
  const auto els = parse_tokens<std::string>(vm["elements"].as<std::string>());
  const auto cnts = parse_tokens<std::size_t>(vm["counts"].as<std::string>());
  BOOST_LEAF_AUTO(gen, make_random_amorphous(
                           els, cnts, vm["spacing"].as<double>(), cfg.seed));
  BOOST_LEAF_CHECK(write_structure_by_ext(gen.structure, gen.box, cfg.out_path));
  spdlog::info("Generated {} atoms; wrote {}", gen.structure.size(),
               cfg.out_path);
  return 0;
}

// --gr: compute the pair distribution g(r) and exit.
Result<int> cmd_compute_gr(RMCContext &ctx) {
  const auto &vm = ctx.options();
  BOOST_LEAF_AUTO(in, ctx.structure());
  analysis::GrParams gp;
  gp.r_min = vm["rmin"].as<double>();
  gp.r_max = vm["rmax"].as<double>();
  gp.n_bins = static_cast<int>(vm["nbins"].as<std::size_t>());
  const auto &s = in->structure;
  BOOST_LEAF_AUTO(g, analysis::compute_gr(s.coordinates, in->bc, s.elements, gp));
  const auto &out = vm["gr-out"].as<std::string>();
  BOOST_LEAF_CHECK(analysis::write_gr(g, out));
  spdlog::info("Wrote g(r) ({} partials) to {}", g.partials.size(), out);
  return 0;
}

// --adf-compute: compute the angular distribution function and exit.
Result<int> cmd_compute_adf(RMCContext &ctx) {
  const auto &vm = ctx.options();
  BOOST_LEAF_AUTO(in, ctx.structure());
  analysis::AdfParams ap;
  ap.max_dis = ctx.config().adf_cutoff;
  ap.n_bins = static_cast<int>(vm["nbins"].as<std::size_t>());
  ap.smooth_range = ctx.config().adf_smooth;
  const auto &s = in->structure;
  BOOST_LEAF_AUTO(a,
                  analysis::compute_adf(s.coordinates, in->bc, s.elements, ap));
  const auto &out = vm["adf-out"].as<std::string>();
  BOOST_LEAF_CHECK(analysis::write_adf(a, out));
  spdlog::info("Wrote ADF ({} triplets) to {}", a.partials.size(), out);
  return 0;
}

double acceptance_pct(std::uint64_t accepted, std::uint64_t tried) {
  return tried > 0 ? 100.0 * static_cast<double>(accepted) /
                         static_cast<double>(tried)
                   : 0.0;
}

// Periodic progress line for a single run.
void log_progress(std::uint64_t step, std::uint64_t acc, std::uint64_t tried,
                  double chi2, const AtomicStructure &) {
  spdlog::info("Step {}  acceptance={:.1f}%  chi2={}", step,
               acceptance_pct(acc, tried), chi2);
}

// Run the refinement: an ensemble of replicas (best chi2 wins) when
// n_ensemble > 1, otherwise a single run with checkpoint + progress callback.
template <typename Factory, typename Prepare>
Engine run_refinement(const RMCConfig &cfg, Factory &&make_engine,
                      Prepare &&prepare, std::size_t n_ensemble) {
  if (n_ensemble > 1) {
    spdlog::info("Ensemble: running {} replicas in parallel", n_ensemble);
    return run_ensemble(make_engine, n_ensemble, cfg.steps, /*tbb=*/0, prepare);
  }
  Engine e = make_engine(0);
  prepare(e); // bind gradient generators to e's final location, if requested
  if (!cfg.checkpoint_path.empty()) {
    e.set_checkpoint(cfg.checkpoint_path);
  }
  e.set_step_callback(log_progress, cfg.log_every);
  spdlog::info("Starting {} steps", cfg.steps);
  e.run(cfg.steps);
  return e;
}

// Default command: refine the input structure against the experimental targets.
Result<int> cmd_refine(RMCContext &ctx) {
  const RMCConfig &cfg = ctx.config();
  // Load structure + experimental data once; the ensemble factory reuses them.
  BOOST_LEAF_AUTO(in, ctx.structure());
  BOOST_LEAF_AUTO(data, load_experimental_data(cfg));

  // Fresh engine per replica with a per-replica seed offset.
  auto make_engine = [&](std::size_t replica) {
    RMCConfig c = cfg;
    c.seed += static_cast<std::uint32_t>(replica);
    return build_engine(*in, data, c);
  };
  // Applied in each engine's final location (gradient generators bind to
  // engine.constraints(), which the build/ensemble moves would invalidate).
  auto prepare = [&](Engine &e) { apply_move_generator(e, cfg); };
  Engine engine = run_refinement(cfg, make_engine, prepare,
                                 ctx.options()["ensemble"].as<std::size_t>());

  BOOST_LEAF_CHECK(write_structure_by_ext(
      engine.structure(), periodic_box_or_zero(in->bc), cfg.out_path));
  spdlog::info("Wrote refined structure to {}", cfg.out_path);

  const auto st = engine.stats();
  std::println("Done. Accepted {} / {} moves ({:.1f}%)  final chi2={:.1f}",
               st.steps_accepted, st.steps_tried,
               acceptance_pct(st.steps_accepted, st.steps_tried),
               st.last_total_err);
  return 0;
}

// Exit-early commands, each selected by the bool flag of the same name; refine
// is the default.
struct RMCCommand {
  const char *flag;
  Result<int> (*run)(RMCContext &);
};
constexpr std::array<RMCCommand, 3> kCommands{{
    {"gen-random", cmd_gen_random},
    {"gr", cmd_compute_gr},
    {"adf-compute", cmd_compute_adf},
}};

Result<int> dispatch(const po::variables_map &vm, const RMCConfig &cfg) {
  RMCContext ctx(vm, cfg);
  const auto cmd = std::ranges::find_if(
      kCommands, [&](const RMCCommand &c) { return vm[c.flag].as<bool>(); });
  return cmd != kCommands.end() ? cmd->run(ctx) : cmd_refine(ctx);
}

// Build the command-line option schema; config knobs bind straight into cfg.
po::options_description make_options_description(RMCConfig &cfg) {
  const auto types = [&cfg](const std::string &s) {
    cfg.lammps_types = parse_tokens<std::string>(s);
  };
  po::options_description desc(
      "RMC_run — Reverse Monte Carlo structural refinement");
  // clang-format off
  desc.add_options()
    ("help,h", "Show this help")
    ("pdb,p", po::value(&cfg.pdb_path), "Input PDB file")
    ("lammps,l", po::value(&cfg.lammps_path),
       "Input LAMMPS data file (atom_style atomic); supplies the periodic box")
    ("types,t", po::value<std::string>()->notifier(types),
       "Element symbols for LAMMPS atom types, in order, e.g. 'Zr Cu Ag'")
    ("pdf,d", po::value(&cfg.pdf_path), "Experimental G(r) data file")
    ("sq,q", po::value(&cfg.sq_path), "Experimental S(Q) data file")
    ("steps,n", po::value(&cfg.steps)->default_value(cfg.steps), "MC steps")
    ("ensemble,e", po::value<std::size_t>()->default_value(1),
       "Number of independent replicas to run in parallel; best chi2 wins")
    ("rho0", po::value(&cfg.rho0)->default_value(cfg.rho0),
       "Number density (atoms/Å³)")
    ("seed", po::value(&cfg.seed)->default_value(cfg.seed), "RNG seed")
    ("out,o", po::value(&cfg.out_path)->default_value(cfg.out_path),
       "Output PDB")
    ("checkpoint,c", po::value(&cfg.checkpoint_path), "Checkpoint file path")
    ("box", po::value(&cfg.box_override),
       "Box vectors: 'a b c' for orthogonal periodic or 'inf' for infinite")
    ("smart", po::bool_switch(&cfg.use_smart), "Use smart adaptive selector")
    ("move-gen", po::value(&cfg.move_gen)->default_value(cfg.move_gen, "random"),
       "Move proposer: 'random' (classic walk), 'langevin' (MALA) or 'leapfrog' "
       "(HMC) — the gradient movers steer atoms along −∇χ² toward the target")
    ("step", po::value(&cfg.move_step)->default_value(cfg.move_step),
       "Gradient step ε (Å) for --move-gen langevin/leapfrog")
    ("gr", po::bool_switch(),
       "Compute g(r) (total + partials) from the input structure and exit; "
       "no MC is run")
    ("gr-out", po::value<std::string>()->default_value("gr.dat"),
       "Output path for g(r) (used with --gr)")
    ("rmin", po::value<double>()->default_value(0.0), "g(r) minimum radius (Å)")
    ("rmax", po::value<double>()->default_value(10.0), "g(r) maximum radius (Å)")
    ("nbins", po::value<std::size_t>()->default_value(200),
       "g(r) / ADF number of bins")
    ("vasp", po::value(&cfg.vasp_path),
       "Input VASP POSCAR/CONTCAR; supplies the periodic cell")
    ("adf,a", po::value(&cfg.adf_path),
       "Experimental ADF (bond-angle distribution) target file")
    ("adf-cutoff", po::value(&cfg.adf_cutoff)->default_value(cfg.adf_cutoff),
       "ADF bond cutoff (Å)")
    ("adf-smooth", po::value(&cfg.adf_smooth)->default_value(cfg.adf_smooth),
       "ADF boxcar smoothing half-width (0 disables)")
    ("adf-compute", po::bool_switch(),
       "Compute the ADF (total + partials) from the input structure and exit; "
       "no MC is run")
    ("adf-out", po::value<std::string>()->default_value("adf.dat"),
       "Output path for the ADF (used with --adf-compute)")
    ("gen-random", po::bool_switch(),
       "Generate a random amorphous structure, write it to --out, and exit")
    ("elements", po::value<std::string>(),
       "Element symbols for --gen-random, e.g. 'Zr Cu'")
    ("counts", po::value<std::string>(),
       "Atom count per element for --gen-random, e.g. '50 50'")
    ("spacing", po::value<double>()->default_value(3.0),
       "Grid spacing (Å) for --gen-random")
    ("verbose,v", po::bool_switch(), "Verbose logging");
  // clang-format on
  return desc;
}

} // namespace

RMCRunner::RMCRunner() : options_(make_options_description(cfg_)) {}

int RMCRunner::run(int argc, char **argv) {
  return run_cli(argc, argv, options_, [this](const po::variables_map &vm) {
    configure_logging(vm["verbose"].as<bool>());
    return leaf::try_handle_all(
        [&]() -> leaf::result<int> { return dispatch(vm, cfg_); },
        [](const std::string &msg) {
          std::cerr << "Error: " << msg << "\n";
          return 1;
        },
        [](const leaf::error_info &unmatched) {
          std::cerr << "Unexpected error: " << unmatched << "\n";
          return 1;
        });
  });
}

} // namespace RMC
