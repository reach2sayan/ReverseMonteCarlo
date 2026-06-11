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

#include <boost/hof/lift.hpp>
#include <boost/hof/match.hpp>
#include <boost/leaf.hpp>
#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/trivial.hpp>

#include <cmath>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace po = boost::program_options;
namespace leaf = boost::leaf;

namespace RMC {
namespace {

// Per-format input descriptors, fused into a variant for std::visit dispatch —
// the value-oriented overload set, now keyed off RMCConfig's input paths.
struct PdbInput {
  std::string path;
};
struct LammpsInput {
  std::string path;
  std::vector<std::string> types; // optional 'Zr Cu Ag' type→element legend
};
struct VaspInput {
  std::string path;
};
using Input = std::variant<PdbInput, LammpsInput, VaspInput>;

// Resolve the one input the config supplies into a typed Input.
Result<Input> select_input(const RMCConfig &cfg) {
  std::optional<Input> chosen;
  int given = 0;
  if (!cfg.pdb_path.empty()) {
    ++given;
    chosen = PdbInput{cfg.pdb_path};
  }
  if (!cfg.lammps_path.empty()) {
    ++given;
    chosen = LammpsInput{cfg.lammps_path, cfg.lammps_types};
  }
  if (!cfg.vasp_path.empty()) {
    ++given;
    chosen = VaspInput{cfg.vasp_path};
  }
  if (given != 1) {
    return leaf::new_error(std::string{
        "Provide exactly one input structure (pdb, lammps or vasp)"});
  }
  return std::move(*chosen);
}

Result<LoadedStructure> load_pdb(const PdbInput &in) {
  BOOST_LEAF_AUTO(ps, io::read_pdb(in.path));
  LoadedStructure out;
  out.structure = std::move(ps);
  BOOST_LOG_TRIVIAL(info) << "Loaded " << out.structure.size() << " atoms";
  return out;
}

Result<LoadedStructure> load_lammps(const LammpsInput &in) {
  BOOST_LEAF_AUTO(data, io::read_lammps_data(in.path, in.types));
  LoadedStructure out;
  out.structure = std::move(data.structure);
  // A LAMMPS data file carries its own cell; --box may override.
  out.bc = data.periodic_bc();
  BOOST_LOG_TRIVIAL(info) << "Loaded " << out.structure.size()
                          << " atoms from LAMMPS data; box " << data.box(0, 0)
                          << " x " << data.box(1, 1) << " x " << data.box(2, 2);
  return out;
}

Result<LoadedStructure> load_vasp(const VaspInput &in) {
  BOOST_LEAF_AUTO(data, io::read_vasp(in.path));
  LoadedStructure out;
  out.structure = std::move(data.structure);
  // A POSCAR carries its own cell; --box may override.
  out.bc = data.periodic_bc();
  BOOST_LOG_TRIVIAL(info) << "Loaded " << out.structure.size()
                          << " atoms from VASP POSCAR; box " << data.box(0, 0)
                          << " x " << data.box(1, 1) << " x " << data.box(2, 2);
  return out;
}

// --box overrides whatever cell the input implied (incl. the LAMMPS/VASP cell).
void apply_box_override(const RMCConfig &cfg, BoundaryConditions &bc) {
  if (!cfg.box_override) {
    return;
  }
  if (*cfg.box_override == "inf") {
    bc = InfiniteBC(1.0);
    BOOST_LOG_TRIVIAL(info) << "Box: infinite (non-periodic)";
    return;
  }
  std::istringstream ss(*cfg.box_override);
  double a, b, c;
  ss >> a >> b >> c;
  mat3_t box = vec3_t(a, b, c).asDiagonal();
  bc = PeriodicBC(box);
  BOOST_LOG_TRIVIAL(info) << "Periodic box: " << a << " x " << b << " x " << c;
}

// One experimental target: a human label, the RMCConfig field naming its file,
// the reader to parse it, and where the parsed matrix lands in ExperimentalData.
// The reader varies — S(Q)/G(r) are two-column, the ADF is multi-column.
struct ExperimentalTarget {
  const char *label;
  std::optional<std::string> RMCConfig::*path;
  std::function<Result<mat_t>(const std::string &)> read;
  std::optional<mat_t> ExperimentalData::*field;
};

// The experimental-target table — the single place to register a new target.
std::vector<ExperimentalTarget> experimental_targets() {
  const auto two_column = [](const std::string &p) {
    return io::read_xy_data(p);
  };
  const auto multi_column = [](const std::string &p) {
    return io::read_columns(p);
  };
  return {
      {"PairDistribution", &RMCConfig::pdf_path, two_column,
       &ExperimentalData::pdf},
      {"StructureFactor", &RMCConfig::sq_path, two_column, &ExperimentalData::sq},
      {"AngularDistribution", &RMCConfig::adf_path, multi_column,
       &ExperimentalData::adf},
  };
}

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
  BOOST_LEAF_AUTO(input, select_input(cfg));
  BOOST_LEAF_AUTO(loaded,
                  std::visit(boost::hof::match(BOOST_HOF_LIFT(load_pdb),
                                               BOOST_HOF_LIFT(load_lammps),
                                               BOOST_HOF_LIFT(load_vasp)),
                             input));
  apply_box_override(cfg, loaded.bc);
  return loaded;
}

Result<ExperimentalData> load_experimental_data(const RMCConfig &cfg) {
  ExperimentalData d;
  for (const ExperimentalTarget &t : experimental_targets()) {
    const std::optional<std::string> &path = cfg.*(t.path);
    if (!path) {
      continue;
    }
    BOOST_LEAF_AUTO(x, t.read(*path));
    (d.*t.field) = std::move(x);
    BOOST_LOG_TRIVIAL(info) << "Loaded " << t.label << " data";
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
  // Engine's ctor takes the structure by value, so this copy gives each replica
  // its own independent structure (the shared `loaded` stays reusable).
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
  mat3_t out = mat3_t::Zero();
  if (const auto *p = std::get_if<PeriodicBC>(&bc)) {
    out = p->box();
  }
  return out;
}

// =====================================================================
// Command-line driver (RMC_run).
// =====================================================================

// Write a structure choosing the format from the output path: a VASP POSCAR for
// .vasp/.poscar (or a POSCAR/CONTCAR name), a LAMMPS data file for
// .lammps/.lmp/.data, otherwise a PDB. The box (from the active periodic cell;
// zero for infinite) is needed by the VASP and LAMMPS writers.
Result<void> write_structure_by_ext(const AtomicStructure &s, const mat3_t &box,
                                    const std::string &path) {
  const std::filesystem::path p(path);
  const std::string ext = p.extension().string();
  const std::string stem = p.filename().string();
  if (ext == ".vasp" || ext == ".poscar" || ext == ".VASP" ||
      stem == "POSCAR" || stem == "CONTCAR") {
    return io::write_vasp(s, box, path);
  }
  if (ext == ".lammps" || ext == ".lmp" || ext == ".data") {
    // The engine wraps atoms into [0,L), and PeriodicBC drops the box origin, so
    // a structure read from a centered cell (xlo=-L/2) would otherwise be
    // written with xlo=0 — the cell appears to jump. Re-center: write the box as
    // [-L/2, L/2) and wrap atoms into it. (Orthogonal cells; any tilt is passed
    // through unwrapped.)
    AtomicStructure centered = s;
    const vec3_t half(0.5 * box(0, 0), 0.5 * box(1, 1), 0.5 * box(2, 2));
    // Wrap each axis into [-L/2, L/2) as a single vectorised column op:
    // x -= L·floor((x + L/2)/L). The d loop is over the three box dims.
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

namespace {

// Split a whitespace-separated string into tokens of type T (e.g. the
// "Zr Cu Ag" element legends and "50 50" count lists from the CLI).
template <typename T> std::vector<T> parse_tokens(const std::string &text) {
  std::vector<T> out;
  std::istringstream is(text);
  for (T v; is >> v;) {
    out.push_back(std::move(v));
  }
  return out;
}

void configure_logging(bool verbose) {
  if (!verbose) {
    boost::log::core::get()->set_filter(boost::log::trivial::severity >=
                                        boost::log::trivial::info);
  }
}

// Map the parsed command line onto the program_options-independent RMCConfig the
// builder consumes. This is the only place the CLI's option names meet the
// builder. Numeric knobs not exposed by the CLI (group amps, log_every) keep
// their RMCConfig defaults, preserving the historical behaviour.
RMCConfig sim_config_from_vm(const po::variables_map &vm) {
  RMCConfig cfg;
  if (vm.count("pdb")) {
    cfg.pdb_path = vm["pdb"].as<std::string>();
  }
  if (vm.count("lammps")) {
    cfg.lammps_path = vm["lammps"].as<std::string>();
  }
  if (vm.count("vasp")) {
    cfg.vasp_path = vm["vasp"].as<std::string>();
  }
  if (vm.count("types")) {
    cfg.lammps_types = parse_tokens<std::string>(vm["types"].as<std::string>());
  }
  if (vm.count("box")) {
    cfg.box_override = vm["box"].as<std::string>();
  }
  if (vm.count("pdf")) {
    cfg.pdf_path = vm["pdf"].as<std::string>();
  }
  if (vm.count("sq")) {
    cfg.sq_path = vm["sq"].as<std::string>();
  }
  if (vm.count("adf")) {
    cfg.adf_path = vm["adf"].as<std::string>();
  }
  if (vm.count("checkpoint")) {
    cfg.checkpoint_path = vm["checkpoint"].as<std::string>();
  }
  cfg.rho0 = vm["rho0"].as<double>();
  cfg.adf_cutoff = vm["adf-cutoff"].as<double>();
  cfg.adf_smooth = vm["adf-smooth"].as<int>();
  cfg.steps = vm["steps"].as<std::uint64_t>();
  cfg.seed = vm["seed"].as<std::uint32_t>();
  cfg.use_smart = vm["smart"].as<bool>();
  const std::string mg = vm["move-gen"].as<std::string>();
  cfg.move_gen = mg == "langevin"  ? MoveGenKind::Langevin
                 : mg == "leapfrog" ? MoveGenKind::Leapfrog
                                    : MoveGenKind::Random;
  cfg.move_step = vm["step"].as<double>();
  cfg.out_path = vm["out"].as<std::string>();
  return cfg;
}

// State threaded through every command. The RMCConfig is derived once from the
// options; the input structure is loaded lazily (via the shared load_structure,
// which also applies --box) the first time a command asks for it, then cached —
// so structure-free commands like --gen-random never read a structure, while the
// structure-consuming commands share a single load.
class RMCContext {
public:
  explicit RMCContext(const po::variables_map &vm)
      : vm_(vm), cfg_(sim_config_from_vm(vm)) {}
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
  RMCConfig cfg_;
  std::optional<LoadedStructure> loaded_;
};

struct RMCCommand {
  std::string_view name;
  std::function<bool(const po::variables_map &)> selected;
  std::function<Result<int>(RMCContext &)> run;
};

// --gen-random: build a random amorphous structure, write it to --out, exit.
Result<int> cmd_gen_random(RMCContext &ctx) {
  const auto &vm = ctx.options();
  if (!vm.count("elements") || !vm.count("counts")) {
    return leaf::new_error(
        std::string{"--gen-random requires --elements and --counts"});
  }
  const auto els = parse_tokens<std::string>(vm["elements"].as<std::string>());
  const auto cnts = parse_tokens<std::size_t>(vm["counts"].as<std::string>());
  BOOST_LEAF_AUTO(gen,
                  make_random_amorphous(els, cnts, vm["spacing"].as<double>(),
                                        vm["seed"].as<std::uint32_t>()));
  const std::string out = vm["out"].as<std::string>();
  BOOST_LEAF_CHECK(write_structure_by_ext(gen.structure, gen.box, out));
  BOOST_LOG_TRIVIAL(info) << "Generated " << gen.structure.size()
                          << " atoms; wrote " << out;
  return 0;
}

// --gr: compute the pair distribution g(r) and exit.
Result<int> cmd_compute_gr(RMCContext &ctx) {
  const auto &vm = ctx.options();
  BOOST_LEAF_AUTO(in, ctx.structure());
  const AtomicStructure &s = in->structure;
  const BoundaryConditions &bc = in->bc;
  analysis::GrParams gp;
  gp.r_min = vm["rmin"].as<double>();
  gp.r_max = vm["rmax"].as<double>();
  gp.n_bins = static_cast<int>(vm["nbins"].as<std::size_t>());
  BOOST_LEAF_AUTO(g, analysis::compute_gr(s.coordinates, bc, s.elements, gp));
  BOOST_LEAF_CHECK(analysis::write_gr(g, vm["gr-out"].as<std::string>()));
  BOOST_LOG_TRIVIAL(info) << "Wrote g(r) (" << g.partials.size()
                          << " partials) to " << vm["gr-out"].as<std::string>();
  return 0;
}

// --adf-compute: compute the angular distribution function and exit.
Result<int> cmd_compute_adf(RMCContext &ctx) {
  const auto &vm = ctx.options();
  BOOST_LEAF_AUTO(in, ctx.structure());
  const AtomicStructure &s = in->structure;
  const BoundaryConditions &bc = in->bc;
  analysis::AdfParams ap;
  ap.max_dis = vm["adf-cutoff"].as<double>();
  ap.n_bins = static_cast<int>(vm["nbins"].as<std::size_t>());
  ap.smooth_range = vm["adf-smooth"].as<int>();
  BOOST_LEAF_AUTO(a, analysis::compute_adf(s.coordinates, bc, s.elements, ap));
  BOOST_LEAF_CHECK(analysis::write_adf(a, vm["adf-out"].as<std::string>()));
  BOOST_LOG_TRIVIAL(info) << "Wrote ADF (" << a.partials.size()
                          << " triplets) to " << vm["adf-out"].as<std::string>();
  return 0;
}

// Periodic progress line for a single run.
void log_progress(std::uint64_t step, std::uint64_t acc, std::uint64_t tried,
                  double chi2, const AtomicStructure &) {
  double rate =
      tried > 0 ? 100.0 * static_cast<double>(acc) / static_cast<double>(tried)
                : 0.0;
  BOOST_LOG_TRIVIAL(info) << "Step " << step << "  acceptance=" << std::fixed
                          << std::setprecision(1) << rate << "%"
                          << "  chi2=" << chi2;
}

// Run the refinement: an ensemble of replicas (best chi2 wins) when
// --ensemble > 1, otherwise a single run with checkpoint + progress callback.
template <typename Factory, typename Prepare>
Engine run_refinement(const po::variables_map &vm, Factory &&make_engine,
                      Prepare &&prepare, std::size_t n_ensemble,
                      std::uint64_t n_steps) {
  if (n_ensemble > 1) {
    BOOST_LOG_TRIVIAL(info)
        << "Ensemble: running " << n_ensemble << " replicas in parallel";
    return run_ensemble(make_engine, n_ensemble, n_steps, /*tbb=*/0, prepare);
  }
  Engine e = make_engine(0);
  prepare(e); // bind gradient generators to e's final location, if requested
  if (vm.count("checkpoint")) {
    e.set_checkpoint(vm["checkpoint"].as<std::string>());
  }
  e.set_step_callback(log_progress, 1000);
  BOOST_LOG_TRIVIAL(info) << "Starting " << n_steps << " steps";
  e.run(n_steps);
  return e;
}

void print_summary(const Engine &engine) {
  auto st = engine.stats();
  std::cout << "Done. Accepted " << st.steps_accepted << " / " << st.steps_tried
            << " moves (" << std::fixed << std::setprecision(1)
            << (st.steps_tried > 0
                    ? 100.0 * static_cast<double>(st.steps_accepted) /
                          static_cast<double>(st.steps_tried)
                    : 0.0)
            << "%)  final chi2=" << st.last_total_err << "\n";
}

// Default command: refine the input structure against the experimental targets.
Result<int> cmd_refine(RMCContext &ctx) {
  const auto &vm = ctx.options();
  const RMCConfig &cfg = ctx.config();
  // Load structure + experimental data once; the ensemble factory reuses them.
  BOOST_LEAF_AUTO(in, ctx.structure());
  BOOST_LEAF_AUTO(data, load_experimental_data(cfg));

  // Engine factory: a fresh engine per replica from the shared inputs, with a
  // per-replica seed offset for an independent stochastic stream.
  auto make_engine = [&](std::size_t replica) {
    RMCConfig c = cfg;
    c.seed = cfg.seed + static_cast<std::uint32_t>(replica);
    return build_engine(*in, data, c);
  };
  // Applied to each engine in its final location (gradient generators bind to
  // engine.constraints(), which the build/ensemble moves would invalidate).
  auto prepare = [&](Engine &e) { apply_move_generator(e, cfg); };

  const auto n_steps = cfg.steps;
  const auto n_ensemble = vm["ensemble"].as<std::size_t>();
  Engine engine =
      run_refinement(vm, make_engine, prepare, n_ensemble, n_steps);

  const mat3_t out_box = periodic_box_or_zero(in->bc);
  BOOST_LEAF_CHECK(
      write_structure_by_ext(engine.structure(), out_box, cfg.out_path));
  BOOST_LOG_TRIVIAL(info) << "Wrote refined structure to " << cfg.out_path;

  print_summary(engine);
  return 0;
}

std::vector<RMCCommand> build_commands() {
  const auto flag = [](const char *name) {
    return [name](const po::variables_map &vm) { return vm[name].as<bool>(); };
  };
  return {
      {"gen-random", flag("gen-random"), cmd_gen_random},
      {"gr", flag("gr"), cmd_compute_gr},
      {"adf-compute", flag("adf-compute"), cmd_compute_adf},
      {"refine", [](const po::variables_map &) { return true; }, cmd_refine},
  };
}

Result<int> dispatch(const po::variables_map &vm) {
  RMCContext ctx(vm);
  for (const RMCCommand &cmd : build_commands()) {
    if (cmd.selected(vm)) {
      BOOST_LOG_TRIVIAL(debug) << "Running command '" << cmd.name << "'";
      return cmd.run(ctx);
    }
  }
  // Unreachable: the trailing "refine" command matches everything.
  return leaf::new_error(std::string{"no command selected"});
}

// Build the command-line option schema.
po::options_description make_options_description() {
  po::options_description desc(
      "RMC_run — Reverse Monte Carlo structural refinement");
  desc.add_options()("help,h", "Show this help")(
      "pdb,p", po::value<std::string>(), "Input PDB file")(
      "lammps,l", po::value<std::string>(),
      "Input LAMMPS data file (atom_style atomic); supplies the periodic box")(
      "types,t", po::value<std::string>(),
      "Element symbols for LAMMPS atom types, in order, e.g. 'Zr Cu Ag'")(
      "pdf,d", po::value<std::string>(), "Experimental G(r) data file")(
      "sq,q", po::value<std::string>(), "Experimental S(Q) data file")(
      "steps,n", po::value<std::uint64_t>()->default_value(100000), "MC steps")(
      "ensemble,e", po::value<std::size_t>()->default_value(1),
      "Number of independent replicas to run in parallel; best chi2 wins")(
      "rho0", po::value<double>()->default_value(0.1),
      "Number density (atoms/Å³)")(
      "seed", po::value<std::uint32_t>()->default_value(42), "RNG seed")(
      "out,o", po::value<std::string>()->default_value("refined.pdb"),
      "Output PDB")("checkpoint,c", po::value<std::string>(),
                    "Checkpoint file path")(
      "box", po::value<std::string>(),
      "Box vectors: 'a b c' for orthogonal periodic or 'inf' for infinite")(
      "smart", po::bool_switch()->default_value(false),
      "Use smart adaptive selector")(
      "move-gen", po::value<std::string>()->default_value("random"),
      "Move proposer: 'random' (classic walk), 'langevin' (MALA) or 'leapfrog' "
      "(HMC) — the gradient movers steer atoms along −∇χ² toward the target")(
      "step", po::value<double>()->default_value(0.05),
      "Gradient step ε (Å) for --move-gen langevin/leapfrog")(
      "gr", po::bool_switch()->default_value(false),
      "Compute g(r) (total + partials) from the input structure and exit; "
      "no MC is run")("gr-out",
                      po::value<std::string>()->default_value("gr.dat"),
                      "Output path for g(r) (used with --gr)")(
      "rmin", po::value<double>()->default_value(0.0),
      "g(r) minimum radius (Å)")("rmax",
                                 po::value<double>()->default_value(10.0),
                                 "g(r) maximum radius (Å)")(
      "nbins", po::value<std::size_t>()->default_value(200),
      "g(r) / ADF number of bins")(
      "vasp", po::value<std::string>(),
      "Input VASP POSCAR/CONTCAR; supplies the periodic cell")(
      "adf,a", po::value<std::string>(),
      "Experimental ADF (bond-angle distribution) target file")(
      "adf-cutoff", po::value<double>()->default_value(3.4),
      "ADF bond cutoff (Å)")("adf-smooth", po::value<int>()->default_value(2),
                             "ADF boxcar smoothing half-width (0 disables)")(
      "adf-compute", po::bool_switch()->default_value(false),
      "Compute the ADF (total + partials) from the input structure and exit; "
      "no MC is run")("adf-out",
                      po::value<std::string>()->default_value("adf.dat"),
                      "Output path for the ADF (used with --adf-compute)")(
      "gen-random", po::bool_switch()->default_value(false),
      "Generate a random amorphous structure, write it to --out, and exit")(
      "elements", po::value<std::string>(),
      "Element symbols for --gen-random, e.g. 'Zr Cu'")(
      "counts", po::value<std::string>(),
      "Atom count per element for --gen-random, e.g. '50 50'")(
      "spacing", po::value<double>()->default_value(3.0),
      "Grid spacing (Å) for --gen-random")(
      "verbose,v", po::bool_switch()->default_value(false), "Verbose logging");
  return desc;
}

} // namespace

RMCRunner::RMCRunner() : options_(make_options_description()) {}

int RMCRunner::run(int argc, char **argv) {
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

  configure_logging(vm["verbose"].as<bool>());

  return leaf::try_handle_all(
      [&]() -> leaf::result<int> { return dispatch(vm); },
      [](std::string const &msg) -> int {
        std::cerr << "Error: " << msg << "\n";
        return 1;
      },
      [](leaf::error_info const &unmatched) -> int {
        std::cerr << "Unexpected error: " << unmatched << "\n";
        return 1;
      });
}

} // namespace RMC
