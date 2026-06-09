#include "RMCRunner.hpp"

#include <RMC/Engine.hpp>
#include <RMC/Ensemble.hpp>
#include <RMC/analysis/AngularDistribution.hpp>
#include <RMC/analysis/RadialDistribution.hpp>
#include <RMC/constraints/AngularDistributionConstraint.hpp>
#include <RMC/constraints/BondConstraint.hpp>
#include <RMC/constraints/DistanceConstraint.hpp>
#include <RMC/constraints/PairDistributionConstraint.hpp>
#include <RMC/constraints/StructureFactorConstraint.hpp>
#include <RMC/core/BoundaryConditions.hpp>
#include <RMC/core/RandomStructure.hpp>
#include <RMC/io/DataReader.hpp>
#include <RMC/io/LammpsReader.hpp>
#include <RMC/io/PdbReader.hpp>
#include <RMC/io/VaspReader.hpp>
#include <RMC/selectors/SmartRandomSelector.hpp>

#include <boost/leaf.hpp>
#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/trivial.hpp>

#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace po = boost::program_options;
namespace leaf = boost::leaf;

namespace RMC {

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

// Write a structure choosing the format from the output path: a VASP POSCAR for
// .vasp/.poscar (or a POSCAR/CONTCAR name), a LAMMPS data file for
// .lammps/.lmp/.data, otherwise a PDB. The box (from the active periodic cell;
// zero for infinite) is needed by the VASP and LAMMPS writers.
RMC::Result<void> write_structure_by_ext(const RMC::AtomicStructure &s,
                                         const RMC::mat3_t &box,
                                         const std::string &path) {
  const std::filesystem::path p(path);
  const std::string ext = p.extension().string();
  const std::string stem = p.filename().string();
  if (ext == ".vasp" || ext == ".poscar" || ext == ".VASP" ||
      stem == "POSCAR" || stem == "CONTCAR") {
    return RMC::io::write_vasp(s, box, path);
  }
  if (ext == ".lammps" || ext == ".lmp" || ext == ".data") {
    return RMC::io::write_lammps_data(s, box, path);
  }
  return RMC::io::write_pdb(s, path);
}

void configure_logging(bool verbose) {
  if (!verbose) {
    boost::log::core::get()->set_filter(boost::log::trivial::severity >=
                                        boost::log::trivial::info);
  }
}

struct LoadedStructure {
  RMC::AtomicStructure structure;
  RMC::BoundaryConditions bc = RMC::InfiniteBC(1.0);
};

// =====================================================================
// Input formats
//
// The chosen input is modelled as a variant of one struct per format, each
// carrying just the data that format needs. select_input() resolves the options
// into the right alternative (validating "exactly one"); a std::visit over an
// overload set — one lambda per format — then loads it. Adding a format is a
// struct, a selection branch, and a lambda, and the visit will not compile if
// the lambda is missing.
// =====================================================================

// The overload-set helper: fuse a pack of lambdas into one visitor.
template <typename... Fs> struct overloaded : Fs... {
  using Fs::operator()...;
};
template <typename... Fs> overloaded(Fs...) -> overloaded<Fs...>;

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

// Resolve the one input option the user supplied into a typed Input.
RMC::Result<Input> select_input(const po::variables_map &vm) {
  std::optional<Input> chosen;
  int given = 0;
  if (vm.count("pdb")) {
    ++given;
    chosen = PdbInput{vm["pdb"].as<std::string>()};
  }
  if (vm.count("lammps")) {
    ++given;
    auto types = vm.count("types")
                     ? parse_tokens<std::string>(vm["types"].as<std::string>())
                     : std::vector<std::string>{};
    chosen = LammpsInput{vm["lammps"].as<std::string>(), std::move(types)};
  }
  if (vm.count("vasp")) {
    ++given;
    chosen = VaspInput{vm["vasp"].as<std::string>()};
  }
  if (given != 1)
    return leaf::new_error(std::string{
        "Provide exactly one of --pdb, --lammps or --vasp as the input "
        "structure"});
  return std::move(*chosen);
}

// Load the single input structure. The format's own cell is captured in `bc`;
// --box may override it later.
RMC::Result<LoadedStructure> load_input_structure(const po::variables_map &vm) {
  BOOST_LEAF_AUTO(input, select_input(vm));
  return std::visit(
      overloaded{
          [](const PdbInput &in) -> RMC::Result<LoadedStructure> {
            BOOST_LEAF_AUTO(ps, RMC::io::read_pdb(in.path));
            LoadedStructure out;
            out.structure = std::move(ps);
            BOOST_LOG_TRIVIAL(info)
                << "Loaded " << out.structure.size() << " atoms";
            return out;
          },
          [](const LammpsInput &in) -> RMC::Result<LoadedStructure> {
            BOOST_LEAF_AUTO(data,
                            RMC::io::read_lammps_data(in.path, in.types));
            LoadedStructure out;
            out.structure = std::move(data.structure);
            // A LAMMPS data file carries its own cell; --box may override.
            out.bc = data.periodic_bc();
            BOOST_LOG_TRIVIAL(info)
                << "Loaded " << out.structure.size()
                << " atoms from LAMMPS data; box " << data.box(0, 0) << " x "
                << data.box(1, 1) << " x " << data.box(2, 2);
            return out;
          },
          [](const VaspInput &in) -> RMC::Result<LoadedStructure> {
            BOOST_LEAF_AUTO(data, RMC::io::read_vasp(in.path));
            LoadedStructure out;
            out.structure = std::move(data.structure);
            // A POSCAR carries its own cell; --box may override.
            out.bc = data.periodic_bc();
            BOOST_LOG_TRIVIAL(info)
                << "Loaded " << out.structure.size()
                << " atoms from VASP POSCAR; box " << data.box(0, 0) << " x "
                << data.box(1, 1) << " x " << data.box(2, 2);
            return out;
          },
      },
      input);
}

// --box overrides whatever box the input implied (incl. the LAMMPS/VASP cell).
void apply_box_override(const po::variables_map &vm,
                        RMC::BoundaryConditions &bc) {
  if (!vm.count("box")) {
    return;
  }
  if (vm["box"].as<std::string>() == "inf") {
    bc = RMC::InfiniteBC(1.0);
    BOOST_LOG_TRIVIAL(info) << "Box: infinite (non-periodic)";
    return;
  }
  std::istringstream ss(vm["box"].as<std::string>());
  double a, b, c;
  ss >> a >> b >> c;
  RMC::mat3_t box = RMC::mat3_t::Zero();
  box(0, 0) = a;
  box(1, 1) = b;
  box(2, 2) = c;
  bc = RMC::PeriodicBC(box);
  BOOST_LOG_TRIVIAL(info) << "Periodic box: " << a << " x " << b << " x " << c;
}

// =====================================================================
// RMCCommand pattern
//
// Each sub-command is a value — a name, a predicate that decides whether the
// parsed flags select it, and a type-erased action over a shared Context. This
// is the "generalized functor" form of the RMCCommand pattern (à la Alexandrescu):
// commands are data, not a class hierarchy. Adding one is a single entry in
// build_commands(); the dispatcher (RMCRunner::run) never changes.
// =====================================================================

// State threaded through every command. The input structure is loaded lazily
// (and the --box override applied) the first time a command asks for it, then
// cached — so structure-free commands like --gen-random never read a structure,
// while the structure-consuming commands share a single load.
class Context {
public:
  explicit Context(const po::variables_map &vm) : vm_(vm) {}

  const po::variables_map &options() const { return vm_; }

  // Load-on-first-use; the pointer stays valid for the Context's lifetime.
  RMC::Result<LoadedStructure *> structure() {
    if (!loaded_) {
      BOOST_LEAF_AUTO(ls, load_input_structure(vm_));
      apply_box_override(vm_, ls.bc);
      loaded_ = std::move(ls);
    }
    return &*loaded_;
  }

private:
  const po::variables_map &vm_;
  std::optional<LoadedStructure> loaded_;
};

// A command, dispatched by value: its name (for diagnostics), a predicate over
// the parsed options, and the action to run.
struct RMCCommand {
  std::string_view name;
  std::function<bool(const po::variables_map &)> selected;
  std::function<RMC::Result<int>(Context &)> run;
};

// --gen-random: build a random amorphous structure, write it to --out, exit.
RMC::Result<int> cmd_gen_random(Context &ctx) {
  const auto &vm = ctx.options();
  if (!vm.count("elements") || !vm.count("counts")) {
    return leaf::new_error(
        std::string{"--gen-random requires --elements and --counts"});
  }
  const auto els = parse_tokens<std::string>(vm["elements"].as<std::string>());
  const auto cnts = parse_tokens<std::size_t>(vm["counts"].as<std::string>());
  BOOST_LEAF_AUTO(
      gen, RMC::make_random_amorphous(els, cnts, vm["spacing"].as<double>(),
                                      vm["seed"].as<std::uint32_t>()));
  const std::string out = vm["out"].as<std::string>();
  BOOST_LEAF_CHECK(write_structure_by_ext(gen.structure, gen.box, out));
  BOOST_LOG_TRIVIAL(info) << "Generated " << gen.structure.size()
                          << " atoms; wrote " << out;
  return 0;
}

// --gr: compute the pair distribution g(r) and exit.
RMC::Result<int> cmd_compute_gr(Context &ctx) {
  const auto &vm = ctx.options();
  BOOST_LEAF_AUTO(in, ctx.structure());
  const RMC::AtomicStructure &s = in->structure;
  const RMC::BoundaryConditions &bc = in->bc;
  RMC::analysis::GrParams gp;
  gp.r_min = vm["rmin"].as<double>();
  gp.r_max = vm["rmax"].as<double>();
  gp.n_bins = static_cast<int>(vm["nbins"].as<std::size_t>());
  BOOST_LEAF_AUTO(g,
                  RMC::analysis::compute_gr(s.coordinates, bc, s.elements, gp));
  BOOST_LEAF_CHECK(RMC::analysis::write_gr(g, vm["gr-out"].as<std::string>()));
  BOOST_LOG_TRIVIAL(info) << "Wrote g(r) (" << g.pair_labels.size()
                          << " partials) to " << vm["gr-out"].as<std::string>();
  return 0;
}

// --adf-compute: compute the angular distribution function and exit.
RMC::Result<int> cmd_compute_adf(Context &ctx) {
  const auto &vm = ctx.options();
  BOOST_LEAF_AUTO(in, ctx.structure());
  const RMC::AtomicStructure &s = in->structure;
  const RMC::BoundaryConditions &bc = in->bc;
  RMC::analysis::AdfParams ap;
  ap.max_dis = vm["adf-cutoff"].as<double>();
  ap.n_bins = static_cast<int>(vm["nbins"].as<std::size_t>());
  ap.smooth_range = vm["adf-smooth"].as<int>();
  BOOST_LEAF_AUTO(
      a, RMC::analysis::compute_adf(s.coordinates, bc, s.elements, ap));
  BOOST_LEAF_CHECK(
      RMC::analysis::write_adf(a, vm["adf-out"].as<std::string>()));
  BOOST_LOG_TRIVIAL(info) << "Wrote ADF (" << a.triplet_labels.size()
                          << " triplets) to "
                          << vm["adf-out"].as<std::string>();
  return 0;
}

// =====================================================================
// Refinement (the default command — the main MC path)
// =====================================================================

// Experimental targets, pre-loaded once and shared across all replicas.
struct ExperimentalData {
  RMC::mat_t pdf, sq, adf;
  bool has_pdf = false, has_sq = false, has_adf = false;
};

RMC::Result<ExperimentalData>
load_experimental_data(const po::variables_map &vm) {
  ExperimentalData d;
  d.has_pdf = vm.count("pdf") > 0;
  d.has_sq = vm.count("sq") > 0;
  d.has_adf = vm.count("adf") > 0;
  if (d.has_pdf) {
    BOOST_LEAF_AUTO(x, RMC::io::read_xy_data(vm["pdf"].as<std::string>()));
    d.pdf = std::move(x);
    BOOST_LOG_TRIVIAL(info) << "Loaded PairDistribution data";
  }
  if (d.has_sq) {
    BOOST_LEAF_AUTO(x, RMC::io::read_xy_data(vm["sq"].as<std::string>()));
    d.sq = std::move(x);
    BOOST_LOG_TRIVIAL(info) << "Loaded StructureFactor data";
  }
  if (d.has_adf) {
    // The ADF target is multi-column (angle + one column per triplet), so read
    // all columns, not just two.
    BOOST_LEAF_AUTO(x, RMC::io::read_columns(vm["adf"].as<std::string>()));
    d.adf = std::move(x);
    BOOST_LOG_TRIVIAL(info) << "Loaded AngularDistribution data";
  }
  return d;
}

// Attach the requested experimental constraints to a freshly built engine.
void attach_constraints(RMC::Engine &engine, const ExperimentalData &data,
                        double rho0, double adf_cutoff, int adf_smooth) {
  if (data.has_pdf) {
    RMC::PairDistributionConstraint c;
    c.set_experimental_data(data.pdf);
    c.set_number_density(rho0);
    c.set_elements(engine.structure().elements);
    c.initialise();
    engine.add_constraint(std::move(c));
  }
  if (data.has_sq) {
    RMC::StructureFactorConstraint c;
    c.set_experimental_data(data.sq);
    c.set_number_density(rho0);
    c.set_elements(engine.structure().elements);
    c.initialise();
    engine.add_constraint(std::move(c));
  }
  if (data.has_adf) {
    RMC::AngularDistributionConstraint c;
    c.set_experimental_data(data.adf);
    c.set_cutoff(adf_cutoff);
    c.set_smoothing(adf_smooth);
    c.set_elements(engine.structure().elements);
    c.initialise();
    engine.add_constraint(std::move(c));
  }
}

// Periodic progress line for a single run.
void log_progress(std::uint64_t step, std::uint64_t acc, std::uint64_t tried,
                  double chi2, const RMC::AtomicStructure &) {
  double rate =
      tried > 0 ? 100.0 * static_cast<double>(acc) / static_cast<double>(tried)
                : 0.0;
  BOOST_LOG_TRIVIAL(info) << "Step " << step << "  acceptance=" << std::fixed
                          << std::setprecision(1) << rate << "%"
                          << "  chi2=" << chi2;
}

// Run the refinement: an ensemble of replicas (best chi2 wins) when
// --ensemble > 1, otherwise a single run with checkpoint + progress callback.
template <typename Factory>
RMC::Engine run_refinement(const po::variables_map &vm, Factory &&make_engine,
                           std::size_t n_ensemble, std::uint64_t n_steps) {
  if (n_ensemble > 1) {
    BOOST_LOG_TRIVIAL(info)
        << "Ensemble: running " << n_ensemble << " replicas in parallel";
    return RMC::run_ensemble(make_engine, n_ensemble, n_steps);
  }
  RMC::Engine e = make_engine(0);
  if (vm.count("checkpoint")) {
    e.set_checkpoint(vm["checkpoint"].as<std::string>());
  }
  e.set_step_callback(log_progress, 1000);
  BOOST_LOG_TRIVIAL(info) << "Starting " << n_steps << " steps";
  e.run(n_steps);
  return e;
}

void print_summary(const RMC::Engine &engine) {
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
RMC::Result<int> cmd_refine(Context &ctx) {
  const auto &vm = ctx.options();
  BOOST_LEAF_AUTO(in, ctx.structure());
  RMC::AtomicStructure s = std::move(in->structure);
  RMC::BoundaryConditions bc = in->bc;

  BOOST_LEAF_AUTO(data, load_experimental_data(vm));

  const double rho0 = vm["rho0"].as<double>();
  const double adf_cutoff = vm["adf-cutoff"].as<double>();
  const int adf_smooth = vm["adf-smooth"].as<int>();
  const auto base_seed = vm["seed"].as<std::uint32_t>();
  const bool use_smart = vm["smart"].as<bool>();

  // Engine factory: builds a fresh engine for replica i. Captures by value so
  // each replica owns its inputs independently.
  auto make_engine = [s, bc, data, rho0, adf_cutoff, adf_smooth, base_seed,
                      use_smart](std::size_t replica) {
    auto seed = base_seed + static_cast<std::uint32_t>(replica);
    RMC::Engine engine(s, bc);
    engine.build_atomic_groups(0.0, 0.2, seed);
    if (use_smart) {
      engine.set_selector(RMC::SmartRandomSelector{1.1, seed});
    }
    attach_constraints(engine, data, rho0, adf_cutoff, adf_smooth);
    return engine;
  };

  const auto n_steps = vm["steps"].as<std::uint64_t>();
  const auto n_ensemble = vm["ensemble"].as<std::size_t>();
  RMC::Engine engine = run_refinement(vm, make_engine, n_ensemble, n_steps);

  RMC::mat3_t out_box = RMC::mat3_t::Zero();
  if (const auto *p = std::get_if<RMC::PeriodicBC>(&bc))
    out_box = p->box();
  BOOST_LEAF_CHECK(write_structure_by_ext(engine.structure(), out_box,
                                          vm["out"].as<std::string>()));
  BOOST_LOG_TRIVIAL(info) << "Wrote refined structure to "
                          << vm["out"].as<std::string>();

  print_summary(engine);
  return 0;
}

// =====================================================================
// Dispatch
// =====================================================================

// The command table — the program's single extension point. Order matters: the
// first command whose predicate fires wins, and the trailing always-true entry
// makes refinement the default. To add a sub-command, add a row here.
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

RMC::Result<int> dispatch(const po::variables_map &vm) {
  Context ctx(vm);
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
