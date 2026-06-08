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
#include <boost/program_options.hpp>

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

namespace po = boost::program_options;
namespace leaf = boost::leaf;

namespace {

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
  if (ext == ".vasp" || ext == ".poscar" || ext == ".VASP" || stem == "POSCAR" ||
      stem == "CONTCAR") {
    return RMC::io::write_vasp(s, box, path);
  }
  if (ext == ".lammps" || ext == ".lmp" || ext == ".data") {
    return RMC::io::write_lammps_data(s, box, path);
  }
  return RMC::io::write_pdb(s, path);
}

} // namespace

int main(int argc, char *argv[]) {
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
      "no MC is run")("gr-out", po::value<std::string>()->default_value("gr.dat"),
                      "Output path for g(r) (used with --gr)")(
      "rmin", po::value<double>()->default_value(0.0), "g(r) minimum radius (Å)")(
      "rmax", po::value<double>()->default_value(10.0),
      "g(r) maximum radius (Å)")(
      "nbins", po::value<std::size_t>()->default_value(200),
      "g(r) / ADF number of bins")(
      "vasp", po::value<std::string>(),
      "Input VASP POSCAR/CONTCAR; supplies the periodic cell")(
      "adf,a", po::value<std::string>(),
      "Experimental ADF (bond-angle distribution) target file")(
      "adf-cutoff", po::value<double>()->default_value(3.4),
      "ADF bond cutoff (Å)")(
      "adf-smooth", po::value<int>()->default_value(2),
      "ADF boxcar smoothing half-width (0 disables)")(
      "adf-compute", po::bool_switch()->default_value(false),
      "Compute the ADF (total + partials) from the input structure and exit; "
      "no MC is run")(
      "adf-out", po::value<std::string>()->default_value("adf.dat"),
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

  if (!vm["verbose"].as<bool>()) {
    boost::log::core::get()->set_filter(boost::log::trivial::severity >=
                                        boost::log::trivial::info);
  }

  return leaf::try_handle_all(
      [&]() -> leaf::result<int> {
        // ---- Random-structure generation: build, write, exit (no input) ----
        if (vm["gen-random"].as<bool>()) {
          if (!vm.count("elements") || !vm.count("counts"))
            return boost::leaf::new_error(std::string{
                "--gen-random requires --elements and --counts"});
          std::vector<std::string> els;
          {
            std::istringstream es(vm["elements"].as<std::string>());
            for (std::string e; es >> e;)
              els.push_back(std::move(e));
          }
          std::vector<std::size_t> cnts;
          {
            std::istringstream cs(vm["counts"].as<std::string>());
            for (std::size_t c; cs >> c;)
              cnts.push_back(c);
          }
          BOOST_LEAF_AUTO(gen, RMC::make_random_amorphous(
                                   els, cnts, vm["spacing"].as<double>(),
                                   vm["seed"].as<std::uint32_t>()));
          const std::string out = vm["out"].as<std::string>();
          BOOST_LEAF_CHECK(write_structure_by_ext(gen.structure, gen.box, out));
          BOOST_LOG_TRIVIAL(info)
              << "Generated " << gen.structure.size() << " atoms; wrote " << out;
          return 0;
        }

        const bool has_pdb_in = vm.count("pdb") > 0;
        const bool has_lammps_in = vm.count("lammps") > 0;
        const bool has_vasp_in = vm.count("vasp") > 0;
        if (static_cast<int>(has_pdb_in) + static_cast<int>(has_lammps_in) +
                static_cast<int>(has_vasp_in) !=
            1)
          return boost::leaf::new_error(std::string{
              "Provide exactly one of --pdb, --lammps or --vasp as the input "
              "structure"});

        RMC::AtomicStructure s;
        RMC::BoundaryConditions bc = RMC::InfiniteBC(1.0);

        if (has_lammps_in) {
          // Parse the optional 'Zr Cu Ag' type→element legend.
          std::vector<std::string> type_to_element;
          if (vm.count("types")) {
            std::istringstream ts(vm["types"].as<std::string>());
            for (std::string e; ts >> e;)
              type_to_element.push_back(std::move(e));
          }
          BOOST_LEAF_AUTO(data, RMC::io::read_lammps_data(
                                    vm["lammps"].as<std::string>(),
                                    type_to_element));
          s = std::move(data.structure);
          // A LAMMPS data file carries its own cell; use it unless the user
          // overrides with --box below.
          bc = data.periodic_bc();
          BOOST_LOG_TRIVIAL(info)
              << "Loaded " << s.size() << " atoms from LAMMPS data; box "
              << data.box(0, 0) << " x " << data.box(1, 1) << " x "
              << data.box(2, 2);
        } else if (has_vasp_in) {
          BOOST_LEAF_AUTO(data, RMC::io::read_vasp(vm["vasp"].as<std::string>()));
          s = std::move(data.structure);
          // A POSCAR carries its own cell; use it unless --box overrides below.
          bc = data.periodic_bc();
          BOOST_LOG_TRIVIAL(info)
              << "Loaded " << s.size() << " atoms from VASP POSCAR; box "
              << data.box(0, 0) << " x " << data.box(1, 1) << " x "
              << data.box(2, 2);
        } else {
          BOOST_LEAF_AUTO(ps, RMC::io::read_pdb(vm["pdb"].as<std::string>()));
          s = std::move(ps);
          BOOST_LOG_TRIVIAL(info) << "Loaded " << s.size() << " atoms";
        }

        // --box overrides whatever box the input implied (incl. the LAMMPS cell).
        if (vm.count("box")) {
          if (vm["box"].as<std::string>() == "inf") {
            bc = RMC::InfiniteBC(1.0);
            BOOST_LOG_TRIVIAL(info) << "Box: infinite (non-periodic)";
          } else {
            std::istringstream ss(vm["box"].as<std::string>());
            double a, b, c;
            ss >> a >> b >> c;
            RMC::mat3_t box = RMC::mat3_t::Zero();
            box(0, 0) = a;
            box(1, 1) = b;
            box(2, 2) = c;
            bc = RMC::PeriodicBC(box);
            BOOST_LOG_TRIVIAL(info)
                << "Periodic box: " << a << " x " << b << " x " << c;
          }
        }

        // ---- g(r) mode: compute the pair distribution and exit (no MC) ----
        if (vm["gr"].as<bool>()) {
          RMC::analysis::GrParams gp;
          gp.r_min = vm["rmin"].as<double>();
          gp.r_max = vm["rmax"].as<double>();
          gp.n_bins = static_cast<int>(vm["nbins"].as<std::size_t>());
          BOOST_LEAF_AUTO(
              g, RMC::analysis::compute_gr(s.coordinates, bc, s.elements, gp));
          BOOST_LEAF_CHECK(
              RMC::analysis::write_gr(g, vm["gr-out"].as<std::string>()));
          BOOST_LOG_TRIVIAL(info)
              << "Wrote g(r) (" << g.pair_labels.size() << " partials) to "
              << vm["gr-out"].as<std::string>();
          return 0;
        }

        // ---- ADF mode: compute the angular distribution and exit (no MC) ----
        if (vm["adf-compute"].as<bool>()) {
          RMC::analysis::AdfParams ap;
          ap.max_dis = vm["adf-cutoff"].as<double>();
          ap.n_bins = static_cast<int>(vm["nbins"].as<std::size_t>());
          ap.smooth_range = vm["adf-smooth"].as<int>();
          BOOST_LEAF_AUTO(
              a, RMC::analysis::compute_adf(s.coordinates, bc, s.elements, ap));
          BOOST_LEAF_CHECK(
              RMC::analysis::write_adf(a, vm["adf-out"].as<std::string>()));
          BOOST_LOG_TRIVIAL(info)
              << "Wrote ADF (" << a.triplet_labels.size() << " triplets) to "
              << vm["adf-out"].as<std::string>();
          return 0;
        }

        // ---- Pre-load experimental data (once, shared across replicas) ----
        double rho0 = vm["rho0"].as<double>();
        RMC::mat_t pdf_data, sq_data;
        bool has_pdf = vm.count("pdf") > 0, has_sq = vm.count("sq") > 0;
        if (has_pdf) {
          BOOST_LEAF_AUTO(d,
                          RMC::io::read_xy_data(vm["pdf"].as<std::string>()));
          pdf_data = std::move(d);
          BOOST_LOG_TRIVIAL(info) << "Loaded PairDistribution data";
        }
        if (has_sq) {
          BOOST_LEAF_AUTO(d, RMC::io::read_xy_data(vm["sq"].as<std::string>()));
          sq_data = std::move(d);
          BOOST_LOG_TRIVIAL(info) << "Loaded StructureFactor data";
        }
        RMC::mat_t adf_data;
        bool has_adf = vm.count("adf") > 0;
        if (has_adf) {
          // The ADF target is multi-column (angle + one column per triplet), so
          // read all columns, not just two.
          BOOST_LEAF_AUTO(d, RMC::io::read_columns(vm["adf"].as<std::string>()));
          adf_data = std::move(d);
          BOOST_LOG_TRIVIAL(info) << "Loaded AngularDistribution data";
        }
        double adf_cutoff = vm["adf-cutoff"].as<double>();
        int adf_smooth = vm["adf-smooth"].as<int>();

        // ---- Engine factory: builds a fresh engine for replica i ----
        auto base_seed = vm["seed"].as<std::uint32_t>();
        bool use_smart = vm["smart"].as<bool>();
        auto make_engine = [s, bc, pdf_data, sq_data, adf_data, has_pdf, has_sq,
                            has_adf, rho0, adf_cutoff, adf_smooth, base_seed,
                            use_smart](std::size_t replica) {
          auto seed = base_seed + static_cast<std::uint32_t>(replica);
          RMC::Engine engine(s, bc);
          engine.build_atomic_groups(0.0, 0.2, seed);
          if (use_smart)
            engine.set_selector(RMC::SmartRandomSelector{1.1, seed});
          if (has_pdf) {
            RMC::PairDistributionConstraint c;
            c.set_experimental_data(pdf_data);
            c.set_number_density(rho0);
            c.set_elements(engine.structure().elements);
            c.initialise();
            engine.add_constraint(std::move(c));
          }
          if (has_sq) {
            RMC::StructureFactorConstraint c;
            c.set_experimental_data(sq_data);
            c.set_number_density(rho0);
            c.set_elements(engine.structure().elements);
            c.initialise();
            engine.add_constraint(std::move(c));
          }
          if (has_adf) {
            RMC::AngularDistributionConstraint c;
            c.set_experimental_data(adf_data);
            c.set_cutoff(adf_cutoff);
            c.set_smoothing(adf_smooth);
            c.set_elements(engine.structure().elements);
            c.initialise();
            engine.add_constraint(std::move(c));
          }
          return engine;
        };

        auto n_steps = vm["steps"].as<std::uint64_t>();
        auto n_ensemble = vm["ensemble"].as<std::size_t>();

        RMC::Engine engine = [&]() -> RMC::Engine {
          if (n_ensemble > 1) {
            BOOST_LOG_TRIVIAL(info) << "Ensemble: running " << n_ensemble
                                    << " replicas in parallel";
            return RMC::run_ensemble(make_engine, n_ensemble, n_steps);
          }
          // Single run — attach callback and checkpoint
          RMC::Engine e = make_engine(0);
          if (vm.count("checkpoint"))
            e.set_checkpoint(vm["checkpoint"].as<std::string>());
          e.set_step_callback(
              [](std::uint64_t step, std::uint64_t acc, std::uint64_t tried,
                 double chi2, const RMC::AtomicStructure &) {
                double rate = tried > 0 ? 100.0 * static_cast<double>(acc) /
                                              static_cast<double>(tried)
                                        : 0.0;
                BOOST_LOG_TRIVIAL(info)
                    << "Step " << step << "  acceptance=" << std::fixed
                    << std::setprecision(1) << rate << "%"
                    << "  chi2=" << chi2;
              },
              1000);
          BOOST_LOG_TRIVIAL(info) << "Starting " << n_steps << " steps";
          e.run(n_steps);
          return e;
        }();

        RMC::mat3_t out_box = RMC::mat3_t::Zero();
        if (const auto *p = std::get_if<RMC::PeriodicBC>(&bc))
          out_box = p->box();
        BOOST_LEAF_CHECK(write_structure_by_ext(
            engine.structure(), out_box, vm["out"].as<std::string>()));
        BOOST_LOG_TRIVIAL(info)
            << "Wrote refined structure to " << vm["out"].as<std::string>();

        auto st = engine.stats();
        std::cout << "Done. Accepted " << st.steps_accepted << " / "
                  << st.steps_tried << " moves (" << std::fixed
                  << std::setprecision(1)
                  << (st.steps_tried > 0
                          ? 100.0 * static_cast<double>(st.steps_accepted) /
                                static_cast<double>(st.steps_tried)
                          : 0.0)
                  << "%)  final chi2=" << st.last_total_err << "\n";
        return 0;
      },
      [](std::string const &msg) -> int {
        std::cerr << "Error: " << msg << "\n";
        return 1;
      },
      [](leaf::error_info const &unmatched) -> int {
        std::cerr << "Unexpected error: " << unmatched << "\n";
        return 1;
      });
}
