#include <RMC/Engine.hpp>
#include <RMC/Ensemble.hpp>
#include <RMC/constraints/BondConstraint.hpp>
#include <RMC/constraints/DistanceConstraint.hpp>
#include <RMC/constraints/PairDistributionConstraint.hpp>
#include <RMC/constraints/StructureFactorConstraint.hpp>
#include <RMC/core/BoundaryConditions.hpp>
#include <RMC/io/DataReader.hpp>
#include <RMC/io/PdbReader.hpp>
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

namespace po = boost::program_options;
namespace leaf = boost::leaf;

int main(int argc, char *argv[]) {
  po::options_description desc(
      "RMC_run — Reverse Monte Carlo structural refinement");
  desc.add_options()("help,h", "Show this help")(
      "pdb,p", po::value<std::string>()->required(), "Input PDB file")(
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
        BOOST_LEAF_AUTO(s, RMC::io::read_pdb(vm["pdb"].as<std::string>()));
        BOOST_LOG_TRIVIAL(info) << "Loaded " << s.size() << " atoms";
        RMC::BoundaryConditions bc = RMC::InfiniteBC(1.0);
        if (vm.count("box") && vm["box"].as<std::string>() != "inf") {
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

        // ---- Engine factory: builds a fresh engine for replica i ----
        auto base_seed = vm["seed"].as<std::uint32_t>();
        bool use_smart = vm["smart"].as<bool>();
        auto make_engine = [s, bc, pdf_data, sq_data, has_pdf, has_sq, rho0,
                            base_seed, use_smart](std::size_t replica) {
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

        BOOST_LEAF_CHECK(RMC::io::write_pdb(engine.structure(),
                                            vm["out"].as<std::string>()));
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
