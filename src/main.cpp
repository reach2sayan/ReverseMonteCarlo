#include <fullrmc/Engine.hpp>
#include <fullrmc/constraints/BondConstraint.hpp>
#include <fullrmc/constraints/DistanceConstraint.hpp>
#include <fullrmc/constraints/PairDistributionConstraint.hpp>
#include <fullrmc/constraints/StructureFactorConstraint.hpp>
#include <fullrmc/core/BoundaryConditions.hpp>
#include <fullrmc/io/DataReader.hpp>
#include <fullrmc/io/PdbReader.hpp>
#include <fullrmc/selectors/SmartRandomSelector.hpp>

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

namespace po   = boost::program_options;
namespace leaf = boost::leaf;

int main(int argc, char *argv[]) {
  po::options_description desc(
      "fullrmc_run — Reverse Monte Carlo structural refinement");
  desc.add_options()
      ("help,h", "Show this help")
      ("pdb,p",  po::value<std::string>()->required(), "Input PDB file")
      ("pdf,d",  po::value<std::string>(), "Experimental G(r) data file")
      ("sq,q",   po::value<std::string>(), "Experimental S(Q) data file")
      ("steps,n", po::value<std::uint64_t>()->default_value(100000), "MC steps")
      ("rho0",  po::value<double>()->default_value(0.1), "Number density (atoms/Å³)")
      ("seed",  po::value<std::uint32_t>()->default_value(42), "RNG seed")
      ("out,o", po::value<std::string>()->default_value("refined.pdb"), "Output PDB")
      ("checkpoint,c", po::value<std::string>(), "Checkpoint file path")
      ("box",   po::value<std::string>(),
                "Box vectors: 'a b c' for orthogonal periodic or 'inf' for infinite")
      ("smart", po::bool_switch()->default_value(false), "Use smart adaptive selector")
      ("verbose,v", po::bool_switch()->default_value(false), "Verbose logging");

  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) { std::cout << desc << "\n"; return 0; }
    po::notify(vm);
  } catch (const po::error &e) {
    std::cerr << "Error: " << e.what() << "\n" << desc << "\n";
    return 1;
  }

  if (!vm["verbose"].as<bool>()) {
    boost::log::core::get()->set_filter(
        boost::log::trivial::severity >= boost::log::trivial::info);
  }

  return leaf::try_handle_all(
      [&]() -> leaf::result<int> {
        // ---- Load structure ----
        BOOST_LEAF_AUTO(s, fullrmc::io::read_pdb(vm["pdb"].as<std::string>()));
        BOOST_LOG_TRIVIAL(info) << "Loaded " << s.size() << " atoms";

        // ---- Boundary conditions ----
        fullrmc::BoundaryConditions bc = fullrmc::InfiniteBC(1.0);
        if (vm.count("box") && vm["box"].as<std::string>() != "inf") {
          std::istringstream ss(vm["box"].as<std::string>());
          double a, b, c;
          ss >> a >> b >> c;
          fullrmc::mat3_t box = fullrmc::mat3_t::Zero();
          box(0, 0) = a; box(1, 1) = b; box(2, 2) = c;
          bc = fullrmc::PeriodicBC(box);
          BOOST_LOG_TRIVIAL(info)
              << "Periodic box: " << a << " x " << b << " x " << c;
        }

        // ---- Engine ----
        fullrmc::Engine engine(std::move(s), std::move(bc));
        engine.build_atomic_groups(0.0, 0.2, vm["seed"].as<std::uint32_t>());

        if (vm["smart"].as<bool>())
          engine.set_selector(std::make_unique<fullrmc::SmartRandomSelector>(
              1.1, vm["seed"].as<std::uint32_t>()));

        // ---- Constraints ----
        double rho0 = vm["rho0"].as<double>();

        if (vm.count("pdf")) {
          BOOST_LEAF_AUTO(data,
              fullrmc::io::read_xy_data(vm["pdf"].as<std::string>()));
          auto c = std::make_unique<fullrmc::PairDistributionConstraint>();
          c->set_experimental_data(data);
          c->set_number_density(rho0);
          c->set_elements(&engine.structure().elements);
          c->initialise();
          engine.add_constraint(std::move(c));
          BOOST_LOG_TRIVIAL(info) << "Added PairDistribution constraint";
        }

        if (vm.count("sq")) {
          BOOST_LEAF_AUTO(data,
              fullrmc::io::read_xy_data(vm["sq"].as<std::string>()));
          auto c = std::make_unique<fullrmc::StructureFactorConstraint>();
          c->set_experimental_data(data);
          c->set_number_density(rho0);
          c->set_elements(&engine.structure().elements);
          c->initialise();
          engine.add_constraint(std::move(c));
          BOOST_LOG_TRIVIAL(info) << "Added StructureFactor constraint";
        }

        if (vm.count("checkpoint"))
          engine.set_checkpoint(vm["checkpoint"].as<std::string>());

        engine.set_step_callback(
            [](std::uint64_t step, std::uint64_t acc, std::uint64_t tried,
               double chi2) {
              double rate = tried > 0
                  ? 100.0 * static_cast<double>(acc) / static_cast<double>(tried)
                  : 0.0;
              BOOST_LOG_TRIVIAL(info)
                  << "Step " << step
                  << "  acceptance=" << std::fixed << std::setprecision(1)
                  << rate << "%" << "  chi2=" << chi2;
            },
            1000);

        BOOST_LOG_TRIVIAL(info)
            << "Starting " << vm["steps"].as<std::uint64_t>() << " steps";
        engine.run(vm["steps"].as<std::uint64_t>());

        BOOST_LEAF_CHECK(fullrmc::io::write_pdb(engine.structure(),
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
