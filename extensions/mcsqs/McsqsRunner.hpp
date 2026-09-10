#pragma once

#include <boost/program_options.hpp>

namespace RMC {

// Special Quasirandom Structure (SQS) generation.
//   [1] Zunger, Wei, Ferreira, Bernard, Phys. Rev. Lett. 65, 353-356 (1990).
//   [2] van de Walle et al., "Efficient stochastic generation of SQS,"
//       Calphad 42, 13-18 (2013). doi:10.1016/j.calphad.2013.06.006
// Cluster basis built via ATAT's corrdump (external tool, CC BY-ND 4.0).

// Driver for mcsqs_rmc. Two input pipelines (both build the same cluster basis):
//   corrdump (--lattice): expand rndstr.in to a supercell, generate clusters.
//   legacy (--structure/--clusters/--species): fixed-site PDB, no corrdump.
class McsqsRunner {
public:
  McsqsRunner();
  // exit code (0 on success, 1 on error).
  int run(int argc, char **argv);

private:
  boost::program_options::options_description options_;
};

} // namespace RMC
