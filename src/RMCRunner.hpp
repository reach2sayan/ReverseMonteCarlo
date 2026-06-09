#pragma once

#include <boost/program_options.hpp>

namespace RMC {

// Application driver for the RMC_run executable. Owns the command-line option
// schema and dispatches to the selected sub-command (--gen-random / --gr /
// --adf-compute, or the default Monte-Carlo refinement). main() just constructs
// one of these and forwards argc/argv to run().
class RMCRunner {
public:
  RMCRunner();
  // exit code (0 on success, 1 on error).
  int run(int argc, char **argv);

private:
  boost::program_options::options_description options_;
};

} // namespace RMC
