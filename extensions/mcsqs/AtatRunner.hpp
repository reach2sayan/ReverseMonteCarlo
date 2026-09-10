#pragma once
// Drive ATAT's `corrdump` as an external binary via boost::process (v2); ATAT
// is not linked. Each call runs corrdump in a workdir and parses its outputs.
#include <filesystem>
#include <map>
#include <vector>

namespace RMC::atat {

struct CorrdumpClusters {
  std::filesystem::path clusters_out; // orbit definitions
  std::filesystem::path sym_out;      // space group
};

// corrdump -l=<rndstr> -ro -noe -nop -clus -2=d2 -3=d3 …  (run in `workdir`).
// Returns the produced clusters.out / sym.out paths (inside workdir).
CorrdumpClusters
corrdump_generate_clusters(const std::filesystem::path &corrdump_exe,
                           const std::filesystem::path &rndstr_in,
                           const std::map<int, double> &diameters,
                           const std::filesystem::path &workdir);

// corrdump -l=<rndstr> -s=<str.out> -c -cf=<clusters.out>; parses per-orbit
// correlations from stdout. Validation oracle for the enumerator.
std::vector<double>
corrdump_correlations(const std::filesystem::path &corrdump_exe,
                      const std::filesystem::path &rndstr_in,
                      const std::filesystem::path &str_out,
                      const std::filesystem::path &clusters_out,
                      const std::filesystem::path &workdir);

} // namespace RMC::atat
