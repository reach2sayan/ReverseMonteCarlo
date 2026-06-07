#pragma once
// Drive ATAT's `corrdump` as an external binary via boost::process (v2). We do
// NOT link ATAT — each call runs corrdump in a working directory, lets it emit
// its files there, and (for correlations) redirects stdout to a file we parse.
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

// corrdump -l=<rndstr> -s=<str.out> -c -cf=<clusters.out>; parses the per-orbit
// correlations printed to stdout. The validation oracle for the enumerator.
std::vector<double>
corrdump_correlations(const std::filesystem::path &corrdump_exe,
                      const std::filesystem::path &rndstr_in,
                      const std::filesystem::path &str_out,
                      const std::filesystem::path &clusters_out,
                      const std::filesystem::path &workdir);

} // namespace RMC::atat
