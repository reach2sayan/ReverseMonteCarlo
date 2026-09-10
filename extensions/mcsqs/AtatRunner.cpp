#include "AtatRunner.hpp"

#include "AtatFormats.hpp"

#include <boost/asio/io_context.hpp>
// Boost 1.88+: Process v2 lives in boost::process; boost::process::v2 is now an
// inline namespace alias.
#include <boost/process.hpp>

#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace RMC::atat {
namespace {

namespace bp = boost::process::v2;
namespace fs = std::filesystem;

std::string slurp(const fs::path &p) {
  std::ifstream f(p);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

// Run corrdump in `workdir`, redirecting stdout→out_file, stderr→err_file.
// Throws (with captured stderr) on non-zero exit. Paths passed as strings and
// stdio bound via FILE* to avoid a std/boost::filesystem path-type mismatch.
void run_corrdump(const fs::path &exe, const std::vector<std::string> &args,
                  const fs::path &workdir, const fs::path &out_file) {
  const fs::path err_file = workdir / "corrdump.stderr";
  std::FILE *out = std::fopen(out_file.string().c_str(), "w");
  std::FILE *err = std::fopen(err_file.string().c_str(), "w");
  if (out == nullptr || err == nullptr) {
    if (out != nullptr) {
      std::fclose(out);
    }
    if (err != nullptr) {
      std::fclose(err);
    }
    throw std::runtime_error("cannot open corrdump output files in " +
                             workdir.string());
  }
  int code = 0;
  try {
    boost::asio::io_context ctx;
    bp::process proc(ctx, exe.string(), args,
                     bp::process_stdio{nullptr, out, err},
                     bp::process_start_dir{workdir.string()});
    code = proc.wait();
  } catch (...) {
    std::fclose(out);
    std::fclose(err);
    throw;
  }
  std::fclose(out);
  std::fclose(err);
  if (code != 0) {
    throw std::runtime_error("corrdump exited with code " +
                             std::to_string(code) + "\n" + slurp(err_file));
  }
}

} // namespace

CorrdumpClusters corrdump_generate_clusters(
    const fs::path &corrdump_exe, const fs::path &rndstr_in,
    const std::map<int, double> &diameters, const fs::path &workdir) {
  fs::create_directories(workdir);
  fs::copy_file(rndstr_in, workdir / "rndstr.in",
                fs::copy_options::overwrite_existing);

  std::vector<std::string> args{"-l=rndstr.in", "-ro", "-noe", "-nop", "-clus"};
  for (const auto &[n, d] : diameters) {
    args.push_back("-" + std::to_string(n) + "=" + std::to_string(d));
  }
  run_corrdump(corrdump_exe, args, workdir, workdir / "corrdump.stdout");

  CorrdumpClusters out{workdir / "clusters.out", workdir / "sym.out"};
  if (!fs::exists(out.clusters_out) || !fs::exists(out.sym_out)) {
    throw std::runtime_error(
        "corrdump did not produce clusters.out / sym.out in " +
        workdir.string());
  }
  return out;
}

std::vector<double> corrdump_correlations(const fs::path &corrdump_exe,
                                          const fs::path &rndstr_in,
                                          const fs::path &str_out,
                                          const fs::path &clusters_out,
                                          const fs::path &workdir) {
  fs::create_directories(workdir);
  fs::copy_file(rndstr_in, workdir / "rndstr.in",
                fs::copy_options::overwrite_existing);
  fs::copy_file(str_out, workdir / "str.out",
                fs::copy_options::overwrite_existing);
  if (fs::weakly_canonical(clusters_out) !=
      fs::weakly_canonical(workdir / "clusters.out")) {
    fs::copy_file(clusters_out, workdir / "clusters.out",
                  fs::copy_options::overwrite_existing);
  }

  const fs::path corr = workdir / "corr.out";
  // -ro: parse lattice in rndstr format so labels are "Au"/"Cu" (matching
  // str.out), not "Au=0.5"/"Cu=0.5".
  run_corrdump(corrdump_exe,
               {"-l=rndstr.in", "-ro", "-s=str.out", "-c", "-cf=clusters.out"},
               workdir, corr);

  std::ifstream f(corr);
  if (!f) {
    throw std::runtime_error("corrdump produced no correlation output");
  }
  return parse_correlations(f);
}

} // namespace RMC::atat
