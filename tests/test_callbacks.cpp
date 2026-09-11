#include <RMC/callbacks/Chi2CollectorCallback.hpp>
#include <RMC/callbacks/HistogramCallback.hpp>
#include <RMC/callbacks/PDBSnapshotCallback.hpp>
#include <RMC/core/Structure.hpp>
#include <RMC/core/Types.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cstdlib>
#include <filesystem>
#include <random>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace RMC;
using namespace RMC::callbacks;
using Catch::Matchers::WithinAbs;

namespace fs = std::filesystem;

// Minimal 3-atom structure for callback invocations.
static AtomicStructure make_small_structure() {
  AtomicStructure s;
  s.coordinates.resize(3, 3);
  s.coordinates << 0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 3.0, 0.0, 0.0;
  s.atomic_numbers.resize(3);
  s.atomic_numbers << 18, 18, 18;
  for (int i = 0; i < 3; ++i) {
    s.names.push_back("Ar");
    s.elements.push_back("Ar");
    s.residues.push_back("ARG");
    s.molecule_ids.push_back(static_cast<std::size_t>(i));
  }
  return s;
}

// Unique per-test scratch directory under the system temp dir. The suffix is a
// random_device draw rather than the pid: ::getpid() is POSIX-only, and the
// directory only has to be unique, not attributable to a process.
static fs::path make_scratch(const char *tag) {
  static const auto salt = std::random_device{}();
  auto dir = fs::temp_directory_path() /
             ("rmc_cb_test_" + std::string(tag) + "_" + std::to_string(salt));
  fs::remove_all(dir);
  return dir;
}

// Invoke a callback as if it were a StepCallback.
template <typename CB>
static void invoke(CB &cb, std::uint64_t step, double chi2,
                   const AtomicStructure &s) {
  cb(step, /*acc=*/1, /*tried=*/1, chi2, s);
}

TEST_CASE("PDBSnapshotCallback - creates PDB with correct name",
          "[callbacks]") {
  auto dir = make_scratch("pdb_name");
  fs::create_directories(dir);

  AtomicStructure s = make_small_structure();
  PDBSnapshotCallback cb{dir};
  invoke(cb, 42, 1.0, s);

  auto expected = dir / "step_0000000042.pdb";
  REQUIRE(fs::exists(expected));
  REQUIRE(fs::file_size(expected) > 0);

  fs::remove_all(dir);
}

TEST_CASE("PDBSnapshotCallback - creates multiple files", "[callbacks]") {
  auto dir = make_scratch("pdb_multi");
  fs::create_directories(dir);

  AtomicStructure s = make_small_structure();
  PDBSnapshotCallback cb{dir};
  invoke(cb, 0, 1.0, s);
  invoke(cb, 1000, 0.8, s);
  invoke(cb, 2000, 0.6, s);

  REQUIRE(fs::exists(dir / "step_0000000000.pdb"));
  REQUIRE(fs::exists(dir / "step_0000001000.pdb"));
  REQUIRE(fs::exists(dir / "step_0000002000.pdb"));

  fs::remove_all(dir);
}

TEST_CASE("PDBSnapshotCallback - creates output directory if absent",
          "[callbacks]") {
  auto dir = make_scratch("pdb_mkdir") / "nested" / "deep";
  // dir does NOT exist yet

  AtomicStructure s = make_small_structure();
  PDBSnapshotCallback cb{dir};
  invoke(cb, 1, 1.0, s);

  REQUIRE(fs::exists(dir / "step_0000000001.pdb"));

  fs::remove_all(make_scratch("pdb_mkdir")); // clean parent
}

static vec_t make_vec(std::initializer_list<double> vals) {
  vec_t v(static_cast<Eigen::Index>(vals.size()));
  Eigen::Index i = 0;
  for (double x : vals)
    v[i++] = x;
  return v;
}

TEST_CASE("HistogramCallback - writes CSV with correct header and data",
          "[callbacks]") {
  auto dir = make_scratch("hist_csv");
  fs::create_directories(dir);

  vec_t axis = make_vec({0.5, 1.0, 1.5, 2.0, 2.5});
  vec_t computed = make_vec({1.1, 2.2, 3.3, 4.4, 5.5});
  vec_t experimental = make_vec({1.0, 2.0, 3.0, 4.0, 5.0});

  HistogramCallback cb{
      .axis = axis,
      .getter = [&] { return std::make_pair(computed, experimental); },
      .dir = dir};
  invoke(cb, 5, 0.0, make_small_structure());

  auto path = dir / "hist_0000000005.csv";
  REQUIRE(fs::exists(path));

  std::ifstream f(path);
  REQUIRE(f.is_open());

  std::string line;
  std::getline(f, line);
  REQUIRE(line == "axis,computed,experimental");

  int rows = 0;
  while (std::getline(f, line)) {
    if (line.empty())
      continue;
    std::istringstream ss(line);
    std::string tok;
    std::vector<double> vals;
    while (std::getline(ss, tok, ','))
      vals.push_back(std::stod(tok));
    REQUIRE(vals.size() == 3);
    REQUIRE_THAT(vals[0], WithinAbs(axis[rows], 1e-5));
    REQUIRE_THAT(vals[1], WithinAbs(computed[rows], 1e-5));
    REQUIRE_THAT(vals[2], WithinAbs(experimental[rows], 1e-5));
    ++rows;
  }
  REQUIRE(rows == 5);

  fs::remove_all(dir);
}

TEST_CASE("HistogramCallback - no-op when getter is empty", "[callbacks]") {
  auto dir = make_scratch("hist_empty");
  fs::create_directories(dir);

  HistogramCallback cb{.axis = make_vec({1.0}), .getter = {}, .dir = dir};
  // Must not crash; no file should be written.
  REQUIRE_NOTHROW(invoke(cb, 0, 0.0, make_small_structure()));
  REQUIRE(!fs::exists(dir / "hist_0000000000.csv"));

  fs::remove_all(dir);
}

TEST_CASE("HistogramCallback - truncates to shortest vector", "[callbacks]") {
  auto dir = make_scratch("hist_trunc");
  fs::create_directories(dir);

  vec_t axis = make_vec({0.5, 1.0, 1.5, 2.0, 2.5});    // 5
  vec_t computed = make_vec({1.0, 2.0, 3.0});          // 3  ← shortest
  vec_t experimental = make_vec({1.0, 2.0, 3.0, 4.0}); // 4

  HistogramCallback cb{
      .axis = axis,
      .getter = [&] { return std::make_pair(computed, experimental); },
      .dir = dir};
  invoke(cb, 7, 0.0, make_small_structure());

  auto path = dir / "hist_0000000007.csv";
  REQUIRE(fs::exists(path));

  std::ifstream f(path);
  std::string line;
  std::getline(f, line); // header
  int rows = 0;
  while (std::getline(f, line))
    if (!line.empty())
      ++rows;
  REQUIRE(rows == 3);

  fs::remove_all(dir);
}

TEST_CASE("Chi2CollectorCallback - accumulates history", "[callbacks]") {
  auto csv = make_scratch("chi2_acc") / "out.csv";
  fs::create_directories(csv.parent_path());

  Chi2CollectorCallback col(csv);
  AtomicStructure s = make_small_structure();

  std::vector<double> vals{1.0, 0.8, 0.6, 0.4, 0.2};
  for (std::size_t i = 0; i < vals.size(); ++i)
    invoke(col, static_cast<std::uint64_t>(i * 1000), vals[i], s);

  REQUIRE(col.history().size() == 5);
  for (std::size_t i = 0; i < vals.size(); ++i)
    REQUIRE_THAT(col.history()[i].second, WithinAbs(vals[i], 1e-12));

  fs::remove_all(csv.parent_path());
}

TEST_CASE("Chi2CollectorCallback - finalize writes CSV", "[callbacks]") {
  auto csv = make_scratch("chi2_csv") / "chi2.csv";
  fs::create_directories(csv.parent_path());

  Chi2CollectorCallback col(csv);
  AtomicStructure s = make_small_structure();
  invoke(col, 0, 1.5, s);
  invoke(col, 1000, 1.0, s);
  invoke(col, 2000, 0.5, s);
  col.finalize();

  REQUIRE(fs::exists(csv));

  std::ifstream f(csv);
  std::string line;
  std::getline(f, line);
  REQUIRE(line == "step,chi2");

  int rows = 0;
  std::vector<double> expected{1.5, 1.0, 0.5};
  while (std::getline(f, line)) {
    if (line.empty())
      continue;
    auto comma = line.find(',');
    double chi2 = std::stod(line.substr(comma + 1));
    REQUIRE_THAT(chi2, WithinAbs(expected[rows], 1e-8));
    ++rows;
  }
  REQUIRE(rows == 3);

  fs::remove_all(csv.parent_path());
}

TEST_CASE("Chi2CollectorCallback - finalize is idempotent", "[callbacks]") {
  auto csv = make_scratch("chi2_idem") / "chi2.csv";
  fs::create_directories(csv.parent_path());

  Chi2CollectorCallback col(csv);
  invoke(col, 0, 2.0, make_small_structure());
  col.finalize();

  auto sz1 = fs::file_size(csv);
  REQUIRE_NOTHROW(col.finalize()); // second call must not throw or re-write
  REQUIRE(fs::file_size(csv) == sz1);

  fs::remove_all(csv.parent_path());
}

TEST_CASE("Chi2CollectorCallback - destructor calls finalize", "[callbacks]") {
  auto csv = make_scratch("chi2_dtor") / "chi2.csv";
  fs::create_directories(csv.parent_path());

  {
    Chi2CollectorCallback col(csv);
    invoke(col, 0, 3.0, make_small_structure());
  } // destructor fires here

  REQUIRE(fs::exists(csv));
  REQUIRE(fs::file_size(csv) > 0);

  fs::remove_all(csv.parent_path());
}

TEST_CASE("Chi2CollectorCallback - finalize does not throw", "[callbacks]") {
  auto csv = make_scratch("chi2_nothrow") / "chi2.csv";
  fs::create_directories(csv.parent_path());

  Chi2CollectorCallback col(csv);
  AtomicStructure s = make_small_structure();
  for (int i = 0; i < 5; ++i)
    invoke(col, static_cast<std::uint64_t>(i), 1.0 - 0.1 * i, s);

  REQUIRE_NOTHROW(col.finalize());

  fs::remove_all(csv.parent_path());
}
