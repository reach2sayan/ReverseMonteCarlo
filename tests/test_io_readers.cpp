#include <RMC/io/DataReader.hpp>
#include <RMC/io/PdbReader.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <filesystem>
#include <format>
#include <fstream>
#include <string>
#include <string_view>
#include <system_error>

using Catch::Matchers::WithinAbs;
using RMC::mat_t;
namespace io = RMC::io;

namespace {

// RAII scratch file under the system temp dir.
struct TmpFile {
  std::filesystem::path path;
  TmpFile(const std::string &name, std::string_view content)
      : path(std::filesystem::temp_directory_path() / name) {
    std::ofstream(path) << content;
  }
  ~TmpFile() {
    std::error_code ec;
    std::filesystem::remove(path, ec);
  }
};

// Build a fixed-column PDB ATOM record (same column layout as write_pdb()).
std::string atom_line(int serial, std::string_view name, std::string_view res,
                      char chain, int resseq, double x, double y, double z,
                      std::string_view element) {
  return std::format("ATOM  {:5d} {:<4.4} {:<3.3} {}{:4d}    "
                     "{:8.3f}{:8.3f}{:8.3f}  1.00  0.00          {:>2.2}",
                     serial, name, res, chain, resseq, x, y, z, element);
}

} // namespace

// ============================================================
// read_xy_data
// ============================================================

TEST_CASE("read_xy_data - whitespace-separated rows", "[io]") {
  TmpFile tf("rmc_xy_ws.dat", "1.0 2.0\n3.0 4.0\n");
  const auto r = io::read_xy_data(tf.path);
  REQUIRE(r);
  const mat_t m = *r;
  REQUIRE(m.rows() == 2);
  REQUIRE(m.cols() == 2);
  REQUIRE_THAT(m(0, 0), WithinAbs(1.0, 1e-12));
  REQUIRE_THAT(m(0, 1), WithinAbs(2.0, 1e-12));
  REQUIRE_THAT(m(1, 0), WithinAbs(3.0, 1e-12));
  REQUIRE_THAT(m(1, 1), WithinAbs(4.0, 1e-12));
}

TEST_CASE("read_xy_data - comma-separated rows (regression)", "[io]") {
  // Previously the whitespace-only `>> x >> y` silently dropped these.
  TmpFile tf("rmc_xy_csv.dat", "1.0,2.0\n3.0, 4.0\n");
  const auto r = io::read_xy_data(tf.path);
  REQUIRE(r);
  const mat_t m = *r;
  REQUIRE(m.rows() == 2);
  REQUIRE_THAT(m(0, 0), WithinAbs(1.0, 1e-12));
  REQUIRE_THAT(m(0, 1), WithinAbs(2.0, 1e-12));
  REQUIRE_THAT(m(1, 1), WithinAbs(4.0, 1e-12));
}

TEST_CASE("read_xy_data - comments and blank lines skipped", "[io]") {
  TmpFile tf("rmc_xy_cmt.dat", "# header\n\n1.0 2.0\n# mid comment\n3.0 4.0\n");
  const auto r = io::read_xy_data(tf.path);
  REQUIRE(r);
  REQUIRE((*r).rows() == 2);
}

TEST_CASE("read_xy_data - line with fewer than two numbers skipped", "[io]") {
  TmpFile tf("rmc_xy_short.dat", "1.0\n2.0 3.0\n");
  const auto r = io::read_xy_data(tf.path);
  REQUIRE(r);
  const mat_t m = *r;
  REQUIRE(m.rows() == 1);
  REQUIRE_THAT(m(0, 0), WithinAbs(2.0, 1e-12));
  REQUIRE_THAT(m(0, 1), WithinAbs(3.0, 1e-12));
}

TEST_CASE("read_xy_data - extra columns ignored", "[io]") {
  TmpFile tf("rmc_xy_extra.dat", "1.0 2.0 9.9 8.8\n");
  const auto r = io::read_xy_data(tf.path);
  REQUIRE(r);
  const mat_t m = *r;
  REQUIRE(m.rows() == 1);
  REQUIRE_THAT(m(0, 0), WithinAbs(1.0, 1e-12));
  REQUIRE_THAT(m(0, 1), WithinAbs(2.0, 1e-12));
}

TEST_CASE("read_xy_data - missing file errors", "[io]") {
  const auto r = io::read_xy_data("/nonexistent/path/rmc_missing.dat");
  REQUIRE_FALSE(r);
}

// ============================================================
// read_pdb
// ============================================================

TEST_CASE("read_pdb - element source, coordinates, molecule grouping", "[io]") {
  // Atom 0: blank element column ⇒ derive from name "1HB" → "H".
  // Atom 1: explicit element column "C".
  // Atom 2: different chain ⇒ new molecule id.
  const std::string body =
      atom_line(1, "1HB", "ALA", 'A', 1, 1.234, 5.678, 9.012, "") + "\n" +
      atom_line(2, "CA", "ALA", 'A', 1, -1.500, 0.000, 2.250, "C") + "\n" +
      atom_line(3, "O", "HOH", 'B', 1, 3.333, 4.444, 5.555, "O") + "\n" +
      "TER\nEND\n";
  TmpFile tf("rmc_test.pdb", body);

  const auto r = io::read_pdb(tf.path);
  REQUIRE(r);
  const auto &s = *r;

  REQUIRE(s.size() == 3);
  REQUIRE(s.elements == std::vector<std::string>{"H", "C", "O"});
  REQUIRE(s.names == std::vector<std::string>{"1HB", "CA", "O"});

  // Coordinates parsed via std::from_chars (PDB %8.3f fields).
  REQUIRE_THAT(s.coordinates(0, 0), WithinAbs(1.234, 1e-3));
  REQUIRE_THAT(s.coordinates(1, 0), WithinAbs(-1.500, 1e-3));
  REQUIRE_THAT(s.coordinates(2, 2), WithinAbs(5.555, 1e-3));

  // Same chain+resSeq ⇒ same molecule; chain change ⇒ new molecule.
  REQUIRE(s.molecule_ids[0] == s.molecule_ids[1]);
  REQUIRE(s.molecule_ids[2] != s.molecule_ids[1]);

  // Atomic numbers: H=1, C=6, O=8 (all in the lookup table).
  REQUIRE(s.atomic_numbers(0) == 1);
  REQUIRE(s.atomic_numbers(1) == 6);
  REQUIRE(s.atomic_numbers(2) == 8);
}

TEST_CASE("read_pdb - unknown element maps to atomic number 0", "[io]") {
  const std::string body =
      atom_line(1, "XX", "UNK", 'A', 1, 0.0, 0.0, 0.0, "Xx") + "\n";
  TmpFile tf("rmc_test_unk.pdb", body);
  const auto r = io::read_pdb(tf.path);
  REQUIRE(r);
  REQUIRE((*r).atomic_numbers(0) == 0);
}

TEST_CASE("read_pdb - non-record and short lines ignored", "[io]") {
  const std::string body =
      "REMARK this is not an atom\n" + std::string("ATOM  short\n") +
      atom_line(1, "CA", "ALA", 'A', 1, 1.0, 2.0, 3.0, "C") + "\n";
  TmpFile tf("rmc_test_mixed.pdb", body);
  const auto r = io::read_pdb(tf.path);
  REQUIRE(r);
  REQUIRE((*r).size() == 1);
}

TEST_CASE("read_pdb - write/read round-trip preserves coordinates", "[io]") {
  const std::string body =
      atom_line(1, "CA", "ALA", 'A', 1, 1.111, 2.222, 3.333, "C") + "\n";
  TmpFile in("rmc_rt_in.pdb", body);
  const auto r1 = io::read_pdb(in.path);
  REQUIRE(r1);

  const auto out = std::filesystem::temp_directory_path() / "rmc_rt_out.pdb";
  const auto w = io::write_pdb(*r1, out);
  REQUIRE(w);
  const auto r2 = io::read_pdb(out);
  REQUIRE(r2);
  std::error_code ec;
  std::filesystem::remove(out, ec);

  REQUIRE((*r2).size() == 1);
  REQUIRE_THAT((*r2).coordinates(0, 0), WithinAbs(1.111, 1e-3));
  REQUIRE_THAT((*r2).coordinates(0, 1), WithinAbs(2.222, 1e-3));
  REQUIRE_THAT((*r2).coordinates(0, 2), WithinAbs(3.333, 1e-3));
}
