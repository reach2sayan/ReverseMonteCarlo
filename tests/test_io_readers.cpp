#include <RMC/io/DataReader.hpp>
#include <RMC/io/LammpsReader.hpp>
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

// ============================================================
// read_lammps_data
// ============================================================

namespace {

// A minimal atom_style atomic data file: comment, header, Atoms section.
constexpr std::string_view kLammpsAtomic = R"(converted from bop software

3 atoms
2 atom types
-5.0 5.0 xlo xhi
-6.0 6.0 ylo yhi
-7.0 7.0 zlo zhi

Atoms

1 1 -5.00000000 -6.00000000 -7.00000000
2 2 0.00000000 0.00000000 0.00000000
3 1 1.25000000 2.50000000 3.75000000
)";

} // namespace

TEST_CASE("read_lammps_data - atomic style, box, types, atomic numbers",
          "[io]") {
  TmpFile tf("rmc_lmp_atomic.dat", kLammpsAtomic);
  const auto r = io::read_lammps_data(tf.path, {"Zr", "Cu"});
  REQUIRE(r);
  const auto &d = *r;
  const auto &s = d.structure;

  REQUIRE(s.size() == 3);

  // Box edge lengths: lx=10, ly=12, lz=14; origin = (xlo,ylo,zlo).
  REQUIRE_THAT(d.box(0, 0), WithinAbs(10.0, 1e-9));
  REQUIRE_THAT(d.box(1, 1), WithinAbs(12.0, 1e-9));
  REQUIRE_THAT(d.box(2, 2), WithinAbs(14.0, 1e-9));
  REQUIRE_THAT(d.box(0, 1), WithinAbs(0.0, 1e-9)); // orthorhombic: no tilt
  REQUIRE_THAT(d.origin.x(), WithinAbs(-5.0, 1e-9));
  REQUIRE_THAT(d.origin.z(), WithinAbs(-7.0, 1e-9));

  // Type 1 -> Zr (40), type 2 -> Cu (29).
  REQUIRE(s.elements == std::vector<std::string>{"Zr", "Cu", "Zr"});
  REQUIRE(s.atomic_numbers(0) == 40);
  REQUIRE(s.atomic_numbers(1) == 29);
  REQUIRE(s.atomic_numbers(2) == 40);

  // Absolute Cartesian coordinates straight from the file.
  REQUIRE_THAT(s.coordinates(0, 0), WithinAbs(-5.0, 1e-9));
  REQUIRE_THAT(s.coordinates(2, 1), WithinAbs(2.5, 1e-9));
  REQUIRE_THAT(s.coordinates(2, 2), WithinAbs(3.75, 1e-9));
}

TEST_CASE("read_lammps_data - periodic_bc uses the parsed cell", "[io]") {
  TmpFile tf("rmc_lmp_bc.dat", kLammpsAtomic);
  const auto r = io::read_lammps_data(tf.path, {"Zr", "Cu"});
  REQUIRE(r);
  const RMC::PeriodicBC bc = r->periodic_bc();
  // Minimum image across the 10 Å x-edge folds a 6 Å separation to -4 Å.
  const RMC::vec3_t mi = bc.min_image(RMC::vec3_t{6.0, 0.0, 0.0});
  REQUIRE_THAT(mi.x(), WithinAbs(-4.0, 1e-9));
}

TEST_CASE("read_lammps_data - unmapped types fall back to X<type>", "[io]") {
  TmpFile tf("rmc_lmp_nomap.dat", kLammpsAtomic);
  const auto r = io::read_lammps_data(tf.path); // no type_to_element
  REQUIRE(r);
  const auto &s = r->structure;
  REQUIRE(s.elements == std::vector<std::string>{"X1", "X2", "X1"});
  // Fallback atomic_number is the type integer, keeping species distinct.
  REQUIRE(s.atomic_numbers(0) == 1);
  REQUIRE(s.atomic_numbers(1) == 2);
}

TEST_CASE("read_lammps_data - partial mapping fills the gaps", "[io]") {
  // Only type 1 named; type 2 must fall back without disturbing type 1.
  TmpFile tf("rmc_lmp_partial.dat", kLammpsAtomic);
  const auto r = io::read_lammps_data(tf.path, {"Zr"});
  REQUIRE(r);
  const auto &s = r->structure;
  REQUIRE(s.elements == std::vector<std::string>{"Zr", "X2", "Zr"});
  REQUIRE(s.atomic_numbers(0) == 40);
  REQUIRE(s.atomic_numbers(1) == 2);
}

TEST_CASE("read_lammps_data - skips Masses/Velocities before Atoms", "[io]") {
  constexpr std::string_view body = R"(comment line

2 atoms
2 atom types
0.0 10.0 xlo xhi
0.0 10.0 ylo yhi
0.0 10.0 zlo zhi

Masses

1 91.224
2 63.546

Velocities

1 0.1 0.2 0.3
2 -0.1 -0.2 -0.3

Atoms # atomic

1 1 1.0 2.0 3.0
2 2 4.0 5.0 6.0
)";
  TmpFile tf("rmc_lmp_sections.dat", body);
  const auto r = io::read_lammps_data(tf.path, {"Zr", "Cu"});
  REQUIRE(r);
  const auto &s = r->structure;
  REQUIRE(s.size() == 2);
  REQUIRE(s.elements == std::vector<std::string>{"Zr", "Cu"});
  REQUIRE_THAT(s.coordinates(1, 2), WithinAbs(6.0, 1e-9));
}

TEST_CASE("read_lammps_data - triclinic tilt factors", "[io]") {
  constexpr std::string_view body = R"(triclinic

1 atoms
1 atom types
0.0 10.0 xlo xhi
0.0 10.0 ylo yhi
0.0 10.0 zlo zhi
1.0 2.0 3.0 xy xz yz

Atoms

1 1 0.5 0.5 0.5
)";
  TmpFile tf("rmc_lmp_tri.dat", body);
  const auto r = io::read_lammps_data(tf.path, {"Zr"});
  REQUIRE(r);
  // box columns: a=(10,0,0), b=(xy,10,0), c=(xz,yz,10).
  REQUIRE_THAT(r->box(0, 1), WithinAbs(1.0, 1e-9)); // xy
  REQUIRE_THAT(r->box(0, 2), WithinAbs(2.0, 1e-9)); // xz
  REQUIRE_THAT(r->box(1, 2), WithinAbs(3.0, 1e-9)); // yz
}

TEST_CASE("read_lammps_data - charge style skips the q column", "[io]") {
  constexpr std::string_view body = R"(charge style

1 atoms
1 atom types
0.0 8.0 xlo xhi
0.0 8.0 ylo yhi
0.0 8.0 zlo zhi

Atoms

1 1 -0.5 1.0 2.0 3.0
)";
  TmpFile tf("rmc_lmp_charge.dat", body);
  const auto r =
      io::read_lammps_data(tf.path, {"Zr"}, RMC::io::LammpsAtomStyle::Charge);
  REQUIRE(r);
  const auto &s = r->structure;
  REQUIRE(s.size() == 1);
  // x,y,z come after id,type,q — not the charge -0.5.
  REQUIRE_THAT(s.coordinates(0, 0), WithinAbs(1.0, 1e-9));
  REQUIRE_THAT(s.coordinates(0, 1), WithinAbs(2.0, 1e-9));
  REQUIRE_THAT(s.coordinates(0, 2), WithinAbs(3.0, 1e-9));
}

TEST_CASE("read_lammps_data - atom count mismatch errors", "[io]") {
  constexpr std::string_view body = R"(mismatch

5 atoms
1 atom types
0.0 10.0 xlo xhi
0.0 10.0 ylo yhi
0.0 10.0 zlo zhi

Atoms

1 1 0.0 0.0 0.0
2 1 1.0 1.0 1.0
)";
  TmpFile tf("rmc_lmp_mismatch.dat", body);
  const auto r = io::read_lammps_data(tf.path, {"Zr"});
  REQUIRE_FALSE(r);
}

TEST_CASE("read_lammps_data - missing box bounds errors", "[io]") {
  constexpr std::string_view body = R"(no box

1 atoms
1 atom types

Atoms

1 1 0.0 0.0 0.0
)";
  TmpFile tf("rmc_lmp_nobox.dat", body);
  const auto r = io::read_lammps_data(tf.path, {"Zr"});
  REQUIRE_FALSE(r);
}

TEST_CASE("read_lammps_data - no Atoms section errors", "[io]") {
  constexpr std::string_view body = R"(only masses

2 atoms
2 atom types
0.0 10.0 xlo xhi
0.0 10.0 ylo yhi
0.0 10.0 zlo zhi

Masses

1 91.224
2 63.546
)";
  TmpFile tf("rmc_lmp_noatoms.dat", body);
  const auto r = io::read_lammps_data(tf.path, {"Zr", "Cu"});
  REQUIRE_FALSE(r);
}

TEST_CASE("read_lammps_data - missing file errors", "[io]") {
  const auto r = io::read_lammps_data("/nonexistent/path/rmc_missing.dat");
  REQUIRE_FALSE(r);
}

TEST_CASE("write_lammps_data - round-trip preserves coords and species",
          "[io]") {
  TmpFile in("rmc_lmp_rt_in.dat", kLammpsAtomic);
  const auto r1 = io::read_lammps_data(in.path, {"Zr", "Cu"});
  REQUIRE(r1);

  const auto out =
      std::filesystem::temp_directory_path() / "rmc_lmp_rt_out.dat";
  const auto w = io::write_lammps_data(r1->structure, r1->box, out, r1->origin);
  REQUIRE(w);
  const auto r2 = io::read_lammps_data(out, {"Zr", "Cu"});
  std::error_code ec;
  std::filesystem::remove(out, ec);
  REQUIRE(r2);

  const auto &a = r1->structure;
  const auto &b = r2->structure;
  REQUIRE(b.size() == a.size());
  REQUIRE(b.elements == a.elements);
  for (std::size_t i = 0; i < a.size(); ++i)
    for (int k = 0; k < 3; ++k)
      REQUIRE_THAT(b.coordinates(static_cast<Eigen::Index>(i), k),
                   WithinAbs(a.coordinates(static_cast<Eigen::Index>(i), k),
                             1e-6));
  // Box round-trips too.
  REQUIRE_THAT(r2->box(0, 0), WithinAbs(r1->box(0, 0), 1e-6));
  REQUIRE_THAT(r2->box(2, 2), WithinAbs(r1->box(2, 2), 1e-6));
}
