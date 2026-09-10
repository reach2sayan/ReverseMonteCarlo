#include "AtatFormats.hpp"
#include "SeitzClusters.hpp"

#include <RMC/constraints/ClusterCorrelationConstraint.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <ranges>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using Catch::Matchers::WithinAbs;
using RMC::mat3_t;
using RMC::vec3_t;
namespace atat = RMC::atat;

// ============================================================
// Lattice parser
// ============================================================

TEST_CASE("parse_lattice - cubic binary rndstr", "[atat]") {
  std::istringstream in(
      "1 0 0\n0 1 0\n0 0 1\n1 0 0\n0 1 0\n0 0 1\n0 0 0 Cu=0.5,Au=0.5\n");
  const auto lat = atat::parse_lattice(in);
  REQUIRE(lat.cell.isApprox(mat3_t::Identity()));
  REQUIRE(lat.sites.size() == 1);
  REQUIRE(lat.sites[0].occ.size() == 2);
  // labels are globally sorted: Au < Cu.
  REQUIRE(lat.labels == std::vector<std::string>{"Au", "Cu"});
  REQUIRE(lat.occupation_index("Au") == 0);
  REQUIRE(lat.occupation_index("Cu") == 1);
}

TEST_CASE("parse_lattice - 6-number 'a b c al be ga' axes form", "[atat]") {
  // Cubic cell as a/b/c/angles: lattice_vectors() must yield identity axes.
  std::istringstream in(
      "1 1 1 90 90 90\n1 0 0\n0 1 0\n0 0 1\n0 0 0 Cu=0.5 Au=0.5\n");
  const auto lat = atat::parse_lattice(in);
  REQUIRE(lat.axes.isApprox(mat3_t::Identity(), 1e-9));
  REQUIRE(lat.cell.isApprox(mat3_t::Identity()));
  REQUIRE(lat.sites.size() == 1);
  REQUIRE(lat.sites[0].occ.size() == 2);
}

TEST_CASE("parse_lattice - species separator variants", "[atat]") {
  const std::string head = "1 0 0\n0 1 0\n0 0 1\n1 0 0\n0 1 0\n0 0 1\n0 0 0 ";
  // Every ATAT separator must split the species list identically.
  const auto sep = GENERATE("Cu Au", "Cu,Au", "Cu;Au", "Cu/Au");
  std::istringstream in(head + sep + "\n");
  const auto lat = atat::parse_lattice(in);
  REQUIRE(lat.sites.size() == 1);
  REQUIRE(lat.sites[0].occ.size() == 2);
  REQUIRE(lat.labels == std::vector<std::string>{"Au", "Cu"});
}

TEST_CASE("parse_lattice - equiatomic default when occ omitted", "[atat]") {
  // No "=occ" on any species ⇒ each defaults to 1/n.
  std::istringstream in(
      "1 0 0\n0 1 0\n0 0 1\n1 0 0\n0 1 0\n0 0 1\n0 0 0 Cu Au\n");
  const auto lat = atat::parse_lattice(in);
  REQUIRE(lat.sites[0].occ.size() == 2);
  REQUIRE_THAT(lat.sites[0].occ[0].second, WithinAbs(0.5, 1e-12));
  REQUIRE_THAT(lat.sites[0].occ[1].second, WithinAbs(0.5, 1e-12));
}

TEST_CASE("parse_lattice - multiple sites", "[atat]") {
  std::istringstream in("1 0 0\n0 1 0\n0 0 1\n1 0 0\n0 1 0\n0 0 1\n"
                        "0 0 0 Cu=0.5,Au=0.5\n0.5 0.5 0.5 Ni\n");
  const auto lat = atat::parse_lattice(in);
  REQUIRE(lat.sites.size() == 2);
  REQUIRE(lat.sites[1].frac.isApprox(vec3_t(0.5, 0.5, 0.5)));
  REQUIRE(lat.sites[1].occ.size() == 1);
  // Ni is a pure (occ 1.0) site by the equiatomic default of a 1-species list.
  REQUIRE_THAT(lat.sites[1].occ[0].second, WithinAbs(1.0, 1e-12));
  REQUIRE(lat.labels == std::vector<std::string>{"Au", "Cu", "Ni"});
}

TEST_CASE("parse_lattice - malformed site line throws", "[atat]") {
  std::istringstream in("1 0 0\n0 1 0\n0 0 1\n1 0 0\n0 1 0\n0 0 1\n0 0 0\n");
  REQUIRE_THROWS_AS(atat::parse_lattice(in), std::runtime_error);
}

// ============================================================
// Enumeration on seitz
// ============================================================

namespace {

// rndstr.in texts. The chain's cell is 1 × 5 × 5, so a 1.1 Å pair cutoff keeps
// only the x-neighbours (multiplicity 1). fcc is the primitive cell of the
// unit cube (NN distance √2/2). The CsCl-type parent has two binary
// sublattices, NN (√3/2) only between them.
constexpr const char *kChain =
    "1 0 0\n0 1 0\n0 0 1\n1 0 0\n0 5 0\n0 0 5\n0 0 0 Cu=0.5,Au=0.5\n";
constexpr const char *kFcc = "1 0 0\n0 1 0\n0 0 1\n0 0.5 0.5\n0.5 0 0.5\n"
                             "0.5 0.5 0\n0 0 0 Cu=0.5,Au=0.5\n";
constexpr const char *kTwoSublattices =
    "1 0 0\n0 1 0\n0 0 1\n1 0 0\n0 1 0\n0 0 1\n"
    "0 0 0 Cu=0.5,Au=0.5\n0.5 0.5 0.5 Ni=0.5,Ti=0.5\n";

atat::AtatLattice lattice(const char *rndstr) {
  std::istringstream in(rndstr);
  return atat::parse_lattice(in);
}

Eigen::Matrix3i diag(int a, int b, int c) {
  return Eigen::Vector3i(a, b, c).asDiagonal();
}

} // namespace

TEST_CASE("enumerate - NN pair instance count on a 1D chain", "[atat]") {
  const auto e = atat::enumerate(lattice(kChain), diag(4, 1, 1), {{2, 1.1}}, 1);
  REQUIRE(e.structure.size() == 4);
  REQUIRE(e.orbits.size() == 1);
  REQUIRE(e.orbits[0].instance_count() == 4); // multiplicity 1 × 4 cells
  // Random equiatomic state: point correlation 0 ⇒ pair target 0.
  REQUIRE_THAT(e.orbits[0].target, WithinAbs(0.0, 1e-9));
}

TEST_CASE("enumerate - correlation matches the trig basis", "[atat]") {
  auto e = atat::enumerate(lattice(kChain), diag(4, 1, 1), {{2, 1.1}}, 1);
  // Alternate Cu (θ = +1) and Au (θ = −1) along x: every NN pair is (Cu, Au).
  // atomic_numbers carry the occupation index the constraint reads.
  for (std::size_t i = 0; i < e.structure.size(); ++i) {
    const auto r = static_cast<Eigen::Index>(i);
    e.structure.elements[i] =
        std::lround(e.structure.coordinates(r, 0)) % 2 == 0 ? "Cu" : "Au";
    e.structure.atomic_numbers[r] = e.occ_index.at(e.structure.elements[i]);
  }
  RMC::ClusterCorrelationConstraint cc{e.structure, e.occ_index, e.table,
                                       e.orbits};
  const auto corr = cc.current_correlations();
  REQUIRE(corr.size() == 1);
  REQUIRE_THAT(corr[0], WithinAbs(-1.0, 1e-9));
}

TEST_CASE("enumerate - fcc 2x2x2 first pair orbit", "[atat]") {
  const auto e = atat::enumerate(lattice(kFcc), diag(2, 2, 2), {{2, 0.75}}, 3);
  REQUIRE(e.structure.size() == 8);
  REQUIRE(e.orbits.size() == 1);
  REQUIRE(e.orbits[0].instance_count() == 48); // 6 NN pairs per cell × 8 cells
  REQUIRE(std::ranges::count(e.structure.elements, "Cu") == 4);
}

TEST_CASE("enumerate - two sublattices", "[atat]") {
  const auto e = atat::enumerate(lattice(kTwoSublattices), diag(2, 2, 2),
                                 {{2, 0.9}}, 5);
  REQUIRE(e.structure.size() == 16);
  REQUIRE(e.table.size() == 2);
  REQUIRE(std::ranges::count(e.structure.residues, "SL0") == 8);
  REQUIRE(std::ranges::count(e.structure.residues, "SL1") == 8);
  for (std::size_t i = 0; i < e.structure.size(); ++i) {
    // Each sublattice holds only its own species.
    const auto &el = e.structure.elements[i];
    REQUIRE((el == "Cu" || el == "Au") == (e.structure.residues[i] == "SL0"));
  }
  REQUIRE(e.orbits.size() == 1); // the Cu/Au–Ni/Ti nearest-neighbour pair
  REQUIRE(e.orbits[0].instance_count() == 8 * 8);
}

// Cross-check against ATAT corrdump when RMC_CORRDUMP names its binary: the
// correlations of a random supercell must match corrdump's for the same
// structure (both sorted: the orbit orders differ only in ties).
TEST_CASE("enumerate - correlations match corrdump", "[atat][corrdump]") {
  const char *corrdump = std::getenv("RMC_CORRDUMP");
  if (corrdump == nullptr) {
    SKIP("RMC_CORRDUMP not set");
  }
  struct Case {
    const char *name;
    const char *rndstr;
    Eigen::Matrix3i sc;
    atat::Diameters diameters;
  };
  const std::vector<Case> cases{
      {"chain", kChain, diag(4, 1, 1), {{2, 1.1}}},
      {"fcc pairs", kFcc, diag(2, 2, 2), {{2, 1.1}}},
      {"fcc pairs + triplets", kFcc, diag(2, 2, 2), {{2, 1.1}, {3, 0.75}}},
      {"fcc 3x3x3 pairs + triplets", kFcc, diag(3, 3, 3), {{2, 1.1}, {3, 1.1}}},
      {"two sublattices", kTwoSublattices, diag(2, 2, 2), {{2, 1.1}}},
  };
  const auto dir = std::filesystem::temp_directory_path() / "rmc_corrdump_oracle";
  for (const Case &c : cases) {
    INFO(c.name);
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    std::ofstream(dir / "rndstr.in") << c.rndstr;
    const auto e = atat::enumerate(lattice(c.rndstr), c.sc, c.diameters, 11);
    atat::write_str_out(dir / "str.out", e.axes, e.supercell, e.frac_positions,
                        e.structure.elements);

    std::string cmd = std::format(
        "cd '{}' && '{}' -l=rndstr.in -ro -noe -nop -sig=12 -s=str.out",
        dir.string(), corrdump);
    for (const auto &[body, d] : c.diameters) {
      cmd += std::format(" -{}={}", body, d);
    }
    REQUIRE(std::system((cmd + " > corr.out 2> corr.err").c_str()) == 0);
    std::ifstream f(dir / "corr.out");
    std::string line;
    std::getline(f, line);
    std::istringstream row(line);
    std::vector<double> expected{std::istream_iterator<double>(row), {}};

    RMC::ClusterCorrelationConstraint cc{e.structure, e.occ_index, e.table,
                                         e.orbits};
    auto got = cc.current_correlations();
    std::ranges::sort(expected);
    std::ranges::sort(got);
    CAPTURE(got, expected);
    REQUIRE(got.size() == expected.size());
    for (const auto [g, x] : std::views::zip(got, expected)) {
      REQUIRE_THAT(g, WithinAbs(x, 1e-9));
    }
  }
}
