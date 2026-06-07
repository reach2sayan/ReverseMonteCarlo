#include "AtatFormats.hpp"
#include "ClusterEnumerator.hpp"

#include <RMC/constraints/ClusterCorrelationConstraint.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <sstream>
#include <string>
#include <vector>

using Catch::Matchers::WithinAbs;
using RMC::mat3_t;
using RMC::vec3_t;
namespace atat = RMC::atat;

// ============================================================
// Format parsers
// ============================================================

TEST_CASE("parse_sym - identity operation", "[atat]") {
  std::istringstream in("1\n1 0 0\n0 1 0\n0 0 1\n0 0 0\n");
  const auto ops = atat::parse_sym(in);
  REQUIRE(ops.size() == 1);
  REQUIRE(ops[0].rot.isApprox(mat3_t::Identity()));
  REQUIRE(ops[0].trans.isApprox(vec3_t::Zero()));
}

TEST_CASE("parse_clusters - one pair orbit", "[atat]") {
  std::istringstream in("6\n1.0\n2\n0 0 0 0 0\n0.5 0.5 0 0 0\n");
  const auto orbits = atat::parse_clusters(in);
  REQUIRE(orbits.size() == 1);
  REQUIRE(orbits[0].multiplicity == 6.0);
  REQUIRE(orbits[0].points.size() == 2);
  REQUIRE(orbits[0].points[1].coord.isApprox(vec3_t(0.5, 0.5, 0.0)));
  REQUIRE(orbits[0].points[0].site_type == 0);
  REQUIRE(orbits[0].points[0].func == 0);
}

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

// ============================================================
// Trigonometric basis
// ============================================================

TEST_CASE("CorrFuncTable - binary trig basis", "[atat]") {
  const auto t = RMC::CorrFuncTable::trigonometric(2);
  REQUIRE_THAT(t.value(0, 0, 0), WithinAbs(-1.0, 1e-12)); // s=0: -cos(0)
  REQUIRE_THAT(t.value(0, 0, 1), WithinAbs(1.0, 1e-12));  // s=1: -cos(pi)
}

// ============================================================
// Enumerator (the symmetry-faithful port)
// ============================================================

// A 1D chain (simple-cubic primitive, one binary site) with identity+inversion
// symmetry. NN-pair orbit multiplicity = 1, so a 4× supercell has 4 instances:
// {0,1},{1,2},{2,3},{3,0}.
static atat::AtatLattice chain_lattice() {
  atat::AtatLattice lat;
  lat.axes = mat3_t::Identity();
  lat.cell = mat3_t::Identity();
  atat::LatticeSite site;
  site.frac = vec3_t::Zero();
  site.occ = {{"Cu", 0.5}, {"Au", 0.5}};
  lat.sites = {site};
  lat.labels = {"Au", "Cu"};
  return lat;
}

TEST_CASE("enumerate - NN pair instance count on a 1D chain", "[atat]") {
  const auto lat = chain_lattice();
  std::vector<atat::SymOp> sym(2);
  sym[0].rot = mat3_t::Identity();
  sym[1].rot = -mat3_t::Identity(); // inversion
  atat::RawOrbit orbit;
  orbit.multiplicity = 1.0;
  orbit.length = 1.0;
  orbit.points = {{vec3_t(0, 0, 0), 0, 0}, {vec3_t(1, 0, 0), 0, 0}};

  Eigen::Matrix3i sc = Eigen::Matrix3i::Zero();
  sc(0, 0) = 4;
  sc(1, 1) = 1;
  sc(2, 2) = 1;

  auto e = atat::enumerate(lat, sym, {orbit}, sc, /*seed=*/1);
  REQUIRE(e.structure.size() == 4);
  REQUIRE(e.orbits.size() == 1);
  REQUIRE(e.orbits[0].instances.size() == 4); // multiplicity(1) × n_cells(4)
  // Random equiatomic target: point correlation = 0 ⇒ orbit target = 0.
  REQUIRE_THAT(e.orbits[0].target, WithinAbs(0.0, 1e-9));
}

TEST_CASE("enumerate - correlation matches the trig basis", "[atat]") {
  const auto lat = chain_lattice();
  std::vector<atat::SymOp> sym(2);
  sym[0].rot = mat3_t::Identity();
  sym[1].rot = -mat3_t::Identity();
  atat::RawOrbit orbit;
  orbit.multiplicity = 1.0;
  orbit.length = 1.0;
  orbit.points = {{vec3_t(0, 0, 0), 0, 0}, {vec3_t(1, 0, 0), 0, 0}};
  Eigen::Matrix3i sc = Eigen::Matrix3i::Zero();
  sc(0, 0) = 4;
  sc(1, 1) = 1;
  sc(2, 2) = 1;

  auto e = atat::enumerate(lat, sym, {orbit}, sc, 1);
  // Deterministic alternating occupation: Cu(+1) Au(-1) Cu(+1) Au(-1).
  // Sites are generated in chain order (atom i at x=i).
  e.structure.elements = {"Cu", "Au", "Cu", "Au"};
  RMC::ClusterCorrelationConstraint cc{e.structure, e.occ_index, e.table,
                                       e.orbits};
  const auto corr = cc.current_correlations();
  REQUIRE(corr.size() == 1);
  // Every NN pair is (Cu,Au) ⇒ sigma product −1 ⇒ mean −1.
  REQUIRE_THAT(corr[0], WithinAbs(-1.0, 1e-9));
}
