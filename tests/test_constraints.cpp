#include <RMC/constraints/AngleConstraint.hpp>
#include <RMC/constraints/BondConstraint.hpp>
#include <RMC/constraints/ConstraintCollection.hpp>
#include <RMC/constraints/CoordinationConstraint.hpp>
#include <RMC/constraints/DihedralAngleConstraint.hpp>
#include <RMC/constraints/DistanceConstraint.hpp>
#include <RMC/constraints/PairCorrelationConstraint.hpp>
#include <RMC/constraints/PairDistributionConstraint.hpp>
#include <RMC/constraints/ReducedStructureFactorConstraint.hpp>
#include <RMC/constraints/StructureFactorConstraint.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <numbers>
#include <numeric>

using namespace RMC;
using Catch::Matchers::WithinAbs;

static constexpr real_t EPS = 1e-8;

static coords_t make2(double ax, double bx) {
  coords_t c(2, 3);
  c << ax, 0, 0, bx, 0, 0;
  return c;
}

// ---- BondConstraint ----
TEST_CASE("BondConstraint - satisfied bond has zero error", "[constraints]") {
  BondConstraint bc;
  bc.add_bond(0, 1, 1.0, 2.0);
  coords_t c = make2(0.0, 1.5);
  std::vector<index_t> all = {0, 1};
  REQUIRE_THAT(bc.compute_error(c, all), WithinAbs(0.0, EPS));
}

TEST_CASE("BondConstraint - too-short bond accumulates error",
          "[constraints]") {
  BondConstraint bc;
  bc.add_bond(0, 1, 1.5, 2.5);
  coords_t c = make2(0.0, 1.0);
  std::vector<index_t> all = {0, 1};
  REQUIRE_THAT(bc.compute_error(c, all), WithinAbs(0.5, EPS));
}

TEST_CASE("BondConstraint - should_reject after worsening move",
          "[constraints]") {
  BondConstraint bc;
  bc.add_bond(0, 1, 1.0, 2.0);
  Constraint c = std::move(bc);

  coords_t good = make2(0.0, 1.5);
  coords_t bad = make2(0.0, 0.5);
  std::vector<index_t> all = {0, 1};
  c.compute_before_move(good, all);
  c.compute_after_move(bad, all);
  REQUIRE(c.should_reject());
}

TEST_CASE("BondConstraint - accept then error_before updates",
          "[constraints]") {
  BondConstraint bc;
  bc.add_bond(0, 1, 1.0, 2.0);
  Constraint c = std::move(bc);

  coords_t c1 = make2(0.0, 1.2);
  coords_t c2 = make2(0.0, 1.8);
  std::vector<index_t> all = {0, 1};
  c.compute_before_move(c1, all);
  c.compute_after_move(c2, all);
  c.accept();
  REQUIRE_THAT(c.standard_error(), WithinAbs(0.0, EPS));
}

// ---- AngleConstraint ----
TEST_CASE("AngleConstraint - 90 degree angle satisfied", "[constraints]") {
  AngleConstraint ac;
  ac.add_angle(0, 1, 2, 0.0, std::numbers::pi);
  coords_t c(3, 3);
  c << 1, 0, 0, 0, 0, 0, 0, 1, 0;
  std::vector<index_t> all = {0, 1, 2};
  REQUIRE_THAT(ac.compute_error(c, all), WithinAbs(0.0, EPS));
}

TEST_CASE("AngleConstraint - angle below minimum accumulates error",
          "[constraints]") {
  AngleConstraint ac;
  ac.add_angle(0, 1, 2, 2.0, std::numbers::pi);
  coords_t c(3, 3);
  c << 1, 0, 0, 0, 0, 0, 0, 1, 0;
  std::vector<index_t> all = {0, 1, 2};
  REQUIRE(ac.compute_error(c, all) > 0.0);
}

// ---- DihedralAngleConstraint ----
TEST_CASE("DihedralAngleConstraint - zero dihedral accepted", "[constraints]") {
  DihedralAngleConstraint dc;
  dc.add_dihedral(0, 1, 2, 3, -std::numbers::pi, std::numbers::pi);
  coords_t c(4, 3);
  c << 0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0;
  std::vector<index_t> all = {0, 1, 2, 3};
  REQUIRE_THAT(dc.compute_error(c, all), WithinAbs(0.0, EPS));
}

// ---- PairDistributionConstraint ----
TEST_CASE("PairDistributionConstraint - large error for random structure vs "
          "flat G(r)",
          "[constraints]") {
  PairDistributionConstraint pdc;
  mat_t exp(50, 2);
  for (int i = 0; i < 50; ++i) {
    exp(i, 0) = 0.1 * (i + 1);
    exp(i, 1) = 0.0;
  }
  pdc.set_experimental_data(exp);
  pdc.set_number_density(0.03);
  pdc.initialise();

  const int N = 8;
  coords_t c(N, 3);
  for (int i = 0; i < N; ++i)
    c.row(i) << i * 3.0, 0.0, 0.0;
  std::vector<index_t> all(N);
  std::iota(all.begin(), all.end(), 0);

  real_t err = pdc.compute_error(c, all);
  REQUIRE(std::isfinite(err));
  REQUIRE(err >= 0.0);
}

// ---- PairCorrelationConstraint (via PairFunctionConstraint<PCF>) ----
TEST_CASE("PairCorrelationConstraint - finite non-negative error",
          "[constraints]") {
  PairCorrelationConstraint pcc;
  mat_t exp(50, 2);
  for (int i = 0; i < 50; ++i) {
    exp(i, 0) = 0.1 * (i + 1);
    exp(i, 1) = 0.0;
  }
  pcc.set_experimental_data(exp);
  pcc.set_number_density(0.03);
  pcc.initialise();

  const int N = 8;
  coords_t c(N, 3);
  for (int i = 0; i < N; ++i)
    c.row(i) << i * 3.0, 0.0, 0.0;
  std::vector<index_t> all(N);
  std::iota(all.begin(), all.end(), 0);

  real_t err = pcc.compute_error(c, all);
  REQUIRE(std::isfinite(err));
  REQUIRE(err >= 0.0);
}

TEST_CASE("PDF and PCF give different errors on same structure",
          "[constraints]") {
  // Non-zero experimental data so the scale-factor optimisation doesn't
  // trivially zero out both errors.
  auto make_exp = []() {
    mat_t exp(50, 2);
    for (int i = 0; i < 50; ++i) {
      exp(i, 0) = 0.1 * (i + 1);
      exp(i, 1) = 1.0; // constant non-zero target
    }
    return exp;
  };

  PairDistributionConstraint pdf;
  pdf.set_experimental_data(make_exp());
  pdf.set_number_density(0.03);
  pdf.initialise();

  PairCorrelationConstraint pcf;
  pcf.set_experimental_data(make_exp());
  pcf.set_number_density(0.03);
  pcf.initialise();

  const int N = 8;
  coords_t c(N, 3);
  for (int i = 0; i < N; ++i)
    c.row(i) << i * 3.0, 0.0, 0.0;
  std::vector<index_t> all(N);
  std::iota(all.begin(), all.end(), 0);

  // G(r) = 4πrρ₀·F(r) so the residuals must differ.
  REQUIRE(pdf.compute_error(c, all) != pcf.compute_error(c, all));
}

TEST_CASE("PairFunctionConstraint - exclude_intra skips same-molecule pairs",
          "[constraints]") {
  // 4 atoms: molecule 0 = atoms {0,1}, molecule 1 = atoms {2,3}.
  // Place each molecule pair at distance 1.0 (intra) and inter at 5.0.
  // With exclude_intra the histogram should be sparser.
  // Non-zero target so errors are not trivially zero.
  mat_t exp(20, 2);
  for (int i = 0; i < 20; ++i) {
    exp(i, 0) = 0.5 * (i + 1);
    exp(i, 1) = 1.0;
  }

  coords_t c(4, 3);
  c << 0.0, 0.0, 0.0, // atom 0, mol 0
      1.0, 0.0, 0.0,  // atom 1, mol 0  (intra dist = 1.0)
      5.0, 0.0, 0.0,  // atom 2, mol 1  (inter dist from atom 0 = 5.0)
      6.0, 0.0, 0.0;  // atom 3, mol 1

  std::vector<std::size_t> mol_ids = {0, 0, 1, 1};
  std::vector<index_t> all = {0, 1, 2, 3};

  PairDistributionConstraint with_intra, without_intra;
  for (auto *pdc : {&with_intra, &without_intra}) {
    pdc->set_experimental_data(exp);
    pdc->set_number_density(0.03);
    pdc->set_molecule_ids(mol_ids);
    pdc->initialise();
  }
  without_intra.set_exclude_intra(true);

  real_t err_with = with_intra.compute_error(c, all);
  real_t err_without = without_intra.compute_error(c, all);
  REQUIRE(err_with != err_without);
}

// ---- Computation cost ordering ----
TEST_CASE("ConstraintCollection - cheap constraint runs before expensive one",
          "[constraints]") {
  // BondConstraint (cost 1.0) must appear before PairDistributionConstraint
  // (cost 1e6) regardless of insertion order.
  ConstraintCollection col;
  {
    PairDistributionConstraint pdc;
    mat_t exp(10, 2);
    for (int i = 0; i < 10; ++i) {
      exp(i, 0) = 0.1 * (i + 1);
      exp(i, 1) = 0.0;
    }
    pdc.set_experimental_data(exp);
    pdc.set_number_density(0.03);
    pdc.initialise();
    col.add(std::move(pdc)); // expensive added first
  }
  {
    BondConstraint b;
    b.add_bond(0, 1, 1.0, 2.0);
    col.add(std::move(b)); // cheap added second
  }
  // After sorted insert: bond (cost 1.0) must be at index 0.
  REQUIRE(col[0].name() == "BondConstraint");
  REQUIRE(col[1].name() == "PairDistributionConstraint");
}

TEST_CASE("ConstraintCollection - short-circuit skips expensive constraint "
          "after cheap rejection",
          "[constraints]") {
  // Bond violates → PDF should not be computed (error stays at 0 / err_before).
  ConstraintCollection col;
  {
    BondConstraint b;
    b.add_bond(0, 1, 1.0, 2.0); // good when dist=1.5, bad when dist=0.3
    col.add(std::move(b));
  }
  {
    PairDistributionConstraint pdc;
    mat_t exp(10, 2);
    for (int i = 0; i < 10; ++i) {
      exp(i, 0) = 0.1 * (i + 1);
      exp(i, 1) = 0.0;
    }
    pdc.set_experimental_data(exp);
    pdc.set_number_density(0.03);
    pdc.initialise();
    col.add(std::move(pdc));
  }

  coords_t good = make2(0.0, 1.5);
  coords_t bad = make2(0.0, 0.3);
  std::vector<index_t> all = {0, 1};

  col.compute_before_move(good, all);
  col.compute_after_move(bad, all); // bond fails → PDF skipped

  // Collection should reject and PDF's standard_error() equals its before-error
  // (set via reject() resetting err_after_ = err_before_).
  REQUIRE(col.should_reject());
  const double pdf_before = col[1].standard_error();
  col.reject();
  // After reject, PDF error must equal what it was before (no stale
  // after-value).
  REQUIRE_THAT(col[1].standard_error(), WithinAbs(pdf_before, EPS));
}

// ---- ConstraintCollection ----
TEST_CASE("ConstraintCollection - rejects when any constraint rejects",
          "[constraints]") {
  ConstraintCollection col;
  BondConstraint b;
  b.add_bond(0, 1, 1.0, 2.0);
  col.add(std::move(b));

  coords_t good = make2(0.0, 1.5);
  coords_t bad = make2(0.0, 0.3);
  std::vector<index_t> all = {0, 1};
  col.compute_before_move(good, all);
  col.compute_after_move(bad, all);
  REQUIRE(col.should_reject());
}

TEST_CASE("ConstraintCollection - accepts when all constraints pass",
          "[constraints]") {
  ConstraintCollection col;
  BondConstraint b;
  b.add_bond(0, 1, 1.0, 2.0);
  col.add(std::move(b));

  coords_t c1 = make2(0.0, 1.2);
  coords_t c2 = make2(0.0, 1.8);
  std::vector<index_t> all = {0, 1};
  col.compute_before_move(c1, all);
  col.compute_after_move(c2, all);
  REQUIRE_FALSE(col.should_reject());
}

// ---- RigidConstraintBase ----
TEST_CASE(
    "RigidConstraintBase - standard_error is 0, should_reject still works",
    "[constraints][rigid]") {
  BondConstraint bc;
  bc.add_bond(0, 1, 1.0, 2.0);
  Constraint c = std::move(bc);

  coords_t good = make2(0.0, 1.5);
  coords_t bad = make2(0.0, 0.3); // too short: violates [1.0, 2.0]
  std::vector<index_t> all = {0, 1};

  c.compute_before_move(good, all);
  c.compute_after_move(bad, all);

  REQUIRE(c.is_rigid());
  REQUIRE(c.should_reject());
  REQUIRE_THAT(c.standard_error(), WithinAbs(0.0, EPS));
}

TEST_CASE("ConstraintCollection - rigid constraint excluded from total_error",
          "[constraints][rigid]") {
  ConstraintCollection col;
  BondConstraint b;
  b.add_bond(0, 1, 0.5, 3.0); // always satisfied
  col.add(std::move(b));

  PairDistributionConstraint pdc;
  mat_t exp(10, 2);
  for (int i = 0; i < 10; ++i) {
    exp(i, 0) = 0.5 + 0.1 * i;
    exp(i, 1) = 0.0;
  }
  pdc.set_experimental_data(exp);
  pdc.set_number_density(0.03);
  pdc.initialise();
  col.add(std::move(pdc));

  coords_t c1 = make2(0.0, 1.5);
  std::vector<index_t> all = {0, 1};
  col.compute_before_move(c1, all);
  col.compute_after_move(c1, all);

  // Bond is rigid: its standard_error() == 0; total_error == PDF error only.
  REQUIRE_THAT(col[0].standard_error(), WithinAbs(0.0, EPS));
  REQUIRE_THAT(col.total_error(), WithinAbs(col[1].standard_error(), EPS));
}

// ---- SingularConstraintBase ----
TEST_CASE("SingularConstraintBase - is_singular flag",
          "[constraints][singular]") {
  PairDistributionConstraint pdc;
  mat_t exp(10, 2);
  for (int i = 0; i < 10; ++i) {
    exp(i, 0) = 0.5 + 0.1 * i;
    exp(i, 1) = 0.0;
  }
  pdc.set_experimental_data(exp);
  pdc.set_number_density(0.03);
  pdc.initialise();
  Constraint c = std::move(pdc);

  REQUIRE(c.is_singular());
  REQUIRE_FALSE(c.is_rigid());
}

// ---- ReducedStructureFactorConstraint ----
static ReducedStructureFactorConstraint make_rfq(int nQ = 20) {
  ReducedStructureFactorConstraint rfq;
  mat_t data(nQ, 2);
  for (int i = 0; i < nQ; ++i) {
    data(i, 0) = 0.5 * (i + 1); // Q values
    data(i, 1) = 0.0;           // F(Q) target
  }
  rfq.set_experimental_data(data);
  rfq.set_number_density(0.03);
  rfq.initialise();
  return rfq;
}

TEST_CASE("ReducedStructureFactorConstraint - name and cost", "[constraints]") {
  ReducedStructureFactorConstraint rfq = make_rfq();
  Constraint c = std::move(rfq);
  REQUIRE(c.name() == "ReducedStructureFactorConstraint");
  REQUIRE(c.computation_cost() >= 1e6);
}

TEST_CASE("ReducedStructureFactorConstraint - is_singular not rigid",
          "[constraints]") {
  Constraint c = make_rfq();
  REQUIRE(c.is_singular());
  REQUIRE_FALSE(c.is_rigid());
}

TEST_CASE(
    "ReducedStructureFactorConstraint - compute_error finite non-negative",
    "[constraints]") {
  ReducedStructureFactorConstraint rfq = make_rfq();
  const int N = 8;
  coords_t c(N, 3);
  for (int i = 0; i < N; ++i)
    c.row(i) << i * 3.0, 0.0, 0.0;
  std::vector<index_t> all(N);
  std::iota(all.begin(), all.end(), 0);

  real_t err = rfq.compute_error(c, all);
  REQUIRE(std::isfinite(err));
  REQUIRE(err >= 0.0);
}

TEST_CASE("ReducedStructureFactorConstraint - F(Q) differs from S(Q)",
          "[constraints]") {
  // Both constraints fit the same zero target with the same structure.
  // The two Fourier kernels differ (1/Q factor), so computed values must
  // differ.
  const int nQ = 20;
  mat_t data(nQ, 2);
  for (int i = 0; i < nQ; ++i) {
    data(i, 0) = 0.5 * (i + 1);
    data(i, 1) = 1.0; // non-zero target so scale matters
  }

  ReducedStructureFactorConstraint rfq;
  rfq.set_experimental_data(data);
  rfq.set_number_density(0.03);
  rfq.initialise();

  StructureFactorConstraint sfq;
  sfq.set_experimental_data(data);
  sfq.set_number_density(0.03);
  sfq.initialise();

  const int N = 8;
  coords_t c(N, 3);
  for (int i = 0; i < N; ++i)
    c.row(i) << i * 3.0, 0.0, 0.0;
  std::vector<index_t> all(N);
  std::iota(all.begin(), all.end(), 0);

  real_t err_f = rfq.compute_error(c, all);
  real_t err_s = sfq.compute_error(c, all);
  REQUIRE(std::isfinite(err_f));
  REQUIRE(std::isfinite(err_s));
  // F(Q) = Q·(S(Q)−1) ≠ S(Q) for Q ≠ 1 or S(Q) ≠ 0
  REQUIRE(err_f != err_s);
}

TEST_CASE("ReducedStructureFactorConstraint - accept/reject cycle",
          "[constraints]") {
  ReducedStructureFactorConstraint rfq = make_rfq();
  Constraint c = std::move(rfq);

  const int N = 4;
  coords_t c1(N, 3), c2(N, 3);
  for (int i = 0; i < N; ++i) {
    c1.row(i) << i * 3.0, 0.0, 0.0;
    c2.row(i) << i * 3.0 + 0.5, 0.0, 0.0;
  }
  std::vector<index_t> all(N);
  std::iota(all.begin(), all.end(), 0);

  c.compute_before_move(c1, all);
  c.compute_after_move(c2, all);
  double err_after = c.standard_error();
  c.accept();
  // After accept, standard_error() should reflect the accepted (after) state.
  REQUIRE_THAT(c.standard_error(), WithinAbs(err_after, EPS));
}
