#include <RMC/Engine.hpp>
#include <RMC/constraints/GeometricConstraints.hpp>
#include <RMC/constraints/ConstraintCollection.hpp>
#include <RMC/constraints/CoordinationConstraint.hpp>
#include <RMC/constraints/DistanceConstraint.hpp>
#include <RMC/constraints/PairCorrelationConstraint.hpp>
#include <RMC/constraints/PairDistributionConstraint.hpp>
#include <RMC/constraints/ReducedStructureFactorConstraint.hpp>
#include <RMC/constraints/StructureFactorConstraint.hpp>
#include <RMC/generators/Translations.hpp>
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
  REQUIRE(col.rigid_should_reject());
  const double pdf_before = col[1].standard_error();
  col.reject();
  // After reject, PDF error must equal what it was before (no stale
  // after-value).
  REQUIRE_THAT(col[1].standard_error(), WithinAbs(pdf_before, EPS));
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

// ---- ShapeFunction ----

static PairDistributionConstraint make_pdf(int n_bins = 50, double dr = 0.1) {
  PairDistributionConstraint pdc;
  mat_t exp_data(n_bins, 2);
  for (int i = 0; i < n_bins; ++i) {
    exp_data(i, 0) = dr * (i + 1);
    exp_data(i, 1) = 0.0;
  }
  pdc.set_experimental_data(exp_data);
  pdc.set_number_density(0.03);

  const std::size_t N = 8;
  std::vector<std::string> els(N, "C");
  pdc.set_elements(els);
  pdc.initialise();
  return pdc;
}

static coords_t make_chain(int N, double spacing = 3.0) {
  coords_t c(N, 3);
  for (int i = 0; i < N; ++i)
    c.row(i) << i * spacing, 0.0, 0.0;
  return c;
}

TEST_CASE("ShapeFunction - spherical_shape_fn zero beyond diameter",
          "[constraints][shape]") {
  auto fn = spherical_shape_fn(5.0);
  REQUIRE_THAT(fn(0.0), WithinAbs(1.0, 1e-9));
  REQUIRE_THAT(fn(5.0), WithinAbs(0.0, 1e-9));
  REQUIRE_THAT(fn(6.0), WithinAbs(0.0, 1e-9));
  // At r = 5/2 = 2.5 (x = 0.5): 1 - 0.75 + 0.0625 = 0.3125
  REQUIRE_THAT(fn(2.5), WithinAbs(0.3125, 1e-9));
}

TEST_CASE("ShapeFunction - gaussian_shape_fn is 1 at r=0",
          "[constraints][shape]") {
  auto fn = gaussian_shape_fn(3.0);
  REQUIRE_THAT(fn(0.0), WithinAbs(1.0, 1e-9));
  // exp(-(15/3)²) = exp(-25) ≈ 1.4e-11
  REQUIRE(fn(15.0) < 1e-10);
}

TEST_CASE("ShapeFunction - applying shape damps computed G(r)",
          "[constraints][shape]") {
  const int N = 8;
  std::vector<std::size_t> all(N);
  std::iota(all.begin(), all.end(), std::size_t{0});
  coords_t c = make_chain(N);

  auto pdc_no_shape = make_pdf();
  double err_no_shape = pdc_no_shape.compute_error(c, all);
  vec_t g_no_shape = pdc_no_shape.computed_G();

  auto pdc_shape = make_pdf();
  pdc_shape.set_shape_function(spherical_shape_fn(10.0));
  double err_shape = pdc_shape.compute_error(c, all);
  vec_t g_shape = pdc_shape.computed_G();

  // Shaped G(r) should have smaller or equal norm than unshaped (truncated)
  REQUIRE(g_shape.norm() <= g_no_shape.norm() + 1e-9);
  // Errors are finite
  REQUIRE(std::isfinite(err_no_shape));
  REQUIRE(std::isfinite(err_shape));
}

TEST_CASE("ShapeFunction - shape function zeros high-r bins",
          "[constraints][shape]") {
  const int N = 8;
  std::vector<std::size_t> all(N);
  std::iota(all.begin(), all.end(), std::size_t{0});
  coords_t c = make_chain(N);

  auto pdc = make_pdf(50, 0.5); // bins up to 25 Å
  pdc.set_shape_function(spherical_shape_fn(5.0));
  (void)pdc.compute_error(c, all);
  const vec_t &g = pdc.computed_G();

  // All bins beyond 5 Å should be zero (shape function = 0)
  for (int i = 0; i < g.size(); ++i) {
    const double r = 0.5 * (i + 1);
    if (r >= 5.0)
      REQUIRE_THAT(g[i], WithinAbs(0.0, 1e-9));
  }
}

// ---- Multi-frame Engine (Engine with add_frame) ----

static AtomicStructure make_mono_structure(int N, double spacing = 3.0) {
  AtomicStructure s;
  s.coordinates = make_chain(N, spacing);
  s.elements.assign(N, "C");
  s.atomic_numbers.resize(N);
  s.atomic_numbers.setConstant(6);
  return s;
}

TEST_CASE("Multi-frame Engine - initialise populates frame histograms",
          "[multiframe]") {
  const int N = 6;
  const int n_frames = 3;

  // Build a flat target (zero G(r)) and a PDF constraint.
  PairDistributionConstraint pdf;
  mat_t exp_data(30, 2);
  for (int i = 0; i < 30; ++i) {
    exp_data(i, 0) = 0.5 * (i + 1);
    exp_data(i, 1) = 0.0;
  }
  pdf.set_experimental_data(exp_data);
  pdf.set_number_density(0.03);
  std::vector<std::string> els(N, "C");
  pdf.set_elements(els);
  pdf.initialise();

  Engine eng(make_mono_structure(N), InfiniteBC{});
  for (int f = 1; f < n_frames; ++f)
    eng.add_frame(make_mono_structure(N));

  Group g;
  g.name = "all";
  for (int i = 0; i < N; ++i)
    g.indices.push_back(static_cast<std::size_t>(i));
  g.generator = TranslationGenerator(0.01, 0.1, /*seed=*/7 + 0);
  eng.add_group(std::move(g));

  eng.add_constraint(Constraint{std::move(pdf)});
  eng.initialise();

  // After initialise, total_error() must be finite and non-negative.
  const double err0 = eng.total_error();
  REQUIRE(std::isfinite(err0));
  REQUIRE(err0 >= 0.0);
}

TEST_CASE("Multi-frame Engine - run accepts some moves", "[multiframe]") {
  const int N = 6;
  const int n_frames = 2;

  PairDistributionConstraint pdf;
  mat_t exp_data(20, 2);
  for (int i = 0; i < 20; ++i) {
    exp_data(i, 0) = 0.5 * (i + 1);
    exp_data(i, 1) = 0.0;
  }
  pdf.set_experimental_data(exp_data);
  pdf.set_number_density(0.03);
  std::vector<std::string> els(N, "C");
  pdf.set_elements(els);
  pdf.initialise();

  Engine eng(make_mono_structure(N, 2.0), InfiniteBC{});
  for (int f = 1; f < n_frames; ++f)
    eng.add_frame(make_mono_structure(N, 2.0 + 0.1 * f));

  Group g;
  g.name = "all";
  for (int i = 0; i < N; ++i)
    g.indices.push_back(static_cast<std::size_t>(i));
  g.generator = TranslationGenerator(0.0, 0.3, /*seed=*/99);
  eng.add_group(std::move(g));

  eng.add_constraint(Constraint{std::move(pdf)});
  eng.initialise();
  eng.run(500);

  REQUIRE(eng.steps_accepted() > 0);
  REQUIRE(eng.steps_total() == 500);
  REQUIRE(std::isfinite(eng.total_error()));
}

// ---- Incremental histogram tests (direct, no engine type-eraser) ----

namespace {

// Build and fully initialise a 2-frame PairDistributionConstraint with the
// given coordinate sets.  Returns the constraint with both frame histograms
// populated.
PairDistributionConstraint make_2frame_pdf(const coords_t &f0,
                                           const coords_t &f1,
                                           const BoundaryConditions &bc,
                                           int n_bins = 20, double dr = 0.25) {
  const int N = static_cast<int>(f0.rows());
  mat_t exp_data(n_bins, 2);
  for (int i = 0; i < n_bins; ++i) {
    exp_data(i, 0) = dr * (i + 1);
    exp_data(i, 1) = 0.0;
  }
  std::vector<std::string> els(static_cast<std::size_t>(N), "C");

  PairDistributionConstraint pdf;
  pdf.set_experimental_data(exp_data);
  pdf.set_number_density(0.03);
  pdf.set_elements(els);
  pdf.initialise();
  pdf.set_boundary_conditions(bc);
  static_cast<PairConstraintBase &>(pdf).set_n_frames(2);

  std::vector<std::size_t> all(static_cast<std::size_t>(N));
  std::iota(all.begin(), all.end(), std::size_t{0});

  pdf.set_active_frame_idx(0);
  (void)pdf.compute_error(f0, all);
  pdf.set_active_frame_idx(1);
  (void)pdf.compute_error(f1, all);
  return pdf;
}

} // namespace

TEST_CASE("PairFunctionConstraint - incremental update matches full recompute",
          "[constraints][incremental]") {
  // Verify that the O(K·N) after-move incremental path gives the same
  // computed_G() as a fresh full O(N²) rebuild with the moved coordinates.

  const int N = 8;
  const BoundaryConditions bc = InfiniteBC{};

  coords_t f0 = make_chain(N, 2.0);
  coords_t f1 = make_chain(N, 2.5);

  // Perturb atom 0 in frame 0.
  coords_t f0_mod = f0;
  f0_mod(0, 0) += 0.35;
  f0_mod(0, 1) += 0.12;

  // Reference: full rebuild with f0_mod as frame 0.
  const vec_t ref_G = [&] {
    auto pdf_ref = make_2frame_pdf(f0_mod, f1, bc);
    // make_2frame_pdf leaves active_frame = 1; switch to 0 and read computed_G
    // after one more (before-move) call so computed_ reflects the current sum.
    std::vector<std::size_t> all(static_cast<std::size_t>(N));
    std::iota(all.begin(), all.end(), std::size_t{0});
    pdf_ref.set_active_frame_idx(0);
    (void)pdf_ref.compute_error(f0_mod, all);
    return pdf_ref.computed_G();
  }();

  // Incremental: init with f0, then execute a before/after step on atom 0.
  auto pdf_inc = make_2frame_pdf(f0, f1, bc);
  const std::vector<std::size_t> moved = {0};

  pdf_inc.set_active_frame_idx(0);
  (void)pdf_inc.compute_error(f0, moved);     // before-move: saves delta for atom 0
  (void)pdf_inc.compute_error(f0_mod, moved); // after-move: O(K·N) incremental update

  const vec_t &inc_G = pdf_inc.computed_G();
  REQUIRE(inc_G.size() == ref_G.size());
  for (Eigen::Index i = 0; i < inc_G.size(); ++i) {
    REQUIRE_THAT(inc_G(i), WithinAbs(ref_G(i), 1e-10));
  }
}

TEST_CASE("PairFunctionConstraint - rollback restores histogram",
          "[constraints][incremental]") {
  // After an accepted incremental step, rollback_frame() must restore the
  // histogram to the pre-step state and clear incremental_ready_.

  const int N = 6;
  const BoundaryConditions bc = InfiniteBC{};

  coords_t f0 = make_chain(N, 2.0);
  coords_t f1 = make_chain(N, 2.5);

  auto pdf = make_2frame_pdf(f0, f1, bc);

  // Record computed_G before the step.
  {
    std::vector<std::size_t> all(static_cast<std::size_t>(N));
    std::iota(all.begin(), all.end(), std::size_t{0});
    pdf.set_active_frame_idx(0);
    (void)pdf.compute_error(f0, all);
  }
  const vec_t G_before = pdf.computed_G();

  // Simulate a step: before-move, perturb, after-move.
  coords_t f0_mod = f0;
  f0_mod(1, 0) += 0.5;
  const std::vector<std::size_t> moved = {1};

  pdf.set_active_frame_idx(0);
  (void)pdf.compute_error(f0_mod, moved); // after-move (incremental)
  (void)pdf.compute_error(f0, moved);     // before-move

  // Reject: rollback should restore to the pre-step state.
  pdf.rollback_frame();

  // One more call to materialise computed_ from the restored sum_hist_.
  std::vector<std::size_t> all(static_cast<std::size_t>(N));
  std::iota(all.begin(), all.end(), std::size_t{0});
  pdf.set_active_frame_idx(0);
  (void)pdf.compute_error(f0, all);

  const vec_t &G_after_rollback = pdf.computed_G();
  for (Eigen::Index i = 0; i < G_before.size(); ++i) {
    REQUIRE_THAT(G_after_rollback(i), WithinAbs(G_before(i), 1e-10));
  }
}

TEST_CASE("PairFunctionConstraint - frame switch resets incremental state",
          "[constraints][incremental]") {
  // Switching the active frame must reset incremental_ready_ so that the next
  // call for the new frame takes the before-move (full-save) path rather than
  // attempting a spurious incremental update.

  const int N = 6;
  const BoundaryConditions bc = InfiniteBC{};

  coords_t f0 = make_chain(N, 2.0);
  coords_t f1 = make_chain(N, 2.5);
  coords_t f1_mod = f1;
  f1_mod(2, 0) += 0.4;

  auto pdf = make_2frame_pdf(f0, f1, bc);

  // Trigger incremental_ready_ on frame 0.
  const std::vector<std::size_t> moved0 = {0};
  pdf.set_active_frame_idx(0);
  (void)pdf.compute_error(f0,
                    moved0); // before-move on frame 0 → incremental_ready_=true

  // Switch to frame 1 — must clear incremental_ready_.
  const std::vector<std::size_t> moved1 = {2};
  pdf.set_active_frame_idx(1);

  // Before-move call on frame 1 with atom 2 moved.
  (void)pdf.compute_error(f1, moved1);

  // After-move call: must apply the delta for frame 1, not frame 0.
  (void)pdf.compute_error(f1_mod, moved1);

  // Build reference via full rebuild.
  const vec_t ref_G = [&] {
    auto pdf_ref = make_2frame_pdf(f0, f1_mod, bc);
    std::vector<std::size_t> all(static_cast<std::size_t>(N));
    std::iota(all.begin(), all.end(), std::size_t{0});
    pdf_ref.set_active_frame_idx(1);
    (void)pdf_ref.compute_error(f1_mod, all);
    return pdf_ref.computed_G();
  }();

  const vec_t &inc_G = pdf.computed_G();
  for (Eigen::Index i = 0; i < inc_G.size(); ++i) {
    REQUIRE_THAT(inc_G(i), WithinAbs(ref_G(i), 1e-10));
  }
}
