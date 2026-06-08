#include <RMC/analysis/RadialDistribution.hpp>
#include <RMC/core/BoundaryConditions.hpp>
#include <RMC/core/Types.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <numbers>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

using Catch::Matchers::WithinAbs;
using RMC::coords_t;
using RMC::vec3_t;
namespace an = RMC::analysis;

namespace {

RMC::BoundaryConditions cubic_bc(double L) {
  RMC::mat3_t box = RMC::mat3_t::Zero();
  box(0, 0) = box(1, 1) = box(2, 2) = L;
  return RMC::PeriodicBC(box);
}

// Independent, deliberately naive g(r): own O(N²) loop with cubic min-image and
// the same normalization compute_gr documents. Validates the production path
// (which reuses accumulate_pair_histogram) end-to-end without sharing code.
struct BruteGr {
  RMC::vec_t total;
  std::vector<RMC::vec_t> partials; // pair (a<=b) flattened in a,b order
  std::vector<std::string> labels;
};

BruteGr brute_gr(const coords_t &c, double L,
                 const std::vector<std::string> &elems, double r_min,
                 double r_max, int n_bins) {
  // distinct sorted species
  std::vector<std::string> species(elems.begin(), elems.end());
  std::sort(species.begin(), species.end());
  species.erase(std::unique(species.begin(), species.end()), species.end());
  const int S = static_cast<int>(species.size());
  auto sid = [&](const std::string &e) {
    return static_cast<int>(
        std::lower_bound(species.begin(), species.end(), e) - species.begin());
  };
  std::vector<int> id(elems.size());
  std::vector<double> count(S, 0.0);
  for (std::size_t i = 0; i < elems.size(); ++i) {
    id[i] = sid(elems[i]);
    count[id[i]] += 1.0;
  }
  const int N = static_cast<int>(c.rows());
  const double V = L * L * L;
  const double bw = (r_max - r_min) / n_bins;

  // pair index for (a<=b)
  auto pidx = [&](int a, int b) {
    if (a > b)
      std::swap(a, b);
    int idx = 0;
    for (int x = 0; x < a; ++x)
      idx += S - x;
    return idx + (b - a);
  };
  const int n_pairs = S * (S + 1) / 2;
  std::vector<RMC::vec_t> hist(n_pairs, RMC::vec_t::Zero(n_bins));

  for (int i = 0; i < N; ++i)
    for (int j = i + 1; j < N; ++j) {
      vec3_t d = c.row(j) - c.row(i);
      for (int k = 0; k < 3; ++k)
        d[k] -= L * std::round(d[k] / L);
      const double r = d.norm();
      const double f = (r - r_min) / bw;
      if (f < 0.0 || f >= n_bins)
        continue;
      hist[pidx(id[i], id[j])](static_cast<int>(f)) += 2.0; // matches 2*w
    }

  BruteGr out;
  out.total = RMC::vec_t::Zero(n_bins);
  const double rho = N / V;
  RMC::vec_t total_hist = RMC::vec_t::Zero(n_bins);
  for (int a = 0; a < S; ++a)
    for (int b = a; b < S; ++b) {
      RMC::vec_t &h = hist[pidx(a, b)];
      total_hist += h;
      RMC::vec_t g = RMC::vec_t::Zero(n_bins);
      const double pf = (a == b) ? 1.0 : 2.0;
      for (int k = 0; k < n_bins; ++k) {
        const double r_lo = r_min + k * bw, r_hi = r_lo + bw;
        const double shell =
            (4.0 * std::numbers::pi / 3.0) * (r_hi * r_hi * r_hi - r_lo * r_lo * r_lo);
        g(k) = h(k) * V / (shell * pf * count[a] * count[b]);
      }
      out.partials.push_back(g);
      out.labels.push_back(species[a] + "-" + species[b]);
    }
  for (int k = 0; k < n_bins; ++k) {
    const double r_lo = r_min + k * bw, r_hi = r_lo + bw;
    const double shell =
        (4.0 * std::numbers::pi / 3.0) * (r_hi * r_hi * r_hi - r_lo * r_lo * r_lo);
    out.total(k) = total_hist(k) / (shell * rho * N);
  }
  return out;
}

} // namespace

TEST_CASE("compute_gr - single A-A peak at the right bin", "[gr][analysis]") {
  const double L = 50.0;
  coords_t c(2, 3);
  c.row(0) << 0.0, 0.0, 0.0;
  c.row(1) << 2.0, 0.0, 0.0; // separation 2.0 Å
  std::vector<std::string> elems{"A", "A"};

  an::GrParams p;
  p.r_min = 0.0;
  p.r_max = 5.0;
  p.n_bins = 50; // bin width 0.1 → distance 2.0 lands in bin 20
  const auto r = an::compute_gr(c, cubic_bc(L), elems, p);
  REQUIRE(r);
  const auto &g = *r;

  REQUIRE(g.species == std::vector<std::string>{"A"});
  REQUIRE(g.pair_labels == std::vector<std::string>{"A-A"});
  REQUIRE(g.counts == std::vector<std::size_t>{2});
  REQUIRE(g.partials.size() == 1);

  // Exactly one populated bin, at index 20, both in total and the partial.
  for (int k = 0; k < p.n_bins; ++k) {
    const double expect_zero = (k == 20) ? 1.0 : 0.0; // 1.0 = "is the peak"
    if (expect_zero == 0.0) {
      REQUIRE_THAT(g.total(k), WithinAbs(0.0, 1e-12));
      REQUIRE_THAT(g.partials[0](k), WithinAbs(0.0, 1e-12));
    } else {
      REQUIRE(g.total(k) > 0.0);
      REQUIRE(g.partials[0](k) > 0.0);
    }
  }
  // Bin center r = 0 + (20+0.5)*0.1 = 2.05.
  REQUIRE_THAT(g.r(20), WithinAbs(2.05, 1e-12));
}

TEST_CASE("compute_gr - matches an independent brute-force g(r)",
          "[gr][analysis]") {
  const double L = 10.0;
  // Deterministic two-species configuration inside the box.
  const double pos[6][3] = {{1.0, 1.0, 1.0}, {2.5, 1.2, 0.8},
                            {4.0, 3.0, 2.0}, {0.5, 4.5, 6.0},
                            {7.0, 7.0, 7.0}, {9.5, 0.2, 3.3}};
  const char *sp[6] = {"Zr", "Cu", "Zr", "Cu", "Zr", "Cu"};
  coords_t c(6, 3);
  std::vector<std::string> elems;
  for (int i = 0; i < 6; ++i) {
    c.row(i) << pos[i][0], pos[i][1], pos[i][2];
    elems.emplace_back(sp[i]);
  }

  an::GrParams p;
  p.r_min = 0.0;
  p.r_max = 5.0;
  p.n_bins = 50;
  const auto r = an::compute_gr(c, cubic_bc(L), elems, p);
  REQUIRE(r);
  const auto &g = *r;

  const BruteGr bf = brute_gr(c, L, elems, p.r_min, p.r_max, p.n_bins);

  // Species are sorted: Cu < Zr; pairs Cu-Cu, Cu-Zr, Zr-Zr.
  REQUIRE(g.species == std::vector<std::string>{"Cu", "Zr"});
  REQUIRE(g.pair_labels == bf.labels);
  REQUIRE(g.partials.size() == bf.partials.size());

  for (int k = 0; k < p.n_bins; ++k) {
    REQUIRE_THAT(g.total(k), WithinAbs(bf.total(k), 1e-9));
    for (std::size_t pr = 0; pr < g.partials.size(); ++pr)
      REQUIRE_THAT(g.partials[pr](k), WithinAbs(bf.partials[pr](k), 1e-9));
  }
}

TEST_CASE("compute_gr - total equals sum of c_a c_b g_ab", "[gr][analysis]") {
  const double L = 12.0;
  const int N = 12;
  coords_t c(N, 3);
  std::vector<std::string> elems;
  for (int i = 0; i < N; ++i) {
    // spread atoms on a coarse integer lattice inside the box
    const int ix = i % 3, iy = (i / 3) % 2, iz = i / 6;
    c.row(i) << ix * 3.0 + 0.5, iy * 4.0 + 0.5, iz * 5.0 + 0.5;
    elems.emplace_back(i % 2 == 0 ? "Cu" : "Zr");
  }

  an::GrParams p;
  p.r_min = 0.0;
  p.r_max = 6.0;
  p.n_bins = 60;
  const auto r = an::compute_gr(c, cubic_bc(L), elems, p);
  REQUIRE(r);
  const auto &g = *r;

  // c_a = N_a / N.
  std::vector<double> conc(g.species.size());
  for (std::size_t s = 0; s < g.species.size(); ++s)
    conc[s] = static_cast<double>(g.counts[s]) / N;

  // Reconstruct total = Σ_{a<=b} (a==b ? 1 : 2) c_a c_b g_ab.
  const int S = static_cast<int>(g.species.size());
  RMC::vec_t recon = RMC::vec_t::Zero(p.n_bins);
  int pr = 0;
  for (int a = 0; a < S; ++a)
    for (int b = a; b < S; ++b, ++pr) {
      const double w = (a == b ? 1.0 : 2.0) * conc[a] * conc[b];
      recon += w * g.partials[pr];
    }
  for (int k = 0; k < p.n_bins; ++k)
    REQUIRE_THAT(g.total(k), WithinAbs(recon(k), 1e-9));
}

TEST_CASE("compute_gr - file overload reads a LAMMPS cell", "[gr][analysis]") {
  // Two Cu atoms 2 Å apart in a 20 Å cubic cell; LAMMPS uses its own box.
  const std::string data = R"(test cell

2 atoms
1 atom types
0.0 20.0 xlo xhi
0.0 20.0 ylo yhi
0.0 20.0 zlo zhi

Atoms

1 1 1.0 1.0 1.0
2 1 3.0 1.0 1.0
)";
  const auto path = std::filesystem::temp_directory_path() / "rmc_gr_in.dat";
  std::ofstream(path) << data;

  an::GrParams p;
  p.r_min = 0.0;
  p.r_max = 5.0;
  p.n_bins = 50;
  // bc is ignored for LAMMPS input (file carries the box); pass infinite.
  const auto r = an::compute_gr(path, RMC::InfiniteBC(1.0), p, {"Cu"});
  std::error_code ec;
  std::filesystem::remove(path, ec);

  REQUIRE(r);
  REQUIRE((*r).species == std::vector<std::string>{"Cu"});
  REQUIRE((*r).pair_labels == std::vector<std::string>{"Cu-Cu"});
  REQUIRE((*r).total(20) > 0.0); // separation 2.0 → bin 20
}

TEST_CASE("compute_gr - non-periodic cell is rejected", "[gr][analysis]") {
  coords_t c(2, 3);
  c.row(0) << 0.0, 0.0, 0.0;
  c.row(1) << 1.0, 0.0, 0.0;
  std::vector<std::string> elems{"A", "A"};
  an::GrParams p;
  const auto r = an::compute_gr(c, RMC::InfiniteBC(1.0), elems, p);
  REQUIRE_FALSE(r);
}

TEST_CASE("compute_gr - bad params and size mismatch error", "[gr][analysis]") {
  coords_t c(2, 3);
  c.row(0) << 0.0, 0.0, 0.0;
  c.row(1) << 1.0, 0.0, 0.0;
  std::vector<std::string> elems{"A", "A"};

  SECTION("n_bins <= 0") {
    an::GrParams p;
    p.n_bins = 0;
    REQUIRE_FALSE(an::compute_gr(c, cubic_bc(20.0), elems, p));
  }
  SECTION("r_max <= r_min") {
    an::GrParams p;
    p.r_min = 5.0;
    p.r_max = 5.0;
    REQUIRE_FALSE(an::compute_gr(c, cubic_bc(20.0), elems, p));
  }
  SECTION("elements/atoms mismatch") {
    an::GrParams p;
    std::vector<std::string> bad{"A"}; // only 1 for 2 atoms
    REQUIRE_FALSE(an::compute_gr(c, cubic_bc(20.0), bad, p));
  }
}
