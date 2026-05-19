// atomic_niti — fullrmc equivalent
// NiTi-like binary alloy: 16 Ni + 16 Ti on a simple-cubic grid, PeriodicBC.
// Phase 1: pure translation toward synthetic G(r).
// Phase 2: species swap (Ni↔Ti) toward synthetic G(r).
// Phase 3: combined translation + swap.
#include <RMC/Engine.hpp>
#include <RMC/constraints/PairCorrelationConstraint.hpp>
#include <RMC/core/BoundaryConditions.hpp>
#include <RMC/core/Group.hpp>
#include <RMC/core/Structure.hpp>
#include <RMC/generators/SpeciesSwap.hpp>
#include <RMC/generators/Translations.hpp>
#include <RMC/selectors/SmartRandomSelector.hpp>

#include <cmath>
#include <iostream>
#include <numbers>

using namespace RMC;

// 4×4×2 = 32 sites on a simple cubic grid with spacing `a`.
static AtomicStructure make_niti(double a = 2.87) {
    AtomicStructure s;
    constexpr int Nx = 4, Ny = 4, Nz = 2;
    constexpr int N = Nx * Ny * Nz;
    s.coordinates.resize(N, 3);
    s.atomic_numbers.resize(N);

    int idx = 0;
    for (int iz = 0; iz < Nz; ++iz)
      for (int iy = 0; iy < Ny; ++iy)
        for (int ix = 0; ix < Nx; ++ix) {
            s.coordinates.row(idx) << ix * a, iy * a, iz * a;
            // Alternate Ni / Ti checkerboard.
            bool ni = ((ix + iy + iz) % 2 == 0);
            s.atomic_numbers[idx] = ni ? 28 : 22;
            s.elements.push_back(ni ? "Ni" : "Ti");
            s.names.push_back(ni ? "Ni" : "Ti");
            s.residues.push_back("ALL");
            s.molecule_ids.push_back(static_cast<std::size_t>(idx));
            ++idx;
        }
    return s;
}

// Generate a synthetic g(r) with a single peak at r0.
static mat_t synthetic_pcf(double r_min, double r_max, int n_bins,
                            double r0, double sigma) {
    mat_t data(n_bins, 2);
    const double dr = (r_max - r_min) / static_cast<double>(n_bins);
    for (int i = 0; i < n_bins; ++i) {
        double r = r_min + (i + 0.5) * dr;
        double g = 1.0 + 5.0 * std::exp(-0.5 * std::pow((r - r0) / sigma, 2));
        data(i, 0) = r;
        data(i, 1) = g - 1.0; // store as g(r)-1 (PCF)
    }
    return data;
}

static Engine build_engine(const AtomicStructure &tmpl, double a) {
    mat3_t box = mat3_t::Zero();
    box(0,0) = a * 4; box(1,1) = a * 4; box(2,2) = a * 2;
    Engine eng(tmpl, PeriodicBC{box});

    auto pcf = synthetic_pcf(0.5, 6.0, 56, a, 0.2);
    PairCorrelationConstraint cc;
    cc.set_experimental_data(pcf);
    cc.set_elements(eng.structure().elements);
    cc.set_number_density(static_cast<double>(tmpl.size()) / (4*a * 4*a * 2*a));
    cc.initialise();
    eng.add_constraint(std::move(cc));
    return eng;
}

int main() {
    constexpr double a = 2.87;
    const auto tmpl = make_niti(a);

    // Phase 1: translation only.
    {
        auto eng = build_engine(tmpl, a);
        eng.build_atomic_groups(0.0, 0.1, 1);
        eng.set_selector(IGroupSelector{SmartRandomSelector{1.1, 1}});
        eng.run(10000);
        std::cout << "Phase 1 (translate): accepted " << eng.stats().steps_accepted
                  << "  err " << eng.stats().last_total_err << "\n";
    }

    // Phase 2: species swap only.
    {
        auto eng = build_engine(tmpl, a);
        std::vector<std::vector<std::size_t>> one_lattice(1);
        for (std::size_t i = 0; i < tmpl.size(); ++i) one_lattice[0].push_back(i);
        SpeciesSwapGenerator gen{eng.structure(), one_lattice, 2};
        for (std::size_t i = 0; i < eng.structure().size(); ++i) {
            Group g;
            g.name = "s" + std::to_string(i);
            g.indices = {i};
            g.generator = IMoveGenerator{gen};
            eng.add_group(std::move(g));
        }
        eng.set_selector(IGroupSelector{SmartRandomSelector{1.1, 2}});
        eng.run(10000);
        std::cout << "Phase 2 (swap):      accepted " << eng.stats().steps_accepted
                  << "  err " << eng.stats().last_total_err << "\n";
    }

    // Phase 3: translate + swap combined (alternating generators per group).
    {
        auto eng = build_engine(tmpl, a);
        std::vector<std::vector<std::size_t>> one_lattice(1);
        for (std::size_t i = 0; i < tmpl.size(); ++i) one_lattice[0].push_back(i);
        SpeciesSwapGenerator swap_gen{eng.structure(), one_lattice, 3};
        for (std::size_t i = 0; i < eng.structure().size(); ++i) {
            Group g;
            g.name = "t" + std::to_string(i);
            g.indices = {i};
            // Even sites: translate; odd sites: swap.
            if (i % 2 == 0)
                g.generator.emplace(TranslationGenerator(0.0, 0.1, static_cast<std::uint32_t>(i + 10)));
            else
                g.generator = IMoveGenerator{swap_gen};
            eng.add_group(std::move(g));
        }
        eng.set_selector(IGroupSelector{SmartRandomSelector{1.1, 3}});
        eng.run(10000);
        std::cout << "Phase 3 (combined):  accepted " << eng.stats().steps_accepted
                  << "  err " << eng.stats().last_total_err << "\n";
    }
}
