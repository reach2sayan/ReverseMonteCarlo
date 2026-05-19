// si_ox_nanosphere — fullrmc equivalent
// SiO2 nanoparticle: 20 Si + 40 O (60 atoms) in a 10 Å sphere.
// Constraints: InterMolecular distance, CoordinationConstraint Si: 3-5 O neighbours.
// SmartRandomSelector. Three phases with different step amplitudes.
#include <RMC/Engine.hpp>
#include <RMC/constraints/CoordinationConstraint.hpp>
#include <RMC/constraints/DistanceConstraint.hpp>
#include <RMC/core/BoundaryConditions.hpp>
#include <RMC/core/Group.hpp>
#include <RMC/core/Structure.hpp>
#include <RMC/generators/Translations.hpp>
#include <RMC/selectors/SmartRandomSelector.hpp>

#include <cmath>
#include <iostream>
#include <numbers>

using namespace RMC;

// Place N_si Si atoms and 2*N_si O atoms in a sphere of radius R.
// Si on inner shell, O on outer shell.
static AtomicStructure make_sio2_nanosphere(int n_si = 20, double R = 8.0) {
    AtomicStructure s;
    int n_o = 2 * n_si;
    int N = n_si + n_o;
    s.coordinates.resize(N, 3);
    s.atomic_numbers.resize(N);

    // Fibonacci sphere for Si (inner).
    double golden = std::numbers::phi;
    for (int i = 0; i < n_si; ++i) {
        double theta = std::acos(1.0 - 2.0 * (i + 0.5) / n_si);
        double phi   = 2.0 * std::numbers::pi * i / golden;
        double r_si  = R * 0.5;
        s.coordinates.row(i) << r_si * std::sin(theta) * std::cos(phi),
                                 r_si * std::sin(theta) * std::sin(phi),
                                 r_si * std::cos(theta);
        s.atomic_numbers[i] = 14;
        s.elements.push_back("Si");
        s.names.push_back("Si");
        s.residues.push_back("SIO");
        s.molecule_ids.push_back(static_cast<std::size_t>(i));
    }

    // Fibonacci sphere for O (outer).
    for (int i = 0; i < n_o; ++i) {
        double theta = std::acos(1.0 - 2.0 * (i + 0.5) / n_o);
        double phi   = 2.0 * std::numbers::pi * i / golden;
        double r_o   = R * 0.85;
        int row = n_si + i;
        s.coordinates.row(row) << r_o * std::sin(theta) * std::cos(phi),
                                   r_o * std::sin(theta) * std::sin(phi),
                                   r_o * std::cos(theta);
        s.atomic_numbers[row] = 8;
        s.elements.push_back("O");
        s.names.push_back("O");
        s.residues.push_back("SIO");
        s.molecule_ids.push_back(static_cast<std::size_t>(row));
    }
    return s;
}

static void run_phase(const char *label, double amp, const AtomicStructure &tmpl,
                      int n_si, std::uint64_t n_steps) {
    Engine eng(tmpl, InfiniteBC{});

    InterMolecularDistanceConstraint dc;
    dc.set_minimum_distance("Si", "Si", 1.75);
    dc.set_minimum_distance("O",  "O",  1.10);
    dc.set_minimum_distance("Si", "O",  1.30);
    dc.set_structure(eng.structure().elements, eng.structure().molecule_ids);
    eng.add_constraint(std::move(dc));

    CoordinationConstraint cc;
    for (int i = 0; i < n_si; ++i)
        cc.add_shell(static_cast<std::size_t>(i), "O", 1.3, 2.0, 3, 5);
    cc.set_elements(eng.structure().elements);
    eng.add_constraint(std::move(cc));

    eng.build_atomic_groups(0.0, amp, 42);
    eng.set_selector(IGroupSelector{SmartRandomSelector{1.1, 42}});
    eng.run(n_steps);

    std::cout << label << " (amp=" << amp << "): accepted "
              << eng.stats().steps_accepted
              << "  err " << eng.stats().last_total_err << "\n";
}

int main() {
    constexpr int n_si = 20;
    const auto tmpl = make_sio2_nanosphere(n_si);

    run_phase("Phase 1", 0.10, tmpl, n_si, 5000);
    run_phase("Phase 2", 0.05, tmpl, n_si, 5000);
    run_phase("Phase 3", 0.02, tmpl, n_si, 5000);
}
