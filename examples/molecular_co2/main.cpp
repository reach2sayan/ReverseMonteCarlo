// molecular_co2 — fullrmc equivalent
// 8 CO2 molecules (24 atoms). InfiniteBC.
// Constraints: BondConstraint C-O, AngleConstraint O-C-O, InterMolecular distance.
// Two move layers: per-atom translation + per-molecule translation.
#include <RMC/Engine.hpp>
#include <RMC/constraints/AngleConstraint.hpp>
#include <RMC/constraints/BondConstraint.hpp>
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

// 8 CO2 molecules: O(3m) C(3m+1) O(3m+2); linear along X, spaced 6 Å apart.
static AtomicStructure make_co2(int n_mol = 8) {
    AtomicStructure s;
    const int N = 3 * n_mol;
    s.coordinates.resize(N, 3);
    s.atomic_numbers.resize(N);

    for (int m = 0; m < n_mol; ++m) {
        double cx = static_cast<double>(m) * 6.0;
        // O-C-O along X; C-O bond = 1.16 Å
        s.coordinates.row(3*m)   << cx - 1.16, 0.0, 0.0;
        s.coordinates.row(3*m+1) << cx,         0.0, 0.0;
        s.coordinates.row(3*m+2) << cx + 1.16, 0.0, 0.0;

        s.atomic_numbers[3*m] = s.atomic_numbers[3*m+2] = 8;
        s.atomic_numbers[3*m+1] = 6;
        s.elements.push_back("O");
        s.elements.push_back("C");
        s.elements.push_back("O");
        s.names.push_back("O"); s.names.push_back("C"); s.names.push_back("O");
        for (int j = 0; j < 3; ++j) {
            s.residues.push_back("CO2");
            s.molecule_ids.push_back(static_cast<std::size_t>(m));
        }
    }
    return s;
}

int main() {
    constexpr int n_mol = 8;
    auto s = make_co2(n_mol);
    Engine eng(std::move(s), InfiniteBC{});

    // C-O bond constraint.
    BondConstraint bc;
    for (int m = 0; m < n_mol; ++m) {
        bc.add_bond(static_cast<std::size_t>(3*m),   static_cast<std::size_t>(3*m+1), 0.9, 1.4);
        bc.add_bond(static_cast<std::size_t>(3*m+2), static_cast<std::size_t>(3*m+1), 0.9, 1.4);
    }
    eng.add_constraint(std::move(bc));

    // O-C-O linearity: [170°,180°].
    AngleConstraint ac;
    for (int m = 0; m < n_mol; ++m) {
        ac.add_angle(static_cast<std::size_t>(3*m),
                     static_cast<std::size_t>(3*m+1),
                     static_cast<std::size_t>(3*m+2),
                     170.0 * std::numbers::pi / 180.0,
                     std::numbers::pi);
    }
    eng.add_constraint(std::move(ac));

    // Intermolecular distance (exclude intra).
    InterMolecularDistanceConstraint dc;
    dc.set_minimum_distance("O", "O", 1.4);
    dc.set_minimum_distance("C", "C", 1.4);
    dc.set_minimum_distance("O", "C", 1.4);
    dc.set_structure(eng.structure().elements, eng.structure().molecule_ids);
    eng.add_constraint(std::move(dc));

    // Per-atom groups (fine moves).
    for (std::size_t i = 0; i < eng.structure().size(); ++i) {
        Group g;
        g.name = "atom_" + std::to_string(i);
        g.indices = {i};
        g.generator.emplace(TranslationGenerator(0.0, 0.1, static_cast<std::uint32_t>(i + 1)));
        eng.add_group(std::move(g));
    }

    // Per-molecule groups (coarse moves).
    for (int m = 0; m < n_mol; ++m) {
        Group g;
        g.name = "mol_" + std::to_string(m);
        g.indices = {static_cast<std::size_t>(3*m),
                     static_cast<std::size_t>(3*m+1),
                     static_cast<std::size_t>(3*m+2)};
        g.generator.emplace(TranslationGenerator(0.0, 0.2, static_cast<std::uint32_t>(m + 100)));
        eng.add_group(std::move(g));
    }

    eng.set_selector(IGroupSelector{SmartRandomSelector{1.1, 7}});
    eng.run(5000);

    std::cout << "CO2 (" << n_mol << " mol): accepted " << eng.stats().steps_accepted
              << " / " << eng.stats().steps_tried
              << "  err " << eng.stats().last_total_err << "\n";
}
