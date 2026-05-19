// dihedral_constraint — fullrmc equivalent
// Butane-like 4-carbon backbone; DihedralAngleConstraint steers the C1-C2-C3-C4
// torsion angle into three different rotamer shells across phases.
#include <RMC/Engine.hpp>
#include <RMC/constraints/AngleConstraint.hpp>
#include <RMC/constraints/BondConstraint.hpp>
#include <RMC/constraints/DihedralAngleConstraint.hpp>
#include <RMC/core/BoundaryConditions.hpp>
#include <RMC/core/Structure.hpp>

#include <cmath>
#include <iostream>
#include <numbers>

using namespace RMC;

// 4 C atoms in a non-collinear chain so the dihedral is well-defined.
static AtomicStructure make_butane() {
    AtomicStructure s;
    s.coordinates.resize(4, 3);
    s.coordinates <<
        0.0, 0.0, 0.0,
        1.5, 0.0, 0.0,
        3.0, 0.5, 0.0,
        4.5, 0.5, 0.8;
    s.atomic_numbers.resize(4);
    s.atomic_numbers.setConstant(6);
    for (int i = 0; i < 4; ++i) {
        s.elements.push_back("C");
        s.names.push_back("C");
        s.residues.push_back("BUT");
        s.molecule_ids.push_back(0);
    }
    return s;
}

int main() {
    using std::numbers::pi;
    struct Phase { const char *name; double lo, hi; };
    const Phase phases[] = {
        {"gauche+ [150,160] deg",  150.0 * pi / 180.0, 160.0 * pi / 180.0},
        {"eclipsed [10, 30] deg",   10.0 * pi / 180.0,  30.0 * pi / 180.0},
        {"anti    [90,100] deg",    90.0 * pi / 180.0, 100.0 * pi / 180.0},
    };

    for (auto &ph : phases) {
        auto s = make_butane();
        Engine eng(std::move(s), InfiniteBC{});

        BondConstraint bc;
        bc.add_bond(0, 1, 1.3, 1.7);
        bc.add_bond(1, 2, 1.3, 1.7);
        bc.add_bond(2, 3, 1.3, 1.7);
        eng.add_constraint(std::move(bc));

        AngleConstraint ac;
        ac.add_angle(0, 1, 2, 100.0 * pi / 180.0, 130.0 * pi / 180.0);
        ac.add_angle(1, 2, 3, 100.0 * pi / 180.0, 130.0 * pi / 180.0);
        eng.add_constraint(std::move(ac));

        DihedralAngleConstraint dc;
        dc.add_dihedral(0, 1, 2, 3, ph.lo, ph.hi);
        eng.add_constraint(std::move(dc));

        eng.build_atomic_groups(0.0, 0.15, 42);
        eng.run(10000);

        std::cout << ph.name << ": accepted " << eng.stats().steps_accepted
                  << "  last_err " << eng.stats().last_total_err << "\n";
    }
}
