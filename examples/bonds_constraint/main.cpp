// bonds_constraint — fullrmc equivalent
// Three water molecules; BondConstraint enforces O-H bond lengths.
// Phase 1: normal target [0.8,1.1] — bonds already satisfied, engine idles.
// Phase 2: distorted target [2.0,2.5] — engine must elongate bonds.
// Phase 3: restored [0.8,1.1] — engine tightens back.
#include <RMC/Engine.hpp>
#include <RMC/constraints/BondConstraint.hpp>
#include <RMC/core/BoundaryConditions.hpp>
#include <RMC/core/Structure.hpp>

#include <cmath>
#include <iostream>

using namespace RMC;

// 3 water molecules: O(3i) H(3i+1) H(3i+2), bond length ~1 Å.
static AtomicStructure make_3_waters() {
    AtomicStructure s;
    s.coordinates.resize(9, 3);
    s.atomic_numbers.resize(9);
    for (int m = 0; m < 3; ++m) {
        double ox = static_cast<double>(m) * 5.0;
        s.coordinates.row(3 * m)     << ox,       0.0, 0.0;
        s.coordinates.row(3 * m + 1) << ox + 1.0, 0.0, 0.0;
        s.coordinates.row(3 * m + 2) << ox,       1.0, 0.0;
        for (int j = 0; j < 3; ++j) {
            const char *e = (j == 0) ? "O" : "H";
            s.atomic_numbers[3 * m + j] = (j == 0) ? 8 : 1;
            s.elements.push_back(e);
            s.names.push_back(e);
            s.residues.push_back("WAT");
            s.molecule_ids.push_back(static_cast<std::size_t>(m));
        }
    }
    return s;
}

static void add_oh_bonds(BondConstraint &bc, double lo, double hi) {
    for (int m = 0; m < 3; ++m) {
        bc.add_bond(static_cast<std::size_t>(3 * m),
                    static_cast<std::size_t>(3 * m + 1), lo, hi);
        bc.add_bond(static_cast<std::size_t>(3 * m),
                    static_cast<std::size_t>(3 * m + 2), lo, hi);
    }
}

int main() {
    struct Phase { const char *name; double lo, hi; };
    const Phase phases[] = {
        {"normal  [0.8,1.1]", 0.8, 1.1},
        {"tight   [2.0,2.5]", 2.0, 2.5},
        {"restore [0.8,1.1]", 0.8, 1.1},
    };

    for (auto &ph : phases) {
        auto s = make_3_waters();
        Engine eng(std::move(s), InfiniteBC{});

        BondConstraint bc;
        add_oh_bonds(bc, ph.lo, ph.hi);
        eng.add_constraint(std::move(bc));
        eng.build_atomic_groups(0.0, 0.1, 42);
        eng.run(2000);

        std::cout << ph.name << ": accepted " << eng.stats().steps_accepted
                  << "  last_err " << eng.stats().last_total_err << "\n";
    }
}
