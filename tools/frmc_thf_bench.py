#!/usr/bin/env python3
"""
fullrmc THF benchmark, pointed at the SAME 260-atom system the C++ RMC
benchmark uses (examples/benchmark/data/{system.pdb,experimental.gr}).

Mirrors fullrmc's own Examples/benchmark/run.py constraint set:
  PDF + InterMolecularDistance + Bond + BondsAngle + ImproperAngle
defined by THF residue name (order-independent), 20 molecules.

Emits time-per-step for:
  A) constraint subsets at a fixed group size
  B) all-constraints, group size 13, varying step counts
so it can be compared 1:1 against the C++ engine.
"""
from __future__ import print_function
import sys, time, json
import numpy as np

# fullrmc 4.1 / pdbparser use the removed np.float/np.int/... aliases.
# Restore them (numpy ABI unchanged; only the Python-level names are gone).
for _a, _t in [("float", float), ("int", int), ("bool", bool),
               ("object", object), ("str", str), ("complex", complex),
               ("long", int), ("unicode", str)]:
    if not hasattr(np, _a):
        setattr(np, _a, _t)

from fullrmc.Globals import LOGGER, maxint
from fullrmc.Engine import Engine
from fullrmc.Constraints.PairDistributionConstraints import PairDistributionConstraint
from fullrmc.Constraints.DistanceConstraints import InterMolecularDistanceConstraint
from fullrmc.Constraints.BondConstraints import BondConstraint
from fullrmc.Constraints.AngleConstraints import BondsAngleConstraint
from fullrmc.Constraints.ImproperAngleConstraints import ImproperAngleConstraint

LOGGER.set_minimum_level(maxint, stdoutFlag=True, fileFlag=True)

PDB = 'system.pdb'
GR  = 'experimental.gr'

ENGINE = Engine(path='thf_engine.rmc', freshStart=True)
ENGINE.set_pdb(PDB)

PDF = PairDistributionConstraint(experimentalData=GR, weighting="atomicNumber")
EMD = InterMolecularDistanceConstraint()
B   = BondConstraint()
BA  = BondsAngleConstraint()
IA  = ImproperAngleConstraint()
ENGINE.add_constraints([PDF, EMD, B, BA, IA])

B.create_bonds_by_definition(bondsDefinition={"THF": [
    ('O','C1',1.22,1.70), ('O','C4',1.22,1.70),
    ('C1','C2',1.25,1.90), ('C2','C3',1.25,1.90), ('C3','C4',1.25,1.90),
    ('C1','H11',0.58,1.22), ('C1','H12',0.58,1.22),
    ('C2','H21',0.58,1.22), ('C2','H22',0.58,1.22),
    ('C3','H31',0.58,1.22), ('C3','H32',0.58,1.22),
    ('C4','H41',0.58,1.22), ('C4','H42',0.58,1.22)]})

BA.create_angles_by_definition(anglesDefinition={"THF": [
    ('O','C1','C4',105,125), ('C1','O','C2',100,120), ('C4','O','C3',100,120),
    ('C2','C1','C3',95,115), ('C3','C2','C4',95,115),
    ('C1','H11','H12',98,118), ('C2','H21','H22',98,118),
    ('C3','H31','H32',98,118), ('C4','H41','H42',98,118),
    ('C1','H11','O',100,120), ('C1','H12','O',100,120),
    ('C4','H41','O',100,120), ('C4','H42','O',100,120),
    ('C1','H11','C2',103,123), ('C1','H12','C2',103,123),
    ('C2','H21','C1',103,123), ('C2','H21','C3',103,123),
    ('C2','H22','C1',103,123), ('C2','H22','C3',103,123),
    ('C3','H31','C2',103,123), ('C3','H31','C4',103,123),
    ('C3','H32','C2',103,123), ('C3','H32','C4',103,123),
    ('C4','H41','C3',103,123), ('C4','H42','C3',103,123)]})

IA.create_angles_by_definition(anglesDefinition={"THF": [
    ('C2','O','C1','C4',-15,15), ('C3','O','C1','C4',-15,15)]})

ENGINE.save()  # fullrmc 4.1 requires a defined repository before run()


def make_groups(GN):
    return [np.array(item, dtype=np.int32)
            for item in zip(*[range(idx, ENGINE.numberOfAtoms, GN)
                              for idx in range(GN)])]


def run(nsteps, groups, pdf, vdw, bond, angle, improper):
    # NB: fullrmc 4.1 forbids resetting the pdb on a normal frame, so runs
    # continue from the prior structure. Timing (our metric) is unaffected;
    # tried/accepted are reported as per-run deltas.
    ENGINE.set_groups(groups)
    PDF.set_used(bool(pdf)); EMD.set_used(bool(vdw)); B.set_used(bool(bond))
    BA.set_used(bool(angle)); IA.set_used(bool(improper))
    ENGINE.initialize_used_constraints()
    tried0, acc0 = int(ENGINE.tried), int(ENGINE.accepted)
    tic = time.time()
    ENGINE.run(numberOfSteps=nsteps, saveFrequency=2*nsteps, restartPdb=None)
    dt = time.time() - tic
    return (dt / nsteps, dt,
            int(ENGINE.tried) - tried0, int(ENGINE.accepted) - acc0)


SUBSETS = [
    ("none",     dict(pdf=0, vdw=0, bond=0, angle=0, improper=0)),
    ("pdf",      dict(pdf=1, vdw=0, bond=0, angle=0, improper=0)),
    ("vdw",      dict(pdf=0, vdw=1, bond=0, angle=0, improper=0)),
    ("bond",     dict(pdf=0, vdw=0, bond=1, angle=0, improper=0)),
    ("angle",    dict(pdf=0, vdw=0, bond=0, angle=1, improper=0)),
    ("improper", dict(pdf=0, vdw=0, bond=0, angle=0, improper=1)),
    ("all",      dict(pdf=1, vdw=1, bond=1, angle=1, improper=1)),
]


def main():
    out = {"natoms": int(ENGINE.numberOfAtoms), "subsets": {}, "nsteps": {}}

    # A) constraint subsets at GN=13 (one molecule)
    gn = 13
    sub_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    groups = make_groups(gn)
    print("== subsets, GN=%d, nsteps=%d ==" % (gn, sub_steps), flush=True)
    for name, fl in SUBSETS:
        tps, dt, tried, acc = run(sub_steps, groups, **fl)
        out["subsets"][name] = dict(time_per_step=tps, total_s=dt,
                                    tried=tried, accepted=acc, nsteps=sub_steps)
        print("  %-9s  %.3e s/step  (%.2fs total, tried=%d acc=%d)"
              % (name, tps, dt, tried, acc), flush=True)

    # B) all-constraints, GN=13, step scaling
    steps_list = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [500, 1000, 2000]
    print("== all-constraints scaling, GN=13 ==", flush=True)
    for ns in steps_list:
        tps, dt, tried, acc = run(ns, groups, pdf=1, vdw=1, bond=1, angle=1, improper=1)
        out["nsteps"][str(ns)] = dict(time_per_step=tps, total_s=dt,
                                      tried=tried, accepted=acc)
        print("  %6d steps  %.3e s/step  (%.2fs total, acc=%d)"
              % (ns, tps, dt, acc), flush=True)

    with open("frmc_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote frmc_results.json", flush=True)


if __name__ == "__main__":
    main()
