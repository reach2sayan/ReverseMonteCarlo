#!/usr/bin/env python3
"""Plot atoms from a PDB file (as written by mcsqs_rmc) as a 3D scatter.

Each ATOM record is parsed for its element/species, sublattice label, and
Cartesian coordinates. Atoms are colored by species (default) or by sublattice.

Usage:
    python plot_pdb.py bestsqs.pdb
    python plot_pdb.py bestsqs.pdb --color-by sublattice
    python plot_pdb.py bestsqs.pdb --save bestsqs.png
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import matplotlib.pyplot as plt


@dataclass
class Atom:
    serial: int
    species: str
    sublattice: str
    x: float
    y: float
    z: float


def parse_pdb(path: str) -> list[Atom]:
    """Parse ATOM/HETATM records. Tolerant of the whitespace-delimited
    layout produced by mcsqs_rmc as well as fixed-column PDB files."""
    atoms: list[Atom] = []
    with open(path) as fh:
        for lineno, line in enumerate(fh, 1):
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            fields = line.split()
            # fields: ATOM serial species sublattice resseq x y z ...
            try:
                serial = int(fields[1])
                species = fields[2]
                sublattice = fields[3]
                x, y, z = (float(v) for v in fields[5:8])
            except (IndexError, ValueError) as exc:
                print(f"warning: skipping line {lineno}: {exc}", file=sys.stderr)
                continue
            atoms.append(Atom(serial, species, sublattice, x, y, z))
    return atoms


def plot(atoms: list[Atom], color_by: str, title: str, save: str | None) -> None:
    keys = sorted({getattr(a, color_by) for a in atoms})
    # High-contrast palette + distinct marker shapes so species stay
    # distinguishable even when depth-overlapping in 3D.
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]
    markers = ["o", "^", "s", "D", "v", "P"]
    colors = {k: palette[i % len(palette)] for i, k in enumerate(keys)}
    shapes = {k: markers[i % len(markers)] for i, k in enumerate(keys)}

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    for k in keys:
        group = [a for a in atoms if getattr(a, color_by) == k]
        ax.scatter(
            [a.x for a in group],
            [a.y for a in group],
            [a.z for a in group],
            color=colors[k],
            marker=shapes[k],
            label=f"{k} ({len(group)})",
            s=110,
            edgecolors="k",
            linewidths=0.6,
            depthshade=False,
        )

    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")
    ax.set_zlabel("z (Å)")
    ax.set_title(title)
    ax.legend(title=color_by, loc="upper left", framealpha=0.9)
    try:
        ax.set_box_aspect((1, 1, 1))
    except (AttributeError, TypeError):
        pass  # older matplotlib
    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=150)
        print(f"saved {save}")
    else:
        plt.show()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pdb", help="path to the PDB file")
    p.add_argument(
        "--color-by",
        choices=["species", "sublattice"],
        default="species",
        help="atom attribute to color by (default: species)",
    )
    p.add_argument("--save", metavar="PNG", help="write to image instead of showing")
    p.add_argument("--title", help="plot title (default: file name)")
    args = p.parse_args(argv)

    atoms = parse_pdb(args.pdb)
    if not atoms:
        print(f"error: no ATOM records found in {args.pdb}", file=sys.stderr)
        return 1
    counts = {}
    for a in atoms:
        counts[a.species] = counts.get(a.species, 0) + 1
    summary = "  ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
    print(f"{len(atoms)} atoms  ({summary})")

    plot(atoms, args.color_by, args.title or args.pdb, args.save)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
