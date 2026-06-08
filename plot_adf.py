#!/usr/bin/env python3
"""Plot an angular distribution function (ADF) written by ``RMC_run --adf-compute``.

Reads the whitespace table produced by ``write_adf`` (angle in radians followed by
one partial per element triplet — central atom first, e.g. ``adf_Cu-Zr-Zr`` is the
Zr–Cu–Zr angle centred on Cu) and renders a publication-quality figure. The total
is reconstructed as the sum of the partials. The output format follows the
``--out`` extension; SVG (vector) is the default.

Usage:
    python plot_adf.py adf.dat                   # interactive window
    python plot_adf.py adf.dat --out adf.svg     # vector for publication
    python plot_adf.py adf.dat --out adf.pdf --no-total
"""
from __future__ import annotations

import argparse
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

# High-contrast palette, shared with plot_gr.py.
PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b",
           "#17becf", "#bcbd22", "#e377c2", "#7f7f7f"]


def parse_header(path: str) -> tuple[list[str], str | None]:
    """Return (triplet_labels, annotation).

    ``triplet_labels`` are the column names after ``theta`` (the leading ``adf_``
    is stripped), parsed from the second comment line. ``annotation`` is a short
    cutoff/species summary parsed from the first comment line (None if absent).
    """
    labels: list[str] = []
    annotation: str | None = None
    with open(path) as fh:
        comments = [ln for ln in fh if ln.lstrip().startswith("#")]

    for ln in comments:
        body = ln.lstrip("#").strip()
        if body.startswith("theta"):
            cols = body.split()[1:]
            labels = [c[4:] if c.startswith("adf_") else c for c in cols]
        elif "max_dis" in body:
            cutoff = re.search(r"max_dis\s*=\s*([0-9.eE+-]+)", body)
            species = re.search(r"species=\s*(.+)$", body)
            parts = []
            if cutoff:
                parts.append(rf"$r_c$ = {float(cutoff.group(1)):.2f} $\AA$")
            if species:
                parts.append(species.group(1).strip())
            annotation = "\n".join(parts) if parts else None
    return labels, annotation


def apply_pub_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "dejavuserif",
            "font.size": 11,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "savefig.bbox": "tight",
        }
    )


def plot(
    data: np.ndarray,
    labels: list[str],
    annotation: str | None,
    title: str,
    out: str | None,
    dpi: int,
    show_total: bool,
) -> None:
    # Column 0 is the angle in radians; convert to degrees for the axis.
    theta_deg = np.degrees(data[:, 0])
    series = data[:, 1:]
    if series.shape[1] != len(labels):
        raise ValueError(
            f"header lists {len(labels)} series but data has {series.shape[1]} "
            f"value columns"
        )

    apply_pub_style()
    fig, ax = plt.subplots(figsize=(6.5, 4.0), constrained_layout=True)

    for j, lbl in enumerate(labels):
        ax.plot(
            theta_deg,
            series[:, j],
            color=PALETTE[j % len(PALETTE)],
            lw=1.4,
            label=lbl,
            zorder=2,
        )

    if show_total and series.shape[1] > 1:
        ax.plot(theta_deg, series.sum(axis=1), color="black", lw=2.0,
                label="total", zorder=3)

    ax.set_xlabel(r"$\theta$ (degrees)")
    ax.set_ylabel(r"ADF $g(\theta)$")
    ax.set_xlim(0.0, 180.0)
    ax.set_ylim(bottom=0.0)
    ax.xaxis.set_major_locator(MultipleLocator(30))
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_title(title)
    ax.legend(loc="best", ncol=2, fontsize=9)

    if annotation:
        ax.text(
            0.03,
            0.95,
            annotation,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9),
        )

    if out:
        fig.savefig(out, dpi=dpi)
        print(f"saved {out}")
    else:
        plt.show()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("adf", help="path to the ADF data file (e.g. adf.dat)")
    p.add_argument(
        "--out",
        metavar="FILE",
        help="write to FILE (.svg/.pdf/.png by extension) instead of showing; "
        "SVG is the default format",
    )
    p.add_argument("--dpi", type=int, default=300, help="DPI for raster output")
    p.add_argument("--no-total", action="store_true", help="omit the total ADF")
    p.add_argument("--title", help="plot title (default: file name)")
    args = p.parse_args(argv)

    try:
        labels, annotation = parse_header(args.adf)
        data = np.loadtxt(args.adf, comments="#")
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if data.ndim != 2 or data.shape[1] < 2:
        print(f"error: {args.adf} has no ADF columns", file=sys.stderr)
        return 1
    if not labels:
        labels = [f"triplet_{i}" for i in range(1, data.shape[1])]

    plot(
        data,
        labels,
        annotation,
        args.title or args.adf,
        args.out,
        args.dpi,
        show_total=not args.no_total,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
