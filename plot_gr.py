#!/usr/bin/env python3
"""Plot a radial distribution function g(r) written by ``RMC_run --gr``.

Reads the whitespace table produced by ``write_gr`` (total g(r) plus one
partial per element pair) and renders a publication-quality figure. The output
format follows the ``--out`` extension; SVG (vector) is the default and is ideal
for publication.

Usage:
    python plot_gr.py gr.dat                  # interactive window
    python plot_gr.py gr.dat --out gr.svg     # vector for publication
    python plot_gr.py gr.dat --out gr.pdf --rmax 8 --no-total
"""
from __future__ import annotations

import argparse
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator

# High-contrast palette, shared with plot_pdb.py.
PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]


def parse_header(path: str) -> tuple[list[str], str | None]:
    """Return (series_labels, annotation).

    ``series_labels`` are the column names after ``r`` (the leading ``g_`` is
    stripped), parsed from the second comment line so the script adapts to
    whatever element pairs are present. ``annotation`` is a short density/species
    summary parsed from the first comment line (None if absent).
    """
    labels: list[str] = []
    annotation: str | None = None
    with open(path) as fh:
        comments = [ln for ln in fh if ln.lstrip().startswith("#")]

    for ln in comments:
        body = ln.lstrip("#").strip()
        if body.startswith("r ") or body.startswith("r\t"):
            # "r  g_total  g_Cu-Cu  g_Cu-Zr ..." -> drop the r column.
            cols = body.split()[1:]
            labels = [c[2:] if c.startswith("g_") else c for c in cols]
        elif "rho" in body:
            rho = re.search(r"rho\)?=\s*([0-9.eE+-]+)", body)
            species = re.search(r"species=\s*(.+)$", body)
            parts = []
            if rho:
                parts.append(rf"$\rho$ = {float(rho.group(1)):.4f} $\AA^{{-3}}$")
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
    rmax: float | None,
    show_total: bool,
) -> None:
    r = data[:, 0]
    series = data[:, 1:]
    if series.shape[1] != len(labels):
        raise ValueError(
            f"header lists {len(labels)} series but data has {series.shape[1]} "
            f"value columns"
        )

    apply_pub_style()
    fig, ax = plt.subplots(figsize=(6.5, 4.0), constrained_layout=True)

    # Ideal-gas baseline.
    ax.axhline(1.0, color="0.6", lw=0.8, ls="--", zorder=0)

    color_idx = 0
    for j, lbl in enumerate(labels):
        if lbl == "total":
            if not show_total:
                continue
            ax.plot(r, series[:, j], color="black", lw=2.0, label="total", zorder=3)
        else:
            ax.plot(
                r,
                series[:, j],
                color=PALETTE[color_idx % len(PALETTE)],
                lw=1.4,
                label=lbl,
                zorder=2,
            )
            color_idx += 1

    ax.set_xlabel(r"$r$ ($\AA$)")
    ax.set_ylabel(r"$g(r)$")
    ax.set_xlim(r.min(), rmax if rmax is not None else r.max())
    ax.set_ylim(bottom=0.0)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_title(title)
    ax.legend(loc="best")

    if annotation:
        ax.text(
            0.97,
            0.05,
            annotation,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
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
    p.add_argument("gr", help="path to the g(r) data file (e.g. gr.dat)")
    p.add_argument(
        "--out",
        metavar="FILE",
        help="write to FILE (.svg/.pdf/.png by extension) instead of showing; "
        "SVG is the default format",
    )
    p.add_argument("--dpi", type=int, default=300, help="DPI for raster output")
    p.add_argument("--rmax", type=float, help="clip the x-axis at this r (Å)")
    p.add_argument("--no-total", action="store_true", help="omit the total g(r)")
    p.add_argument("--title", help="plot title (default: file name)")
    args = p.parse_args(argv)

    try:
        labels, annotation = parse_header(args.gr)
        data = np.loadtxt(args.gr, comments="#")
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if data.ndim != 2 or data.shape[1] < 2:
        print(f"error: {args.gr} has no g(r) columns", file=sys.stderr)
        return 1
    if not labels:
        # No parsable header: synthesise labels (total, then partials).
        labels = ["total"] + [f"partial_{i}" for i in range(1, data.shape[1] - 1)]

    plot(
        data,
        labels,
        annotation,
        args.title or args.gr,
        args.out,
        args.dpi,
        args.rmax,
        show_total=not args.no_total,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
