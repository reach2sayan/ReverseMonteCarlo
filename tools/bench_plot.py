#!/usr/bin/env python3
"""
Run RMC_tests benchmarks and render a colour-coded bar-chart figure.

Usage (from the repo root or build dir):
    python3 tools/bench_plot.py                     # 30 samples, show window
    python3 tools/bench_plot.py --samples 10        # fewer samples
    python3 tools/bench_plot.py --out bench.png     # save instead of show
    python3 tools/bench_plot.py --binary ./build/RMC_tests
"""

import argparse
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# ── helpers ──────────────────────────────────────────────────────────────────

def human_time(ns: float) -> tuple[float, str]:
    """Return (value, unit-label) scaled to the most readable unit."""
    for scale, unit in [(1e9, "s"), (1e6, "ms"), (1e3, "µs")]:
        if ns >= scale:
            return ns / scale, unit
    return ns, "ns"


def pick_colors(means_ns: list[float]) -> list[str]:
    """Green→yellow→red gradient based on relative magnitude."""
    lo, hi = min(means_ns), max(means_ns)
    cmap = plt.colormaps["RdYlGn_r"]
    if lo == hi:
        return [cmap(0.2)] * len(means_ns)
    return [cmap(0.1 + 0.8 * (v - lo) / (hi - lo)) for v in means_ns]


def parse_xml(xml_text: str) -> list[dict]:
    """
    Returns a list of benchmark groups:
        [{"title": str, "bars": [{"name", "mean_ns", "lo_ns", "hi_ns"}, ...]}, ...]
    """
    root = ET.fromstring(xml_text)
    groups = []
    for tc in root.iter("TestCase"):
        bars = []
        for br in tc.iter("BenchmarkResults"):
            mean_el = br.find("mean")
            if mean_el is None:
                continue
            mean = float(mean_el.attrib["value"])
            lo   = float(mean_el.attrib["lowerBound"])
            hi   = float(mean_el.attrib["upperBound"])
            bars.append({"name": br.attrib["name"].strip(),
                         "mean_ns": mean, "lo_ns": lo, "hi_ns": hi})
        if bars:
            groups.append({"title": tc.attrib["name"], "bars": bars})
    return groups


# ── plotting ─────────────────────────────────────────────────────────────────

def plot_group(ax: plt.Axes, group: dict, use_log: bool) -> None:
    bars  = group["bars"]
    names = [b["name"] for b in bars]
    means = [b["mean_ns"] for b in bars]
    errs_lo = [b["mean_ns"] - b["lo_ns"] for b in bars]
    errs_hi = [b["hi_ns"] - b["mean_ns"] for b in bars]

    colors = pick_colors(means)
    y = np.arange(len(names))

    ax.barh(y, means, xerr=[errs_lo, errs_hi],
            color=colors, edgecolor="white", linewidth=0.6,
            error_kw={"ecolor": "#555", "capsize": 3, "elinewidth": 1},
            height=0.55)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_title(group["title"], fontsize=9, fontweight="bold", pad=6)

    if use_log:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda v, _: human_time(v)[0].__format__(".3g")
                                 + " " + human_time(v)[1]))
    else:
        # Choose a common unit for all bars in this panel
        ref_val, ref_unit = human_time(max(means))
        scale = max(means) / ref_val
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda v, _: f"{v/scale:.3g}"))
        ax.set_xlabel(ref_unit, fontsize=8)

    ax.tick_params(axis="x", labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.6)

    # Annotate each bar with its mean ± CI
    for i, b in enumerate(bars):
        val, unit = human_time(b["mean_ns"])
        lo_v = (b["mean_ns"] - b["lo_ns"])
        hi_v = (b["hi_ns"] - b["mean_ns"])
        lo_s, _ = human_time(lo_v)
        hi_s, _ = human_time(hi_v)
        label = f"{val:.3g} {unit}  [−{lo_s:.2g}/+{hi_s:.2g}]"
        ax.text(b["mean_ns"] * (1.05 if use_log else 1.0),
                i, label, va="center", fontsize=6.5, color="#333")


def make_figure(groups: list[dict], out_path: str | None) -> None:
    # Decide which panels need log scale (span > 100×)
    use_log = []
    for g in groups:
        means = [b["mean_ns"] for b in g["bars"]]
        use_log.append(max(means) / min(means) > 100)

    n = len(groups)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, max(3.5, 0.55 * max(len(g["bars"]) for g in groups) + 2)))
    if n == 1:
        axes = [axes]

    fig.patch.set_facecolor("#f8f8f8")
    for ax, group, log in zip(axes, groups, use_log):
        ax.set_facecolor("#f8f8f8")
        plot_group(ax, group, log)

    fig.suptitle("RMC benchmark results  (500 steps · 95 % CI error bars)",
                 fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {out_path}")
    else:
        plt.show()


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Run & plot RMC benchmarks.")
    ap.add_argument("--binary", default=None,
                    help="Path to RMC_tests binary (auto-detected if omitted)")
    ap.add_argument("--samples", type=int, default=30,
                    help="Catch2 --benchmark-samples (default 30)")
    ap.add_argument("--out", default=None,
                    help="Save figure to this path instead of opening a window")
    args = ap.parse_args()

    # locate binary
    if args.binary:
        binary = Path(args.binary)
    else:
        candidates = [
            Path(__file__).parent.parent / "build" / "RMC_tests",
            Path("RMC_tests"),
            Path("build/RMC_tests"),
        ]
        binary = next((p for p in candidates if p.exists()), None)
        if binary is None:
            sys.exit("Cannot find RMC_tests. Build the project first or pass --binary.")

    print(f"Running {binary}  (samples={args.samples}) …")
    result = subprocess.run(
        [str(binary), "[!benchmark]",
         "--reporter", "XML",
         "--benchmark-samples", str(args.samples)],
        capture_output=True, text=True
    )
    if result.returncode != 0 and not result.stdout.strip().startswith("<?xml"):
        sys.exit(f"RMC_tests failed:\n{result.stderr}")

    groups = parse_xml(result.stdout)
    if not groups:
        sys.exit("No benchmark results found in XML output.")

    print(f"Parsed {sum(len(g['bars']) for g in groups)} benchmarks "
          f"across {len(groups)} groups.")
    make_figure(groups, args.out)


if __name__ == "__main__":
    main()
