#!/usr/bin/env python3
"""
Run RMC_tests benchmarks and render a colour-coded bar-chart figure.

Usage (from the repo root or build dir):
    python3 tools/bench_plot.py                           # all [!benchmark], 30 samples
    python3 tools/bench_plot.py --tags "[!benchmark][tbb]"   # TBB / incremental only
    python3 tools/bench_plot.py --samples 10 --cols 2    # fewer samples, 2-column layout
    python3 tools/bench_plot.py --out bench.png           # save instead of showing window
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


# ── speedup annotation ────────────────────────────────────────────────────────

def detect_speedup_pairs(bars: list[dict]) -> dict[int, float]:
    """
    For the incremental-vs-full benchmark, bars alternate full/incr at each N.
    Returns {incr_bar_index: speedup_ratio} for every such pair found.
    """
    pairs = {}
    i = 0
    while i < len(bars) - 1:
        a, b = bars[i], bars[i + 1]
        a_is_full = a["name"].startswith("full")
        b_is_incr = b["name"].startswith("incr")
        if a_is_full and b_is_incr and b["mean_ns"] > 0:
            pairs[i + 1] = a["mean_ns"] / b["mean_ns"]
            i += 2
        else:
            i += 1
    return pairs


# ── N-scaling reference line ──────────────────────────────────────────────────

def extract_n_series(bars: list[dict]) -> tuple[list[int], list[float]] | None:
    """
    If every bar name is of the form 'N=<int>', return ([N, ...], [mean_ns, ...]).
    """
    ns, means = [], []
    for b in bars:
        name = b["name"].strip()
        if name.startswith("N="):
            try:
                ns.append(int(name[2:]))
                means.append(b["mean_ns"])
            except ValueError:
                return None
        else:
            return None
    return (ns, means) if len(ns) >= 3 else None


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
        ref_val, ref_unit = human_time(max(means))
        scale = max(means) / ref_val
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda v, _: f"{v/scale:.3g}"))
        ax.set_xlabel(ref_unit, fontsize=8)

    ax.tick_params(axis="x", labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.6)

    # Mean ± CI labels
    for i, b in enumerate(bars):
        val, unit = human_time(b["mean_ns"])
        lo_v = b["mean_ns"] - b["lo_ns"]
        hi_v = b["hi_ns"] - b["mean_ns"]
        lo_s, _ = human_time(lo_v)
        hi_s, _ = human_time(hi_v)
        label = f"{val:.3g} {unit}  [−{lo_s:.2g}/+{hi_s:.2g}]"
        ax.text(b["mean_ns"] * (1.05 if use_log else 1.0),
                i, label, va="center", fontsize=6.5, color="#333")

    # Speedup annotations for full/incr pairs
    speedup_pairs = detect_speedup_pairs(bars)
    for incr_idx, ratio in speedup_pairs.items():
        ax.annotate(f"  ×{ratio:.0f} faster",
                    xy=(bars[incr_idx]["mean_ns"], incr_idx),
                    xytext=(bars[incr_idx - 1]["mean_ns"] * 0.5, incr_idx),
                    fontsize=7, color="#1a7abf", fontweight="bold",
                    va="center",
                    arrowprops=dict(arrowstyle="-", color="#1a7abf",
                                   lw=0.8, linestyle="dashed"))


def plot_n_scaling_inset(ax: plt.Axes, group: dict) -> None:
    """
    Overlay a log-log N vs time scatter + O(N²) reference line on a twin axis,
    when the group contains only 'N=<int>' bars.
    """
    result = extract_n_series(group["bars"])
    if result is None:
        return
    ns, means = result
    ns_arr = np.array(ns, dtype=float)
    means_arr = np.array(means)

    ax2 = ax.inset_axes([0.55, 0.05, 0.42, 0.38])
    ax2.scatter(ns_arr, means_arr, s=30, color="#c0392b", zorder=5)
    ax2.plot(ns_arr, means_arr, color="#c0392b", linewidth=1)

    # O(N²) reference anchored at first point
    ref = means_arr[0] * (ns_arr / ns_arr[0]) ** 2
    ax2.plot(ns_arr, ref, "--", color="#888", linewidth=1, label="O(N²)")
    ax2.legend(fontsize=6, loc="upper left")

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("N", fontsize=6)
    ax2.set_ylabel("time", fontsize=6)
    ax2.tick_params(labelsize=5)
    ax2.set_title("scaling", fontsize=6)
    ax2.spines[["top", "right"]].set_visible(False)


def make_figure(groups: list[dict], out_path: str | None, n_cols: int) -> None:
    use_log = []
    has_n_scaling = []
    for g in groups:
        means = [b["mean_ns"] for b in g["bars"]]
        use_log.append(max(means) / max(min(means), 1e-9) > 100)
        has_n_scaling.append(extract_n_series(g["bars"]) is not None)

    n = len(groups)
    n_cols = min(n_cols, n)
    n_rows = (n + n_cols - 1) // n_cols

    max_bars = max(len(g["bars"]) for g in groups)
    panel_h = max(3.5, 0.55 * max_bars + 2.0)
    fig, axes_grid = plt.subplots(
        n_rows, n_cols,
        figsize=(5.5 * n_cols, panel_h * n_rows),
        squeeze=False,
    )

    fig.patch.set_facecolor("#f8f8f8")
    axes_flat = axes_grid.flatten()

    for idx, (group, log, scaling) in enumerate(zip(groups, use_log, has_n_scaling)):
        ax = axes_flat[idx]
        ax.set_facecolor("#f8f8f8")
        plot_group(ax, group, log)
        if scaling:
            plot_n_scaling_inset(ax, group)

    # Hide unused panels
    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("RMC benchmark results  (95 % CI error bars)",
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
    ap.add_argument("--tags", default="[!benchmark]",
                    help='Catch2 tag filter (default "[!benchmark]"). '
                         'Use "[!benchmark][tbb]" for TBB/incremental benchmarks only.')
    ap.add_argument("--cols", type=int, default=3,
                    help="Number of panel columns in the figure (default 3)")
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

    print(f"Running {binary}  tags={args.tags!r}  samples={args.samples} …")
    result = subprocess.run(
        [str(binary), args.tags,
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
    make_figure(groups, args.out, args.cols)


if __name__ == "__main__":
    main()
