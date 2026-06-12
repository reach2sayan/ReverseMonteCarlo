#!/usr/bin/env python3
"""Combine fullrmc (frmc_results.json) and C++ (Catch2 XML) THF benchmark
results into a side-by-side table + bar chart. Same 260-atom THF system,
GN=13, all measured as time-per-step.

Run from a directory containing the three input artifacts:
    frmc_results.json   (from tools/frmc_thf_bench.py)
    cpp_subsets.xml     (RMC_tests "bench: THF constraint subsets*" --reporter XML)
    cpp_steps.xml       (RMC_tests "bench: THF step-count scaling*"  --reporter XML)
Writes fullrmc_comparison.png. See BENCHMARKS.md for the full recipe."""
import json, xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- fullrmc ----
frmc = json.load(open("frmc_results.json"))
f_sub = {k: v["time_per_step"]*1e6 for k, v in frmc["subsets"].items()}   # us/step
f_all_scaling = np.mean([v["time_per_step"]*1e6
                         for v in frmc["nsteps"].values()])

# ---- C++ : subset means (500 steps/run) ----
def cpp_subsets(path):
    out = {}
    for br in ET.parse(path).getroot().iter("BenchmarkResults"):
        n = br.attrib["name"]
        if n.startswith("GN=13"):
            key = n.split()[1]
            out[key] = float(br.find("mean").attrib["value"]) / 500 / 1e3  # us/step
    return out
c_sub = cpp_subsets("cpp_subsets.xml")

# ---- C++ : all-constraints from dedicated step-count scaling ----
c_all = []
for br in ET.parse("cpp_steps.xml").getroot().iter("BenchmarkResults"):
    n = int(br.attrib["name"].split()[0])
    c_all.append(float(br.find("mean").attrib["value"]) / n / 1e3)  # us/step
c_all_scaling = np.mean(c_all)

order = ["none", "pdf", "vdw", "bond", "angle", "improper", "all"]
labels = {"none":"none", "pdf":"PDF g(r)", "vdw":"inter-mol dist",
          "bond":"bond", "angle":"angle", "improper":"improper", "all":"ALL"}

print("THF 260-atom system, group size 13 (one molecule), time per step")
print("=" * 70)
print(f"{'constraint':16s}{'fullrmc 4.1':>14s}{'C++ RMC':>12s}{'speedup':>10s}")
print(f"{'':16s}{'(us/step)':>14s}{'(us/step)':>12s}")
print("-" * 70)
rows = []
for k in order:
    fv = f_sub[k]
    cv = c_all_scaling if k == "all" else c_sub[k]
    sp = fv / cv
    rows.append((labels[k], fv, cv, sp))
    print(f"{labels[k]:16s}{fv:14.2f}{cv:12.3f}{sp:9.1f}x")
print("-" * 70)
print(f"all-constraints (scaling runs):  fullrmc {f_all_scaling:.1f}  "
      f"C++ {c_all_scaling:.2f} us/step  ->  {f_all_scaling/c_all_scaling:.1f}x")

# ---- chart ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
fig.patch.set_facecolor("white")

names = [r[0] for r in rows]
fvals = [r[1] for r in rows]
cvals = [r[2] for r in rows]
y = np.arange(len(names))
h = 0.38
ax1.barh(y+h/2, fvals, h, color="#c0392b", label="fullrmc 4.1 (Python/Cython)")
ax1.barh(y-h/2, cvals, h, color="#2980b9", label="C++ RMC (this repo)")
ax1.set_yticks(y); ax1.set_yticklabels(names)
ax1.invert_yaxis(); ax1.set_xscale("log")
ax1.set_xlabel("time per step (µs, log scale)")
ax1.set_title("THF (260 atoms), GN=13 — time per step", fontweight="bold")
ax1.legend(frameon=False, loc="lower right")
for yi, (fv, cv) in enumerate(zip(fvals, cvals)):
    ax1.text(fv*1.1, yi+h/2, f"{fv:.0f}", va="center", fontsize=7, color="#c0392b")
    ax1.text(cv*1.1, yi-h/2, f"{cv:.2f}", va="center", fontsize=7, color="#2980b9")
ax1.grid(axis="x", ls=":", alpha=0.5)
ax1.spines[["top","right"]].set_visible(False)

sp = [r[3] for r in rows]
colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(sp)))
ax2.barh(y, sp, color=colors)
ax2.set_yticks(y); ax2.set_yticklabels(names); ax2.invert_yaxis()
ax2.set_xlabel("speedup  (fullrmc / C++)")
ax2.set_title("C++ speedup over fullrmc", fontweight="bold")
for yi, s in enumerate(sp):
    ax2.text(s*1.01, yi, f"{s:.0f}×", va="center", fontsize=8, fontweight="bold")
ax2.grid(axis="x", ls=":", alpha=0.5)
ax2.spines[["top","right"]].set_visible(False)

fig.suptitle("RMC engine benchmark — this C++ implementation vs fullrmc 4.1.0\n"
             "identical 260-atom THF system, same constraint set, single core",
             fontweight="bold")
plt.tight_layout(rect=[0,0,1,0.94])
fig.savefig("fullrmc_comparison.png", dpi=150, bbox_inches="tight")
print("\nwrote fullrmc_comparison.png")
