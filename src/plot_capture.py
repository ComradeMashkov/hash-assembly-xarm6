"""Capture curve and clearance history for the report."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

res = Path(__file__).resolve().parents[1] / "results"
rows = list(csv.DictReader(open(res / "capture.csv", encoding="utf8")))
edges = np.arange(0, 17, 2)


def seated(r):
    return (abs(float(r["dx"])) < .1 and abs(float(r["dy"])) < .1
            and abs(float(r["dz"])) < .3 and abs(float(r["yaw_after"])) < 1)


fig, ax = plt.subplots(figsize=(4.6, 2.7))
styles = {"p1a": ("#1f6fb5", "o", "Pos 1 on plate"), "p2a": ("#c96a1b", "s", "Pos 2 on Pos 1")}
for name, (colour, marker, label) in styles.items():
    part = [r for r in rows if r["part"] == name]
    y = []
    for lo, hi in zip(edges, edges[1:]):
        b = [seated(r) for r in part if lo <= float(r["r_mm"]) < hi]
        y.append(100 * np.mean(b))
    ax.plot(edges[:-1] + 1, y, color=colour, marker=marker, lw=2, ms=6, label=label)
ax.set(xlabel="placement error of the robot [mm]", ylabel="correctly seated [%]", ylim=(-3, 104))
ax.axvline(11, color="#555", ls="--", lw=1)
ax.text(11.2, 60, "designed limit\n4 + 7 mm", fontsize=8, color="#333")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#ddd", lw=.6)
ax.legend(frameon=False, fontsize=8, loc="lower left")
fig.tight_layout()
fig.savefig(res / "capture.pdf")
fig.savefig(res / "capture.png", dpi=170)

fig, ax = plt.subplots(figsize=(4.6, 2.7))
for mode, colour, label in (("lock", "#8a8f96", "zone lock"), ("cbf", "#1f6fb5", "barrier filter")):
    log = np.load(res / f"clearance_{mode}.npy")
    ax.plot(log[:, 0], log[:, 1] * 1e3, color=colour, lw=2, label=f"{label}: {log[-1, 0]:.1f} s")
ax.axhline(30, color="#b3261e", ls="--", lw=1)
ax.text(.2, 42, "safety margin 30 mm", fontsize=8, color="#7a1a14")
ax.set(xlabel="time [s]", ylabel="minimum arm clearance [mm]", ylim=(0, None))
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#ddd", lw=.6)
ax.legend(frameon=False, fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig(res / "clearance.pdf")
fig.savefig(res / "clearance.png", dpi=170)
