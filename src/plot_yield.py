"""Yield matrix for the report: requirement vs. process chain."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tolerance_mc import REQUIREMENTS

res = Path(__file__).resolve().parents[1] / "results"
rows = [r for r in csv.DictReader(open(res / "tolerance_yield.csv", encoding="utf8"))
        if r["distribution"] == "normal"]
cols = list(REQUIREMENTS) + ["all"]
data = np.array([[float(r[c]) * 100 for c in cols] for r in rows])

fig, ax = plt.subplots(figsize=(9.2, 3.5))
ax.imshow(data, cmap="Blues", vmin=0, vmax=130, aspect="auto")
for (i, j), v in np.ndenumerate(data):
    text = f"{v:.0f}" if v >= 1 or v == 0 else f"{v:.1f}"
    ax.text(j, i, text, ha="center", va="center", fontsize=9, color="white" if v > 75 else "#1b2a3a",
            fontweight="bold" if j == len(cols) - 1 else "normal")
ax.set_xticks(range(len(cols)), [c.replace(" ", "\n", 1) for c in cols[:-1]] + ["all\ntogether"], fontsize=8.5)
ax.set_yticks(range(len(rows)), [r["scenario"] for r in rows], fontsize=9)
ax.axvline(len(cols) - 1.5, color="white", lw=3)
ax.tick_params(length=0, top=True, bottom=False, labeltop=True, labelbottom=False)
for s in ax.spines.values():
    s.set_visible(False)
fig.tight_layout()
fig.savefig(res / "yield_matrix.pdf")
fig.savefig(res / "yield_matrix.png", dpi=180)
