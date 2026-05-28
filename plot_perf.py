#!/usr/bin/env python3

import sys
import pandas as pd
import matplotlib.pyplot as plt

if len(sys.argv) != 3:
    print("usage: plot_perf.py <csv> <outdir>")
    sys.exit(1)

csv_path = sys.argv[1]
outdir = sys.argv[2]

df = pd.read_csv(csv_path)

metrics = [
    ("ipc", "IPC", True),
    ("cache_misses", "Cache Misses", True),
    ("dtlb_rate", "dTLB Miss Rate", True),
    ("backend_stalls", "Backend Stalls", True),
]

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "figure.figsize": (8, 5),
})

for col, title, zero_origin in metrics:

    fig, ax = plt.subplots()

    ax.plot(
        df["grid"],
        df[col],
        marker="o",
        linewidth=2,
        markersize=7,
    )

    ax.set_xscale("log", base=2)

    ax.set_xlabel("Grid Size")
    ax.set_ylabel(title)
    ax.set_title(title)

    ax.grid(True, which="both", alpha=0.3)

    if zero_origin:
        ax.set_ylim(bottom=0)

    ax.set_xlim(left=512)

    fig.tight_layout()

    fig.savefig(
        f"{outdir}/{col}.png",
        dpi=220,
        bbox_inches="tight",
    )

print(df)