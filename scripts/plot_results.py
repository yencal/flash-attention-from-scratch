#!/usr/bin/env python3
import sys

import matplotlib.pyplot as plt
import pandas as pd

if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} <csv1> [csv2] ...")
    sys.exit(1)

df = pd.concat([pd.read_csv(f) for f in sys.argv[1:]], ignore_index=True)

plt.figure(figsize=(10, 6))

labels = df["Label"].unique()
for label in sorted(labels):
    data = df[df["Label"] == label]
    plt.plot(
        data["N"], data["TFLOPS"], marker="o", label=label, linewidth=2, markersize=6
    )

sizes = sorted(df["N"].unique())
plt.xticks(sizes, [str(s) for s in sizes])
plt.xlabel("Sequence Length (N)", fontsize=14)
plt.ylabel("TFLOPS", fontsize=14)
plt.title("Flash Attention Performance (FP16, d=128)", fontsize=16)
plt.legend(fontsize=10, loc="upper left", bbox_to_anchor=(1.02, 1))
plt.grid(True, which="major", alpha=0.3)
plt.tight_layout()

plt.savefig("flash_attention_plot.png", dpi=300, bbox_inches="tight")
print("Plot saved: flash_attention_plot.png")
