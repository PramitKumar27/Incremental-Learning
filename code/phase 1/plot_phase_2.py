import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- load CSV ----
sweep = pd.read_csv("phase2_knee_sweep_per_param.csv")
sweep = sweep.sort_values("ber")

ber = sweep["ber"].to_numpy()
mean = sweep["mean_acc"].to_numpy()
std = sweep["std_acc"].to_numpy()

# -------------------------
# Plot 1: Accuracy vs BER
# (mean with clipped error bars)
# -------------------------
# Clip mean±std into [0, 100] so error bars don't extend beyond valid accuracy range
lower = np.clip(mean - std, 0, 100)
upper = np.clip(mean + std, 0, 100)
yerr = np.vstack([mean - lower, upper - mean])  # asymmetric errors

plt.figure(figsize=(8.5, 5.5))
plt.xscale("log")
plt.errorbar(ber, mean, yerr=yerr, fmt="o-", capsize=4, linewidth=2, markersize=6)

plt.ylim(0, 100)
plt.xlabel("BER (per_param, log scale)", fontsize=12)
plt.ylabel("Accuracy (%)", fontsize=12)
plt.title("Accuracy vs BER (mean ± std, clipped to [0,100])", fontsize=14)
plt.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6)

plt.tight_layout()
plt.savefig("accuracy_vs_ber_better.png", dpi=250)
plt.close()

# -------------------------
# Plot 2: Failure probability vs BER
# -------------------------
fail50 = sweep["fail_rate(<50%)"].to_numpy()
fail20 = sweep["cata_rate(<20%)"].to_numpy()

plt.figure(figsize=(8.5, 5.5))
plt.xscale("log")
plt.plot(ber, fail50, "o-", linewidth=2, markersize=6, label="P(acc < 50%)")
plt.plot(ber, fail20, "o-", linewidth=2, markersize=6, label="P(acc < 20%)")

plt.ylim(0, 1.0)
plt.xlabel("BER (per_param, log scale)", fontsize=12)
plt.ylabel("Probability", fontsize=12)
plt.title("Failure probability vs BER (Monte-Carlo)", fontsize=14)
plt.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6)
plt.legend(fontsize=11)

plt.tight_layout()
plt.savefig("failureprob_vs_ber_better.png", dpi=250)
plt.close()

print("Saved: accuracy_vs_ber_better.png, failureprob_vs_ber_better.png")