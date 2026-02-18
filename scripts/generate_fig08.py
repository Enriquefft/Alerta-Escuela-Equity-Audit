"""Generate calibration-by-decile bar chart (fig08) for paper."""
import json
import matplotlib.pyplot as plt
import numpy as np

with open("data/exports/predictive_validity.json") as f:
    pv = json.load(f)

deciles = pv["calibration_by_decile"]["lightgbm_calibrated"]["deciles"]
x = np.arange(1, 11)
pred = [d["mean_predicted"] for d in deciles]
obs = [d["mean_observed"] for d in deciles]
baseline = pv["brier_skill_scores"]["lightgbm_calibrated"]["prevalence"]

fig, ax = plt.subplots(figsize=(8, 5))
width = 0.35
ax.bar(x - width / 2, pred, width, label="Predicted Probability", color="#4C72B0")
ax.bar(x + width / 2, obs, width, label="Observed Dropout Rate", color="#DD8452")
ax.axhline(y=baseline, color="gray", linestyle="--", linewidth=1, label=f"Baseline Prevalence ({baseline:.3f})")

ax.set_xlabel("Score Decile (1=Lowest, 10=Highest)")
ax.set_ylabel("Dropout Rate / Predicted Probability")
ax.set_title("Calibration by Prediction Decile (LightGBM Calibrated)")
ax.set_xticks(x)
ax.legend()
ax.set_ylim(0, 0.45)

plt.tight_layout()
plt.savefig("paper/figures/fig08_calibration_decile.pdf", dpi=150)
plt.savefig("paper/figures/fig08_calibration_decile.png", dpi=150)
print("Saved fig08_calibration_decile.pdf and .png")
