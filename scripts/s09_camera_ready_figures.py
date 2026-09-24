#!/usr/bin/env python3
"""
Stage 9 (camera-ready): the two single-column figures added for the BIBM 2026
camera-ready version.

  fig_null_distributions.pdf  [R4.1] naive pairwise-t, subject-level LOO and
                              label-permutation null distributions for H3 on
                              one axis (difference in routing consistency).
  fig_cross_domain.pdf        [R1.2, R1.4, R2.9] cortical Gini at matched unit
                              counts versus corrected MoE Gini, pooled over
                              layers and per layer. Replaces the submitted
                              Fig. 4, whose MoE values came from the faulty
                              router hook.

Reads results/run_20260419_182908/camera_ready/camera_ready_analyses.json,
perm_null.npy, and the corrected layer-wise MoE routing (sister project).
"""
import os
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT = PROJECT_DIR / "results" / os.environ.get("CR_RUN", "run_20260419_182908") / "camera_ready"
R = json.load(open(OUT / "camera_ready_analyses.json"))
perm = np.load(OUT / "perm_null.npy")
LAYERWISE = Path(os.environ.get("MOE_SOURCE", PROJECT_DIR / "results" / "moe_corrected")) \
    / "moe_analysis" / "layerwise_routing.json"

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "ps.fonttype": 42})
C_NAIVE, C_LOO, C_PERM, C_OBS = "#b2182b", "#2166ac", "#4d4d4d", "#000000"

# ----------------------------------------------------------------------------
# Figure: null distributions on one axis
# ----------------------------------------------------------------------------
G = R["G_null"]
h3 = R["F_precision"]["H3_LOO"]
H3R = json.load(open(OUT.parent / "neuroscience" / "h3_robust.json"))
naive_p = H3R["naive_reference"]["p"]
perm_p = H3R["label_permutation_test"]["p_value"]  # the s03b permutation reported in the text
_m, _e = f"{naive_p:.1e}".split("e")
naive_txt = rf"$p = {_m}\times10^{{{int(_e)}}}$"
x = np.linspace(-0.11, 0.11, 1200)
fig, axes = plt.subplots(3, 1, figsize=(3.45, 3.0), sharex=True)
rows = [
    ("Naive pairwise $t$ (1225 + 1225 pairs)", C_NAIVE, G["se_naive"], G["pairwise_diff"],
     naive_txt),
    ("Subject-level LOO Welch $t$ (50 + 50)", C_LOO, G["se_loo"], G["loo_diff"],
     rf"$p = {h3['p']:.3f}$"),
    ("Label permutation (10,000)", C_PERM, None, G["pairwise_diff"],
     rf"$p = {perm_p:.3f}$"),
]
for ax, (title, col, se, obs, ptxt) in zip(axes, rows):
    if se is not None:
        y = stats.norm.pdf(x, 0, se)
        ax.fill_between(x, y, color=col, alpha=0.35, lw=0)
        ax.plot(x, y, color=col, lw=0.9)
        ymax = y.max()
    else:
        cnt, edges, _ = ax.hist(perm, bins=60, density=True, color=col, alpha=0.45, lw=0)
        ymax = cnt.max()
    ax.axvline(obs, color=C_OBS, lw=1.0, ls="--")
    ax.set_ylim(0, ymax * 1.45)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.text(0.0, 0.98, title, transform=ax.transAxes, ha="left", va="top")
    ax.text(0.0, 0.78, ptxt, transform=ax.transAxes, ha="left", va="top")
axes[-1].set_xlabel(r"High $-$ low difference in routing consistency ($\Delta r$)")
axes[0].text(G["pairwise_diff"] + 0.003, axes[0].get_ylim()[1] * 0.35, "observed", fontsize=7)
fig.tight_layout(h_pad=0.3)
fig.savefig(OUT / "fig_null_distributions.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig_null_distributions.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ----------------------------------------------------------------------------
# Figure: cortex vs MoE Gini at matched unit counts
# ----------------------------------------------------------------------------
I = R["I_matched_gini"]
moe = R["J_moe"]["models"]
lw = json.load(open(LAYERWISE))["models"]
fig, ax = plt.subplots(figsize=(3.45, 2.25))
units = [("8", "Switch-8", 0), ("60", "Qwen-MoE", 1), ("64", "DeepSeek-16B", 2)]
for k, model, pos in units:
    m = I["subsampled"][k]
    ax.errorbar(pos - 0.18, m["mean"], yerr=[[m["mean"] - m["p2.5"]], [m["p97.5"] - m["mean"]]],
                fmt="s", color="#1b7837", ms=4, capsize=2, lw=0.9,
                label="Cortex, %s random parcels" % "$k$" if pos == 0 else None)
    per_layer = [L["gini"] for L in lw[model]["per_layer"]]
    jit = np.linspace(-0.07, 0.07, len(per_layer))
    ax.scatter(pos + 0.18 + jit, per_layer, s=6, facecolors="none", edgecolors="#762a83", lw=0.6,
               label="MoE, single layer" if pos == 0 else None)
    ax.scatter([pos + 0.18], [moe[model]["aggregate_gini"]], marker="D", s=22, color="#762a83",
               zorder=3, label="MoE, pooled over layers" if pos == 0 else None)
full = I["full_360"]["mean"]
ax.axhline(full, color="#1b7837", lw=0.8, ls=":")
ax.text(1.5, full + 0.015, "cortex, 360 parcels", color="#1b7837", fontsize=7, ha="center")
ax.set_xticks([0, 1, 2])
ax.set_xticklabels([f"{m}\n($k$ = {k})" for k, m, _ in units])
ax.set_xlim(-0.55, 2.55)
ax.set_ylim(0, 1)
ax.set_ylabel("Gini coefficient")
ax.legend(loc="upper left", frameon=False, ncol=1, handletextpad=0.3, borderaxespad=0.1)
fig.tight_layout()
fig.savefig(OUT / "fig_cross_domain.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig_cross_domain.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("figures written")
