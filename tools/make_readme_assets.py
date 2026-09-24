#!/usr/bin/env python3
"""
Build the images and animations used in README.md.

  assets/naive_vs_honest.gif   the label-permutation null building up, next to
                               the null implied by the naive pairwise t-test
  assets/task_brains.gif       group-mean activation of each HCP task, four
                               glass-brain views per frame
  assets/*.png                 static versions of the paper figures

Inputs are the outputs of scripts/run_glmfix.sh (CR_RUN, default
run_20260924_glmfix) and the Glasser-360 MNI atlas in data/_atlas/.

Usage:  python tools/make_readme_assets.py
"""
import io
import os
import json
import pickle
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nilearn import plotting
from PIL import Image
from scipy import stats

PROJECT_DIR = Path(__file__).resolve().parent.parent
RUN = PROJECT_DIR / "results" / os.environ.get("CR_RUN", "run_20260924_glmfix")
ASSETS = PROJECT_DIR / "assets"
ASSETS.mkdir(exist_ok=True)
ATLAS_NII = PROJECT_DIR / "data" / "_atlas" / "glasser360MNI.nii.gz"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})
C_NAIVE, C_LOO, C_PERM = "#c0392b", "#2166ac", "#7f8c8d"


def fig_to_image(fig, dpi=110):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def save_gif(frames, path, durations):
    pal = [f.convert("P", palette=Image.ADAPTIVE, colors=128) for f in frames]
    pal[0].save(path, save_all=True, append_images=pal[1:], duration=durations,
                loop=0, optimize=True, disposal=2)
    print(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB, {len(frames)} frames)")


# ----------------------------------------------------------------------------
# 1. Naive vs honest null
# ----------------------------------------------------------------------------
def naive_vs_honest():
    R = json.load(open(RUN / "camera_ready" / "camera_ready_analyses.json"))
    h3 = json.load(open(RUN / "neuroscience" / "h3_robust.json"))
    perm = np.load(RUN / "camera_ready" / "perm_null.npy")
    G = R["G_null"]
    obs = G["pairwise_diff"]
    x = np.linspace(-0.11, 0.11, 800)
    naive = stats.norm.pdf(x, 0, G["se_naive"])
    bins = np.linspace(-0.11, 0.11, 56)
    width = bins[1] - bins[0]
    steps = np.unique(np.geomspace(20, len(perm), 34).astype(int))
    naive_p = h3["naive_reference"]["p"]
    perm_p = h3["label_permutation_test"]["p_value"]
    m, e = f"{naive_p:.1e}".split("e")
    frames, durs = [], []
    for k, n in enumerate(steps):
        fig, ax = plt.subplots(figsize=(8, 4.2))
        cnt, _ = np.histogram(perm[:n], bins=bins)
        dens = cnt / (n * width)
        ax.bar(bins[:-1], dens, width=width, align="edge", color=C_PERM, alpha=0.55,
               label=f"label-permutation null ({n:,} permutations)")
        ax.fill_between(x, naive, color=C_NAIVE, alpha=0.18)
        ax.plot(x, naive, color=C_NAIVE, lw=2, label="null assumed by the naive pairwise t-test")
        ax.axvline(obs, color="black", lw=2, ls="--")
        ax.text(obs + 0.002, naive.max() * 0.93, "observed\ndifference", fontsize=10, va="top")
        tail = np.mean(np.abs(perm[:n]) >= abs(obs))
        ax.text(-0.108, naive.max() * 0.95,
                f"naive t-test:  p = {m}e{int(e)}\npermutation:   p = {max(tail, 1 / n):.3f}"
                if n < len(perm) else
                f"naive t-test:  p = {m}e{int(e)}\npermutation:   p = {perm_p:.3f}",
                family="monospace", fontsize=11, va="top",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc"))
        ax.set_xlim(-0.11, 0.11)
        ax.set_ylim(0, naive.max() * 1.08)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("High − low difference in inter-subject routing consistency (Δr)")
        ax.set_title("Same data, two tests: pooling 2,450 correlated pairs makes the null far too narrow",
                     fontsize=11.5)
        ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.74), frameon=False, fontsize=9.5)
        fig.tight_layout()
        frames.append(fig_to_image(fig))
        durs.append(2600 if k == len(steps) - 1 else 110)
    save_gif(frames, ASSETS / "naive_vs_honest.gif", durs)


# ----------------------------------------------------------------------------
# 2. Task glass brains
# ----------------------------------------------------------------------------
def task_brains():
    d = pickle.load(open(RUN / "neuroscience" / "activation_matrix.pkl", "rb"))
    avg = np.nanmean(d["data"], axis=0)
    tasks = d["tasks"]
    atlas = nib.load(str(ATLAS_NII))
    lab = atlas.get_fdata().astype(int)

    def to_vol(v):
        v = np.concatenate([v[180:], v[:180]])  # pipeline L,R -> atlas R,L order
        out = np.zeros(lab.shape, dtype=np.float32)
        for p in range(1, 361):
            out[lab == p] = v[p - 1]
        return nib.Nifti1Image(out, atlas.affine)

    vmax = float(np.percentile(np.abs(avg), 97))
    names = {"WM": "Working memory · 2-back vs 0-back", "MOTOR": "Motor · movement vs fixation",
             "LANGUAGE": "Language · story vs math", "SOCIAL": "Social cognition · mental vs random",
             "RELATIONAL": "Relational processing · relation vs match",
             "EMOTION": "Emotion · fearful faces vs shapes", "GAMBLING": "Gambling · win vs loss"}
    frames = []
    for j, t in enumerate(tasks):
        fig = plt.figure(figsize=(9, 3.1), facecolor="white")
        plotting.plot_glass_brain(to_vol(avg[j]), figure=fig, display_mode="lyrz", plot_abs=False,
                                  cmap="RdBu_r", vmin=-vmax, vmax=vmax, threshold=vmax * 0.05,
                                  colorbar=True, title=f"{names[t]}   (N = 100, z)")
        frames.append(fig_to_image(fig, dpi=100))
    save_gif(frames, ASSETS / "task_brains.gif", [1600] * len(frames))


# ----------------------------------------------------------------------------
# 3. Static figures
# ----------------------------------------------------------------------------
def statics():
    import shutil
    import subprocess
    pairs = {RUN / "camera_ready" / "fig_null_distributions.png": "fig_null_distributions.png",
             RUN / "camera_ready" / "fig_cross_domain.png": "fig_cross_domain.png",
             RUN / "figures" / "fig_tasks_vertical.png": "fig_task_maps.png"}
    for src, dst in pairs.items():
        shutil.copy(src, ASSETS / dst)
    pdf = RUN / "figures" / "fig04_modularity_comparison.pdf"
    subprocess.run(["pdftoppm", "-r", "130", "-png", "-singlefile", str(pdf),
                    str(ASSETS / "fig_modularity")], check=True)
    print("static figures copied")


if __name__ == "__main__":
    import sys
    todo = sys.argv[1:] or ["statics", "naive", "brains"]
    if "statics" in todo:
        statics()
    if "naive" in todo:
        naive_vs_honest()
    if "brains" in todo:
        task_brains()
