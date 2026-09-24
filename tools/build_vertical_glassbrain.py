#!/usr/bin/env python3
"""
Render a 7-column horizontal composite of task-specific glass-brain maps on
the real HCP-MMP1.0 (Glasser-360) volumetric atlas in MNI152 space. Top row
carries the task labels, bottom row the brains. This wide-short layout makes
each brain substantially larger than the earlier 7-row vertical arrangement
and fits cleanly into a single IEEE column at `\\linewidth` width.

The volumetric atlas is the PennLINC/xcpEngine release (`glasser360MNI.nii.gz`),
whose parcel indices run 1..180 for Right and 181..360 for Left. Our pipeline's
`Glasser360Indices_LR.dscalar.nii` uses the opposite convention (1..180 L,
181..360 R), so the two halves are swapped before writing voxel values.

Each task shows a dorsal (axial, z) glass-brain view, which gives clear
bilateral coverage and avoids cramming four sub-views into a narrow column.

Output: `results/run_<stamp>/figures/fig_tasks_vertical.{pdf,png}`.
"""

import os
import sys
from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
import nibabel as nib
from nilearn import plotting

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))
from configs import config

RUN_DIR = PROJECT_DIR / "results" / os.environ.get("CR_RUN", "run_20260419_182908")
OUT_STEM = RUN_DIR / "figures" / "fig_tasks_vertical"
ATLAS_NII = PROJECT_DIR / "data" / "_atlas" / "glasser360MNI.nii.gz"


def pennlinc_from_pipeline(activation_360):
    """Reorder 360-vector from pipeline (L then R) to PennLINC (R then L)."""
    a = np.asarray(activation_360)
    return np.concatenate([a[180:], a[:180]])


def activation_to_volume(activation_360, atlas_data):
    reordered = pennlinc_from_pipeline(activation_360)
    out = np.zeros_like(atlas_data, dtype=np.float32)
    for label in range(1, 361):
        mask = atlas_data == label
        if mask.any():
            out[mask] = reordered[label - 1]
    return out


def main():
    with open(RUN_DIR / "neuroscience" / "activation_matrix.pkl", "rb") as f:
        data = pickle.load(f)
    activation_matrix = data["data"]
    tasks = data["tasks"]
    avg_activation = np.nanmean(activation_matrix, axis=0)  # (7, 360)

    atlas = nib.load(str(ATLAS_NII))
    atlas_data = atlas.get_fdata().astype(int)
    affine = atlas.affine

    volumes = [activation_to_volume(avg_activation[t], atlas_data) for t in range(len(tasks))]
    all_vals = np.concatenate([v[v != 0] for v in volumes])
    vmax = float(np.nanpercentile(np.abs(all_vals), 97))
    vmin = -vmax

    task_names = {
        "WM": "Working\nMemory",
        "MOTOR": "Motor",
        "LANGUAGE": "Language",
        "SOCIAL": "Social\nCognition",
        "RELATIONAL": "Relational\nProcessing",
        "EMOTION": "Emotion",
        "GAMBLING": "Gambling",
    }

    n_tasks = len(tasks)
    # Wide-short 2-row layout: labels on top, one enlarged brain per task below.
    # A single display_mode="z" view per task keeps each brain large enough to
    # be read at column width, unlike the earlier 4-view `lzry` mode which
    # shrank each sub-view to ~0.5 in.
    fig = plt.figure(figsize=(18, 4.2), facecolor="white")
    gs = GridSpec(
        2, n_tasks + 1, figure=fig,
        height_ratios=[0.14, 1.0],
        width_ratios=[1.0] * n_tasks + [0.06],
        hspace=0.02, wspace=0.08,
        left=0.01, right=0.97, top=0.97, bottom=0.02,
    )
    cmap_name = "RdBu_r"

    # --- Row 1: task labels ---
    for c, task in enumerate(tasks):
        ax_lbl = fig.add_subplot(gs[0, c])
        ax_lbl.axis("off")
        ax_lbl.text(0.5, 0.25, task_names.get(task, task),
                    transform=ax_lbl.transAxes, ha="center", va="center",
                    fontsize=18, fontweight="bold")

    # --- Row 2: brains (single axial / dorsal view per task) ---
    for c, task in enumerate(tasks):
        ax = fig.add_subplot(gs[1, c])
        img = nib.Nifti1Image(volumes[c], affine=affine)
        display = plotting.plot_glass_brain(
            img, axes=ax, colorbar=False, plot_abs=False,
            cmap=cmap_name, vmin=vmin, vmax=vmax,
            display_mode="z", threshold=vmax * 0.05,
        )
        for sub_ax in display.axes.values():
            for txt in sub_ax.ax.texts:
                txt.set_fontsize(12)

    # --- Shared colorbar spanning both rows on the right ---
    cbar_ax = fig.add_subplot(gs[:, -1])
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_name)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label("z-scored activation", fontsize=14)
    cb.ax.tick_params(labelsize=12)

    for ext in ("pdf", "png"):
        out = OUT_STEM.with_suffix(f".{ext}")
        fig.savefig(str(out), dpi=200, bbox_inches="tight", facecolor="white")
        print(f"  saved {out}  ({out.stat().st_size // 1024} KB)")

    plt.close(fig)


if __name__ == "__main__":
    main()
