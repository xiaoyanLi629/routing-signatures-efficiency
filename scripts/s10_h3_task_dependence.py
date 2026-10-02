#!/usr/bin/env python3
"""
Stage 10 (camera-ready): dependence of H3 on the Working Memory task.

The efficiency score is computed from Working Memory behavior, and WM is also
one of the seven tasks in each participant's 7 x 7 routing profile. This stage
re-runs the H3 tests with one task removed at a time, and with the WM row alone:

  - group test: subject-level LOO r (Welch t, Mann-Whitney) and the label
    permutation on within-group mean pairwise r, using the s03b functions;
  - continuous score: Pearson / Spearman r between the efficiency score and
    LOO r against the whole-sample template, as in s08 (D_continuous), plus a
    permutation of the efficiency score.

Reads  results/<CR_RUN>/neuroscience/{routing_profiles.pkl, efficiency_groups.json,
       activation_matrix.pkl, behavioral_validation.csv}
Writes results/<CR_RUN>/camera_ready/h3_task_dependence.json
"""
import os
import json
import pickle
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_DIR = Path(__file__).resolve().parent.parent
RUN = PROJECT_DIR / "results" / os.environ.get("CR_RUN", "run_20260924_glmfix")
NEURO = RUN / "neuroscience"
N_PERM = 10000

spec = importlib.util.spec_from_file_location("s03b", PROJECT_DIR / "scripts" / "s03b_h3_robust.py")
s03b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s03b)

act = pickle.load(open(NEURO / "activation_matrix.pkl", "rb"))
SUBJ = [str(s) for s in act["subjects"]]
TASKS = list(act["tasks"])
profiles = {str(s): np.asarray(m) for s, m in pickle.load(open(NEURO / "routing_profiles.pkl", "rb")).items()}
groups = json.load(open(NEURO / "efficiency_groups.json"))["groups"]
high = [str(s) for s in groups["high"] if str(s) in profiles]
low = [str(s) for s in groups["low"] if str(s) in profiles]
beh = pd.read_csv(NEURO / "behavioral_validation.csv", dtype={"subject": str}).set_index("subject")
eff = beh.loc[SUBJ, "efficiency_score"].values


def group_tests(prof):
    rh, rl = s03b.loo_group_mean_r(prof, high), s03b.loo_group_mean_r(prof, low)
    t, p = stats.ttest_ind(rh, rl, equal_var=False)
    _, pu = stats.mannwhitneyu(rh, rl, alternative="two-sided")
    obs, pp, _, _ = s03b.permutation_test_vectorized(prof, high, low)
    return {"mean_high": float(rh.mean()), "mean_low": float(rl.mean()), "d": float(s03b.cohens_d(rh, rl)),
            "welch_t": float(t), "p": float(p), "mannwhitney_p": float(pu),
            "perm_diff": float(obs), "perm_p": float(pp)}


def continuous(prof):
    F = np.stack([prof[s].ravel() for s in SUBJ])
    loo = np.array([stats.pearsonr(F[i], np.delete(F, i, axis=0).mean(axis=0))[0] for i in range(len(F))])
    r, p = stats.pearsonr(eff, loo)
    rho, prho = stats.spearmanr(eff, loo)
    z, se = np.arctanh(r), 1 / np.sqrt(len(loo) - 3)
    rng = np.random.default_rng(0)
    null = np.array([stats.pearsonr(rng.permutation(eff), loo)[0] for _ in range(N_PERM)])
    return {"pearson_r": float(r), "r_ci95": [float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))],
            "p": float(p), "spearman_rho": float(rho), "p_spearman": float(prho),
            "perm_p": float((np.sum(np.abs(null) >= abs(r)) + 1) / (N_PERM + 1))}


def subset(rows):
    return {s: m[rows] for s, m in profiles.items()}


out = {"tasks": TASKS, "all_tasks": {"group": group_tests(profiles), "continuous": continuous(profiles)},
       "drop_one_task": {}}
for j, t in enumerate(TASKS):
    rows = [k for k in range(len(TASKS)) if k != j]
    out["drop_one_task"][t] = {"group": group_tests(subset(rows)), "continuous": continuous(subset(rows))}
wm = TASKS.index("WM")
out["wm_only"] = {"group": group_tests(subset([wm])), "continuous": continuous(subset([wm]))}

dest = RUN / "camera_ready" / "h3_task_dependence.json"
json.dump(out, open(dest, "w"), indent=2)
for name, v in [("all", out["all_tasks"])] + [("drop " + t, v) for t, v in out["drop_one_task"].items()] \
        + [("WM only", out["wm_only"])]:
    g, c = v["group"], v["continuous"]
    print(f"{name:16s} d={g['d']:+.2f} p={g['p']:.4f} perm_p={g['perm_p']:.4f} | "
          f"r_E={c['pearson_r']:+.2f} [{c['r_ci95'][0]:+.2f}, {c['r_ci95'][1]:+.2f}] p={c['p']:.4f}")
print("written", dest)
