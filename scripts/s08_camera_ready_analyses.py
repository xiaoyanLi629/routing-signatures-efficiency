#!/usr/bin/env python3
"""
=============================================================================
Stage 8 (camera-ready): analyses requested by the BIBM 2026 reviewers
=============================================================================
Everything here reads the frozen outputs of run_20260419_182908 (the run the
submitted manuscript reports) plus, for the MoE side, the corrected router
extraction produced in the sister project (see MOE_SOURCE below). Nothing is
re-extracted from raw fMRI except the per-run head-motion summaries and the
per-run WM behavioural summaries that HCP ships as text files.

Sections (reviewer in brackets):
  A. Sample description: NaN cells, head motion             [R2.8, R3.3]
  B. Efficiency score: run-to-run (LR vs RL) reliability    [R3.4]
  C. H1 at the subject level (the submitted H1 test pooled
     700 subject-task rows as if independent)              [R2.7]
  D. Continuous efficiency-score sensitivity analyses       [R2.1]
  E. H2 as a group test, and H2 at the native 12-network
     Cole-Anticevic resolution                              [R2.2, R2.3, R3.1]
  F. Effect-size CIs, TOST equivalence, JZS Bayes factors,
     minimum detectable effect, prospective N               [R2.7, R2.8, R3.2]
  G. Null distributions of naive / LOO / permutation tests  [R4.1]
  H. Louvain seed stability and a positive-edge-only graph  [R2.6]
  I. Parcel-count-matched cortical Gini                     [R1.2, R2.9]
  J. MoE: corrected extraction, per-layer and per-sequence
     concentration, token-level sparsity                    [R1.4]
  K. Cross-domain structure: permutation-invariant Mantel
     test replacing the submitted flattened-matrix r         [R1.3]

H2 canonical expectations at 12-network resolution are fixed BEFORE looking at
the 12-network result by carrying each Yeo-7 expectation over to the Cole-
Anticevic networks that config.COLE_TO_YEO7 collapses into it. No new
expectation is introduced.

Output: results/run_20260419_182908/camera_ready/camera_ready_analyses.json
        results/run_20260419_182908/camera_ready/fig_null_distributions.pdf
        results/run_20260419_182908/camera_ready/fig_cross_domain.pdf
=============================================================================
"""
import os
# One BLAS thread per process: the Louvain pool below forks workers, and
# default BLAS threading multiplied by the pool size overloads a shared host.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import os
import json
import pickle
import itertools
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import stats, integrate
import networkx as nx
from networkx.algorithms.community import louvain_communities
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import TTestIndPower

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
from configs import config  # noqa: E402
from scripts.s02_sparsity_analysis import (  # noqa: E402
    compute_gini_coefficient, compute_entropy_sparsity, compute_concentration_ratio)
from scripts.s04_functional_modularity import (  # noqa: E402
    compute_functional_connectivity, compute_modularity,
    compute_module_segregation, compute_participation_coefficient)

RUN = PROJECT_DIR / "results" / os.environ.get("CR_RUN", "run_20260419_182908")
NEURO = RUN / "neuroscience"
OUT = RUN / "camera_ready"
OUT.mkdir(parents=True, exist_ok=True)
DATA_ROOT = config.DATA_ROOT
# Corrected MoE router extraction (sister project, commits 55c0cf0 / 8b0e77b):
# same three checkpoints, same text samples, fixed hook classification.
MOE_SOURCE = Path("/root/autodl-fs/TST_Journal_2026/Routing_MoE_TST/results/"
                  "run_20260419_182908")
MOE_MODELS = {"switch_8": "Switch-8", "qwen_moe": "Qwen-MoE", "deepseek_16b": "DeepSeek-16B"}
CATS = ["scientific", "social", "emotional", "procedural", "narrative",
        "mathematical", "risk_decision"]
CAT2TASK = {c['name']: c['analog_task'] for c in config.MOE_CONFIG['input_categories']}
COLE12 = {1: 'Visual1', 2: 'Visual2', 3: 'Somatomotor', 4: 'Cingulo-Opercular',
          5: 'Dorsal-Attention', 6: 'Language', 7: 'Frontoparietal', 8: 'Auditory',
          9: 'Default', 10: 'Posterior-Multimodal', 11: 'Ventral-Multimodal',
          12: 'Orbito-Affective'}
SEED = 20260924
rng = np.random.default_rng(SEED)
res = {}


def cohens_d(a, b):
    """Pooled-SD Cohen's d (ddof=1)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / sp)


def d_ci(d, na, nb, alpha=0.05):
    """Large-sample CI for Cohen's d (Hedges & Olkin SE)."""
    se = np.sqrt((na + nb) / (na * nb) + d ** 2 / (2 * (na + nb)))
    z = stats.norm.ppf(1 - alpha / 2)
    return [float(d - z * se), float(d + z * se)]


def tost(a, b, bound_d=0.5):
    """Two one-sided Welch tests against +/- bound_d (in pooled-SD units)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    delta = bound_d * sp
    diff = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / na + b.var(ddof=1) / nb)
    df = (a.var(ddof=1) / na + b.var(ddof=1) / nb) ** 2 / (
        (a.var(ddof=1) / na) ** 2 / (na - 1) + (b.var(ddof=1) / nb) ** 2 / (nb - 1))
    p_lower = 1 - stats.t.cdf((diff + delta) / se, df)
    p_upper = stats.t.cdf((diff - delta) / se, df)
    return {"bound_d": bound_d, "p_tost": float(max(p_lower, p_upper)),
            "equivalent": bool(max(p_lower, p_upper) < 0.05)}


def jzs_bf10(t, n1, n2, r=np.sqrt(2) / 2):
    """Two-sample JZS Bayes factor BF10 (Rouder et al., 2009), Cauchy prior scale r."""
    nu = n1 + n2 - 2
    neff = n1 * n2 / (n1 + n2)

    def integrand(g):
        return ((1 + neff * g * r ** 2) ** -0.5
                * (1 + t ** 2 / ((1 + neff * g * r ** 2) * nu)) ** (-(nu + 1) / 2)
                * (2 * np.pi) ** -0.5 * g ** -1.5 * np.exp(-1 / (2 * g)))
    num, _ = integrate.quad(integrand, 0, np.inf, limit=200)
    den = (1 + t ** 2 / nu) ** (-(nu + 1) / 2)
    return float(num / den)


def group_test(a, b, welch=True, name=""):
    t, p = stats.ttest_ind(a, b, equal_var=not welch)
    d = cohens_d(a, b)
    return {"name": name, "n_high": len(a), "n_low": len(b),
            "mean_high": float(np.mean(a)), "mean_low": float(np.mean(b)),
            "t": float(t), "p": float(p), "d": d, "d_ci95": d_ci(d, len(a), len(b)),
            "tost_d0.5": tost(a, b, 0.5), "bf10": jzs_bf10(float(t), len(a), len(b))}


# ----------------------------------------------------------------------------
# Load frozen inputs
# ----------------------------------------------------------------------------
act = pickle.load(open(NEURO / "activation_matrix.pkl", "rb"))
A = act["data"]                      # (100, 7, 360)
SUBJ = [str(s) for s in act["subjects"]]
TASKS = act["tasks"]
groups = json.load(open(NEURO / "efficiency_groups.json"))["groups"]
HIGH, LOW = set(groups["high"]), set(groups["low"])
is_high = np.array([s in HIGH for s in SUBJ])
beh = pd.read_csv(NEURO / "behavioral_validation.csv", dtype={"subject": str}).set_index("subject")
eff = beh.loc[SUBJ, "efficiency_score"].values
profiles = pickle.load(open(NEURO / "routing_profiles.pkl", "rb"))
P = np.stack([np.asarray(profiles[s]) for s in SUBJ])  # (100, 7, 7)
mod = pd.read_csv(NEURO / "modularity_metrics.csv", dtype={"subject": str}).set_index("subject").loc[SUBJ]

# ----------------------------------------------------------------------------
# A. Sample description
# ----------------------------------------------------------------------------
fd = np.full((len(SUBJ), len(TASKS), 2), np.nan)
for i, s in enumerate(SUBJ):
    for j, t in enumerate(TASKS):
        for k, run in enumerate(["LR", "RL"]):
            f = DATA_ROOT / s / "MNINonLinear" / "Results" / f"tfMRI_{t}_{run}" / "Movement_RelativeRMS_mean.txt"
            if f.exists():
                fd[i, j, k] = float(f.read_text().split()[0])
subj_fd = np.nanmean(fd.reshape(len(SUBJ), -1), axis=1)
res["A_sample"] = {
    "n_subject_task_cells": int(A.shape[0] * A.shape[1]),
    "n_nan_cells": int(np.isnan(A[:, :, 0]).sum()),
    "n_runs_with_motion_file": int(np.isfinite(fd).sum()),
    "relRMS_subject_mean": {"mean": float(np.nanmean(subj_fd)), "sd": float(np.nanstd(subj_fd, ddof=1)),
                            "max": float(np.nanmax(subj_fd))},
    "n_subjects_mean_relRMS_ge_0.5": int((subj_fd >= 0.5).sum()),
    "n_runs_relRMS_ge_0.5": int((fd >= 0.5).sum()),
    "max_run_relRMS": float(np.nanmax(fd)),
    "relRMS_high_vs_low": group_test(subj_fd[is_high], subj_fd[~is_high], name="relRMS"),
    "demographics_note": "age bands from HCP open access: 22-25:17, 26-30:42, 31-35:40, 36+:1; 53 F / 47 M",
}

# ----------------------------------------------------------------------------
# B. Efficiency score reliability (LR vs RL run)
# ----------------------------------------------------------------------------
def run_eff(s, run):
    f = DATA_ROOT / s / "MNINonLinear" / "Results" / f"tfMRI_WM_{run}" / "EVs" / "WM_Stats.csv"
    if not f.exists():
        return np.nan
    w = pd.read_csv(f)
    acc = w[w.Measure == "ACC"].Value.mean()
    rt = w[w.Measure == "MEDIAN_RT"].Value.mean()
    return acc / rt * 1000 if rt > 0 else np.nan

e_lr = np.array([run_eff(s, "LR") for s in SUBJ])
e_rl = np.array([run_eff(s, "RL") for s in SUBJ])
ok = np.isfinite(e_lr) & np.isfinite(e_rl)
r_runs, p_runs = stats.pearsonr(e_lr[ok], e_rl[ok])
rho_runs = stats.spearmanr(e_lr[ok], e_rl[ok])[0]
res["B_efficiency_reliability"] = {
    "n": int(ok.sum()), "r_LR_RL": float(r_runs), "p": float(p_runs),
    "spearman_LR_RL": float(rho_runs),
    "spearman_brown": float(2 * r_runs / (1 + r_runs)),
    "r_runmean_vs_published_score": float(stats.pearsonr(((e_lr + e_rl) / 2)[ok], eff[ok])[0]),
    # How stable is the median split itself?
    "split_agreement_LR_vs_RL": float(np.mean((e_lr[ok] > np.median(e_lr[ok])) ==
                                              (e_rl[ok] > np.median(e_rl[ok])))),
}

# ----------------------------------------------------------------------------
# C. H1 at the subject level
# ----------------------------------------------------------------------------
sp = {m: np.zeros((len(SUBJ), len(TASKS))) for m in ["gini", "entropy", "conc"]}
for i in range(len(SUBJ)):
    for j in range(len(TASKS)):
        x = A[i, j]
        sp["gini"][i, j] = compute_gini_coefficient(x)
        sp["entropy"][i, j] = compute_entropy_sparsity(x)
        sp["conc"][i, j] = compute_concentration_ratio(x)
sp["composite"] = (sp["gini"] + sp["entropy"] + sp["conc"]) / 3
h1 = {}
for m, M in sp.items():
    subj_mean = M.mean(axis=1)
    h1[m] = {"subject_level": group_test(subj_mean[is_high], subj_mean[~is_high], welch=False, name=m),
             "pooled_700_rows_as_submitted": {
                 "t": float(stats.ttest_ind(M[is_high].ravel(), M[~is_high].ravel())[0]),
                 "p": float(stats.ttest_ind(M[is_high].ravel(), M[~is_high].ravel())[1])}}
h1_family = ["gini", "entropy", "conc", "composite"]
h1_fdr = multipletests([h1[m]["subject_level"]["p"] for m in h1_family], method="fdr_bh")[1]
for m, q in zip(h1_family, h1_fdr):
    h1[m]["subject_level"]["p_fdr_within_H1"] = float(q)
# Per-task composite (7 tests), already one row per subject
per_task = {}
for j, t in enumerate(TASKS):
    c = sp["composite"][:, j]
    per_task[t] = group_test(c[is_high], c[~is_high], welch=False, name=t)
q = multipletests([per_task[t]["p"] for t in TASKS], method="fdr_bh")[1]
for t, qq in zip(TASKS, q):
    per_task[t]["p_fdr"] = float(qq)
res["C_H1_subject_level"] = {"overall": h1, "per_task_composite": per_task,
                             "brain_gini_subject_task_mean": float(sp["gini"].mean()),
                             "brain_gini_subject_task_sd": float(sp["gini"].std(ddof=1))}

# ----------------------------------------------------------------------------
# H3 LOO-r (reproduced) and H4 metrics
# ----------------------------------------------------------------------------
def loo_r(Pflat, idx):
    out = np.zeros(len(idx))
    for n, i in enumerate(idx):
        others = [k for k in idx if k != i]
        out[n] = stats.pearsonr(Pflat[i], Pflat[others].mean(axis=0))[0]
    return out

Pflat = P.reshape(len(SUBJ), -1)
hi_idx, lo_idx = list(np.where(is_high)[0]), list(np.where(~is_high)[0])
loo_hi, loo_lo = loo_r(Pflat, hi_idx), loo_r(Pflat, lo_idx)
h3 = group_test(loo_hi, loo_lo, welch=True, name="H3_LOO_r")
h4 = {m: group_test(mod[m].values[is_high], mod[m].values[~is_high], welch=False, name=m)
      for m in ["modularity_Q", "segregation", "mean_participation"]}

# ----------------------------------------------------------------------------
# E. H2: group test and 12-network resolution
# ----------------------------------------------------------------------------
def entropy_sel(vec):
    a = np.abs(vec)
    if a.sum() == 0:
        return 0.0
    p = a / a.sum()
    return float(1 - (-(p * np.log2(p + 1e-10)).sum()) / np.log2(len(vec)))

sel7_subj = np.array([np.mean([entropy_sel(P[i, t]) for t in range(7)]) for i in range(len(SUBJ))])
h2_group = group_test(sel7_subj[is_high], sel7_subj[~is_high], welch=True, name="H2_selectivity_yeo7")

import nibabel as nib  # noqa: E402
g = nib.load(str(config.GLASSER_INDICES_CIFTI)).get_fdata().flatten().astype(int)
n12 = nib.load(str(config.COLE_NETWORK_CIFTI)).get_fdata().flatten().astype(int)
parcel12 = {}
for pid in range(1, 361):
    nets = n12[g == pid]
    nets = nets[nets > 0]
    if len(nets):
        parcel12[pid - 1] = int(np.bincount(nets).argmax())
nets12 = sorted(set(parcel12.values()))
names12 = [COLE12[n] for n in nets12]
# Expectations carried over from Yeo-7 (no new choices): a Yeo-7 expectation E
# becomes the set of Cole networks that COLE_TO_YEO7 maps to E.
exp12 = {}
for t in TASKS:
    exp_y7 = config.TASKS[t]['expected_networks']
    exp12[t] = [COLE12[c] for c, y in config.COLE_TO_YEO7.items() if y in exp_y7]
M12 = np.zeros((len(SUBJ), len(TASKS), len(nets12)))
for k, n in enumerate(nets12):
    idx = [p for p, c in parcel12.items() if c == n]
    M12[:, :, k] = A[:, :, idx].mean(axis=2)
G12 = M12.mean(axis=0)
h2_12 = {}
for j, t in enumerate(TASKS):
    order = np.argsort(-G12[j])
    ranked = [names12[o] for o in order]
    best_rank = min(ranked.index(e) for e in exp12[t]) + 1
    h2_12[t] = {"expected": exp12[t], "argmax": ranked[0], "top3": ranked[:3],
                "match": ranked[0] in exp12[t], "rank_of_best_expected": int(best_rank),
                "entropy_selectivity_12": entropy_sel(G12[j])}
n_match12 = sum(v["match"] for v in h2_12.values())
# Permutation baseline for the number of matches: shuffle network labels of the
# group-mean vectors (task-wise), count how often >= observed matches arise.
null_matches = []
for _ in range(10000):
    cnt = 0
    for j, t in enumerate(TASKS):
        cnt += names12[rng.permutation(len(nets12))[np.argmax(G12[j])]] in exp12[t]
    null_matches.append(cnt)
null_matches = np.array(null_matches)
sel12_subj = np.array([np.mean([entropy_sel(M12[i, t]) for t in range(7)]) for i in range(len(SUBJ))])
res["E_H2"] = {
    "yeo7_group_test_selectivity": h2_group,
    "yeo7_mean_selectivity": float(sel7_subj.mean()),
    "cole12": {"n_parcels_per_network": {COLE12[n]: int(sum(1 for c in parcel12.values() if c == n)) for n in nets12},
               "per_task": h2_12, "n_match": int(n_match12),
               "chance_expected_matches": float(null_matches.mean()),
               "p_perm_ge_observed": float(np.mean(null_matches >= n_match12)),
               "mean_selectivity_12": float(np.mean([v["entropy_selectivity_12"] for v in h2_12.values()])),
               "group_test_selectivity_12": group_test(sel12_subj[is_high], sel12_subj[~is_high], name="H2_sel12")},
}

# ----------------------------------------------------------------------------
# FDR families: pre-specified primary (H1, H3, H4) and sensitivity (+H2)
# ----------------------------------------------------------------------------
prim = {"H1": h1["composite"]["subject_level"]["p"], "H3": h3["p"], "H4": h4["modularity_Q"]["p"]}
q3 = multipletests(list(prim.values()), method="fdr_bh")[1]
q4 = multipletests(list(prim.values()) + [h2_group["p"]], method="fdr_bh")[1]
res["FDR"] = {"primary_H1_H3_H4": dict(zip(prim, map(float, q3))),
              "sensitivity_H1_H2_H3_H4": dict(zip(list(prim) + ["H2"], map(float, q4))),
              "raw": {**prim, "H2": h2_group["p"]}}
h4_fdr = multipletests([h4[m]["p"] for m in h4], method="fdr_bh")[1]
for m, qq in zip(h4, h4_fdr):
    h4[m]["p_fdr_within_H4"] = float(qq)

# ----------------------------------------------------------------------------
# D. Continuous efficiency score
# ----------------------------------------------------------------------------
Pall = Pflat
loo_all = loo_r(Pall, list(range(len(SUBJ))))  # vs whole-sample LOO template
cont_vars = {"composite_sparsity": sp["composite"].mean(axis=1),
             "H2_selectivity_yeo7": sel7_subj,
             "H3_LOO_r_whole_sample": loo_all,
             "H4_modularity_Q": mod["modularity_Q"].values,
             "H4_segregation": mod["segregation"].values,
             "H4_participation": mod["mean_participation"].values}
cont = {}
for k, v in cont_vars.items():
    r, p = stats.pearsonr(eff, v)
    rho, prho = stats.spearmanr(eff, v)
    z = np.arctanh(r); se = 1 / np.sqrt(len(v) - 3)
    cont[k] = {"pearson_r": float(r), "r_ci95": [float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))],
               "p": float(p), "spearman_rho": float(rho), "p_spearman": float(prho)}
qc = multipletests([cont[k]["p"] for k in cont], method="fdr_bh")[1]
for k, qq in zip(cont, qc):
    cont[k]["p_fdr"] = float(qq)
res["D_continuous"] = cont

# ----------------------------------------------------------------------------
# F. Precision: MDE and prospective N
# ----------------------------------------------------------------------------
pw = TTestIndPower()
res["F_precision"] = {
    "H1_composite": h1["composite"]["subject_level"], "H3_LOO": h3, "H4": h4,
    "mde_d_80pct_n50": float(pw.solve_power(nobs1=50, alpha=0.05, power=0.8, ratio=1)),
    "n_per_group_80pct_d0.40": float(pw.solve_power(effect_size=0.40, alpha=0.05, power=0.8, ratio=1)),
    "n_per_group_80pct_d0.40_alpha_fdr_rank1_of3": float(pw.solve_power(effect_size=0.40, alpha=0.05 / 3, power=0.8, ratio=1)),
}

# ----------------------------------------------------------------------------
# G. Null distributions: naive pairwise t, LOO Welch, label permutation
# ----------------------------------------------------------------------------
Xn = Pflat - Pflat.mean(axis=1, keepdims=True)
Xn = Xn / np.linalg.norm(Xn, axis=1, keepdims=True)
R = Xn @ Xn.T

def within_pairs(idx):
    idx = np.asarray(idx)
    sub = R[np.ix_(idx, idx)]
    return sub[np.triu_indices(len(idx), 1)]

pr_hi, pr_lo = within_pairs(hi_idx), within_pairs(lo_idx)
obs_pair = pr_hi.mean() - pr_lo.mean()
se_naive = np.sqrt(pr_hi.var(ddof=1) / len(pr_hi) + pr_lo.var(ddof=1) / len(pr_lo))
obs_loo = loo_hi.mean() - loo_lo.mean()
se_loo = np.sqrt(loo_hi.var(ddof=1) / len(loo_hi) + loo_lo.var(ddof=1) / len(loo_lo))
perm = np.zeros(10000)
all_idx = np.arange(len(SUBJ))
for b in range(10000):
    pidx = rng.permutation(all_idx)
    perm[b] = within_pairs(pidx[:50]).mean() - within_pairs(pidx[50:]).mean()
res["G_null"] = {"pairwise_diff": float(obs_pair), "se_naive": float(se_naive),
                 "z_naive": float(obs_pair / se_naive),
                 "loo_diff": float(obs_loo), "se_loo": float(se_loo), "z_loo": float(obs_loo / se_loo),
                 "perm_sd": float(perm.std()), "p_perm": float(np.mean(np.abs(perm) >= abs(obs_pair))),
                 "se_ratio_perm_over_naive": float(perm.std() / se_naive)}
np.save(OUT / "perm_null.npy", perm)

# ----------------------------------------------------------------------------
# H. Louvain seed stability, positive-only graph
# ----------------------------------------------------------------------------
N_SEEDS = 20

def louvain_labels(W, seed):
    G = nx.from_numpy_array(W)
    comms = louvain_communities(G, resolution=1.0, seed=seed, weight='weight')
    lab = np.zeros(len(W), dtype=int)
    for c, nodes in enumerate(comms):
        lab[list(nodes)] = c
    return lab

def one_subject(i):
    C = compute_functional_connectivity(A[i])
    W = np.abs(C).copy(); np.fill_diagonal(W, 0); W[W < 1e-6] = 0
    labs = [louvain_labels(W, s) for s in range(N_SEEDS)]
    Qs = [compute_modularity(C, l) for l in labs]
    segs = [compute_module_segregation(C, l) for l in labs]
    parts = [float(np.mean(compute_participation_coefficient(C, l))) for l in labs]
    ari = [adjusted_rand_score(labs[a], labs[b]) for a, b in itertools.combinations(range(N_SEEDS), 2)]
    nmi = [normalized_mutual_info_score(labs[a], labs[b]) for a, b in itertools.combinations(range(N_SEEDS), 2)]
    # positive-edge-only graph (negative correlations dropped instead of folded)
    Wp = np.clip(C, 0, None); np.fill_diagonal(Wp, 0)
    lp = louvain_labels(Wp, 0)
    m = Wp.sum() / 2; k = Wp.sum(axis=1)
    Qpos = float(sum(Wp[np.ix_(lp == c, lp == c)].sum() - k[lp == c].sum() ** 2 / (2 * m)
                     for c in np.unique(lp)) / (2 * m))
    return {"Q_mean": float(np.mean(Qs)), "Q_sd": float(np.std(Qs)), "seg_mean": float(np.mean(segs)),
            "part_mean": float(np.mean(parts)), "ari_mean": float(np.mean(ari)),
            "nmi_mean": float(np.mean(nmi)), "ncomm": [int(l.max() + 1) for l in labs],
            "Q_pos": Qpos, "ncomm_pos": int(lp.max() + 1)}

with Pool(12) as pool:
    lv = pool.map(one_subject, range(len(SUBJ)))
lvdf = pd.DataFrame(lv)
res["H_louvain"] = {
    "n_seeds": N_SEEDS,
    "ARI_between_seeds": {"median": float(lvdf.ari_mean.median()), "min": float(lvdf.ari_mean.min())},
    "NMI_between_seeds": {"median": float(lvdf.nmi_mean.median()), "min": float(lvdf.nmi_mean.min())},
    "Q_within_subject_sd_median": float(lvdf.Q_sd.median()),
    "Q_between_subject_sd": float(lvdf.Q_mean.std()),
    "seed_averaged_tests": {m: group_test(lvdf[c].values[is_high], lvdf[c].values[~is_high], welch=False, name=m)
                            for m, c in [("Q", "Q_mean"), ("segregation", "seg_mean"), ("participation", "part_mean")]},
    "positive_only_Q_test": group_test(lvdf.Q_pos.values[is_high], lvdf.Q_pos.values[~is_high], welch=False, name="Q_pos"),
    "positive_only_ncomm_median": float(lvdf.ncomm_pos.median()),
}

# ----------------------------------------------------------------------------
# I. Parcel-count-matched cortical Gini
# ----------------------------------------------------------------------------
absA = np.abs(A)
matched = {}
for k in [8, 60, 64]:
    vals = np.zeros((len(SUBJ), len(TASKS), 200))
    for b in range(200):
        pick = rng.choice(360, size=k, replace=False)
        sub = absA[:, :, pick]
        vals[:, :, b] = np.apply_along_axis(compute_gini_coefficient, 2, sub)
    m = vals.mean(axis=2)  # expected Gini per subject-task at this unit count
    matched[str(k)] = {"mean": float(m.mean()), "sd_subject_task": float(m.std(ddof=1)),
                       "p2.5": float(np.percentile(m, 2.5)), "p97.5": float(np.percentile(m, 97.5))}
# network-aggregated brain Gini: same number of units as a small MoE
netgini = {}
for label, M in [("yeo7", P), ("cole12", M12)]:
    g_ = np.array([[compute_gini_coefficient(M[i, t]) for t in range(7)] for i in range(len(SUBJ))])
    netgini[label] = {"mean": float(g_.mean()), "sd": float(g_.std(ddof=1))}
res["I_matched_gini"] = {"full_360": {"mean": float(sp['gini'].mean())}, "subsampled": matched,
                         "network_aggregated": netgini}

# ----------------------------------------------------------------------------
# J. MoE (corrected extraction)
# ----------------------------------------------------------------------------
moe = {}
layerwise = json.load(open(MOE_SOURCE / "moe_analysis" / "layerwise_routing.json"))["models"]
for key, disp in MOE_MODELS.items():
    m = json.load(open(MOE_SOURCE / "moe_analysis" / f"moe_analysis_{key}.json"))
    E, k = m["n_experts"], m["top_k"]
    ps = pickle.load(open(MOE_SOURCE / "moe_analysis" / "persample" / f"persample_{key}.pkl", "rb"))
    seq_gini, seq_layer_gini = [], []
    for cat in CATS:
        for sample in ps["data"].get(cat, []):
            tot = np.zeros(E)
            for layer, cnt in sample.items():
                v = np.zeros(E)
                for e, c in cnt.items():
                    v[int(e)] += c
                tot += v
                if v.sum() > 0:
                    seq_layer_gini.append(compute_gini_coefficient(v))
            seq_gini.append(compute_gini_coefficient(tot))
    lw = layerwise.get(key) or layerwise.get(disp) or {}
    moe[disp] = {"n_experts": E, "top_k": k,
                 "aggregate_gini": m["sparsity_metrics"]["gini"],
                 "effective_experts": m["sparsity_metrics"]["effective_experts"],
                 "dominant_category_share": m["specialization_summary"]["mean_selectivity"],
                 "specialization_rate": m["specialization_summary"]["specialization_rate"],
                 "token_level_fraction_active": k / E,
                 "per_sequence_gini_pooled_layers": float(np.mean(seq_gini)),
                 "per_sequence_per_layer_gini": float(np.mean(seq_layer_gini)),
                 "n_sequences": len(seq_gini),
                 "total_tokens": m["total_tokens_processed"],
                 "layerwise_summary": {kk: vv for kk, vv in lw.items() if not isinstance(vv, (list, dict))}}
moe_mean = float(np.mean([v["aggregate_gini"] for v in moe.values()]))
res["J_moe"] = {"source": str(MOE_SOURCE), "models": moe, "mean_aggregate_gini": moe_mean,
                "brain_over_moe_ratio": float(sp['gini'].mean() / moe_mean),
                "submitted_values_buggy_extractor": {"Switch-8": 0.244, "Qwen-MoE": 0.070, "DeepSeek-16B": 0.215}}

# ----------------------------------------------------------------------------
# K. Cross-domain structure: Mantel test on 7x7 similarity matrices
# ----------------------------------------------------------------------------
brain_sim = np.corrcoef(A.mean(axis=0))            # task x task over 360 parcels
iu = np.triu_indices(7, 1)
mantel = {}
for key, disp in MOE_MODELS.items():
    m = json.load(open(MOE_SOURCE / "moe_analysis" / f"moe_analysis_{key}.json"))
    E = m["n_experts"]
    prof = {}
    for cat in CATS:
        layers = m["per_category_layer_expert_counts"][cat]
        vecs = []
        for layer in sorted(layers, key=lambda x: int(x)):
            v = np.zeros(E)
            for e, c in layers[layer].items():
                v[int(e)] += c
            if v.sum() > 0:
                vecs.append(v / v.sum())
        prof[CAT2TASK[cat]] = np.concatenate(vecs)
    Mprof = np.stack([prof[t] for t in TASKS])
    moe_sim = np.corrcoef(Mprof)
    obs = stats.spearmanr(brain_sim[iu], moe_sim[iu])[0]
    null = []
    for perm_ in itertools.permutations(range(7)):
        ms = moe_sim[np.ix_(perm_, perm_)]
        null.append(stats.spearmanr(brain_sim[iu], ms[iu])[0])
    null = np.array(null)
    mantel[disp] = {"spearman_r": float(obs), "p_exact_perm": float(np.mean(null >= obs)),
                    "n_perm": len(null)}
res["K_mantel"] = {"description": "Spearman r between upper triangles of the 7x7 brain task-similarity "
                                  "matrix (group-mean 360-parcel maps) and each model's 7x7 category-"
                                  "similarity matrix (layer-concatenated expert-frequency profiles); "
                                  "exact one-sided permutation over all 5040 label orderings.",
                   "models": mantel,
                   "submitted_r_note": "submitted r=+0.25 correlated element-wise the truncated flattened "
                                       "z-scored 7x7 brain task-network matrix and 7x8 Switch category-expert "
                                       "matrix; expert indices are exchangeable, so that r is not interpretable."}

# ----------------------------------------------------------------------------
# Save
# ----------------------------------------------------------------------------
def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o

json.dump(_clean(res), open(OUT / "camera_ready_analyses.json", "w"), indent=2)
lvdf.assign(subject=SUBJ).to_csv(OUT / "louvain_seed_stability.csv", index=False)
pd.DataFrame({"subject": SUBJ, "high": is_high, "eff": eff, "eff_LR": e_lr, "eff_RL": e_rl,
              "relRMS": subj_fd, "loo_r_group": np.r_[loo_hi, loo_lo][np.argsort(hi_idx + lo_idx)],
              "composite_sparsity": sp["composite"].mean(axis=1), "sel7": sel7_subj,
              "sel12": sel12_subj}).to_csv(OUT / "subject_level_table.csv", index=False)
print(json.dumps(_clean({k: res[k] for k in ["A_sample", "B_efficiency_reliability", "FDR"]}), indent=1))
print("done")
