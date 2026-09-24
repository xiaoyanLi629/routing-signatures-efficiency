#!/usr/bin/env python3
"""
=============================================================================
Stage 3b: H3 (Inter-Subject Routing Consistency) under Independence-Safe Tests
=============================================================================

The original s03 tests H3 by flattening all within-group pairwise Pearson
correlations into a long vector and running an independent-samples t-test.
Those pairs are NOT independent (each subject contributes 49 pairs), so the
effective sample size is inflated and the nominal p underestimates the true p.

This stage re-tests H3 with two statistically sound alternatives:

  (A) Subject-level LOO-r: for each subject, compute the correlation between
      their routing profile and the leave-one-out mean of their group. This
      yields n_high + n_low independent scalars; a standard t-test is valid.

  (B) Label-permutation test: hold the subject->profile mapping fixed, shuffle
      the efficiency labels, recompute the naive mean within-group r difference.
      The p-value is the fraction of shuffles whose absolute mean-difference
      equals or exceeds the observed.

Reads:
    neuroscience/routing_profiles.pkl
    neuroscience/efficiency_groups.json

Writes:
    neuroscience/h3_robust.json
"""

import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import json
import pickle
import numpy as np
from scipy import stats

from configs import config

config.ensure_run_directories()
logger = config.setup_logging('h3_robust')

RNG_SEED = 0
N_PERM = 10000
N_BOOT = 10000


def loo_group_mean_r(profiles, subjects):
    """For each subject in `subjects`, return r between their flattened
    profile and the mean of the other (n-1) group members' profiles."""
    profs = np.stack([profiles[s].flatten() for s in subjects], axis=0)  # (n, d)
    n, d = profs.shape
    # leave-one-out mean: (sum - self) / (n - 1)
    total = profs.sum(axis=0, keepdims=True)
    loo_mean = (total - profs) / (n - 1)
    rs = np.zeros(n)
    for i in range(n):
        r, _ = stats.pearsonr(profs[i], loo_mean[i])
        rs[i] = r if np.isfinite(r) else np.nan
    return rs


def cohens_d(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    va = np.var(a, ddof=1); vb = np.var(b, ddof=1)
    pooled = np.sqrt((va + vb) / 2.0)
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def bootstrap_mean_diff_ci(a, b, n_boot=N_BOOT, seed=RNG_SEED, alpha=0.05):
    rng = np.random.default_rng(seed)
    a = np.asarray(a, float); b = np.asarray(b, float)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        diffs[i] = rng.choice(a, a.size, replace=True).mean() \
                 - rng.choice(b, b.size, replace=True).mean()
    lo, hi = np.quantile(diffs, [alpha/2, 1-alpha/2])
    return float(lo), float(hi)


def _row_normalize(X):
    """Z-score rows so that dot(X[i], X[j]) / d = pearson r of raw rows."""
    X = X - X.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def mean_pairwise_r_from_indices(Xn, idx):
    """Mean pairwise Pearson r among rows Xn[idx]. Xn is row-normalized."""
    sub = Xn[idx]
    # Sum of off-diagonal of sub @ sub.T, divided by number of off-diag pairs
    gram = sub @ sub.T
    n = gram.shape[0]
    if n < 2:
        return 0.0
    total = (gram.sum() - np.trace(gram)) / 2.0
    return float(total / (n * (n - 1) / 2.0))


def permutation_test_vectorized(profiles, high, low, n_perm=N_PERM, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    all_subs = list(high) + list(low)
    n_total = len(all_subs)
    n_high = len(high)
    # Build (n_total, d) matrix, row-normalized for fast r via dot product
    X = np.stack([profiles[s].flatten() for s in all_subs], axis=0).astype(np.float64)
    Xn = _row_normalize(X)
    full_gram = Xn @ Xn.T  # (n_total, n_total) r-matrix
    np.fill_diagonal(full_gram, 0.0)  # we never want self-pairs

    def mean_r_indices(idx):
        if len(idx) < 2:
            return 0.0
        sub = full_gram[np.ix_(idx, idx)]
        return float(sub.sum() / 2.0 / (len(idx) * (len(idx) - 1) / 2.0))

    obs = mean_r_indices(np.arange(n_high)) - mean_r_indices(np.arange(n_high, n_total))
    null = np.empty(n_perm)
    for k in range(n_perm):
        perm = rng.permutation(n_total)
        null[k] = mean_r_indices(perm[:n_high]) - mean_r_indices(perm[n_high:])
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)
    return float(obs), float(p), null, full_gram


def main():
    logger.info("=" * 60)
    logger.info("Stage 3b: H3 robust re-test (LOO-r + label permutation)")
    logger.info("=" * 60)

    nd = config.NEUROSCIENCE_DIR
    with open(nd / "routing_profiles.pkl", "rb") as f:
        profiles = pickle.load(f)
    with open(nd / "efficiency_groups.json") as f:
        groups_data = json.load(f)
    high = [s for s in groups_data['groups']['high'] if s in profiles]
    low  = [s for s in groups_data['groups']['low']  if s in profiles]
    logger.info(f"n_high={len(high)}  n_low={len(low)}")

    # (A) Subject-level LOO-r
    r_high = loo_group_mean_r(profiles, high)
    r_low  = loo_group_mean_r(profiles, low)
    # Drop any NaN
    r_high = r_high[np.isfinite(r_high)]
    r_low  = r_low[np.isfinite(r_low)]
    t_loo, p_loo = stats.ttest_ind(r_high, r_low, equal_var=False)  # Welch
    u_loo, pu_loo = stats.mannwhitneyu(r_high, r_low, alternative='two-sided')
    d_loo = cohens_d(r_high, r_low)
    ci_lo, ci_hi = bootstrap_mean_diff_ci(r_high, r_low)

    logger.info("--- (A) Subject-level LOO-r (independent observations) ---")
    logger.info(f"  high mean={r_high.mean():.4f}  low mean={r_low.mean():.4f}")
    logger.info(f"  mean diff={r_high.mean() - r_low.mean():+.4f}  95%% CI=[{ci_lo:+.4f}, {ci_hi:+.4f}]")
    logger.info(f"  Welch t={t_loo:.3f}  p={p_loo:.4f}  Cohen d={d_loo:+.3f}")
    logger.info(f"  Mann-Whitney U={u_loo:.1f}  p={pu_loo:.4f}")

    # (B) Label permutation (preserves dependence structure) - vectorized
    obs_diff, p_perm, null, full_gram = permutation_test_vectorized(profiles, high, low)
    logger.info("--- (B) Label-permutation on mean pairwise r (n_perm=%d, vectorized) ---", N_PERM)
    logger.info(f"  observed diff={obs_diff:+.4f}  null mean={null.mean():+.5f}  null std={null.std():.5f}")
    logger.info(f"  two-sided p={p_perm:.4f}")

    # Reference: the original naive test (recovered from the same Gram matrix)
    n_high = len(high)
    high_pairs = full_gram[:n_high, :n_high][np.triu_indices(n_high, k=1)]
    low_pairs  = full_gram[n_high:, n_high:][np.triu_indices(len(low), k=1)]
    t_naive, p_naive = stats.ttest_ind(high_pairs, low_pairs)
    d_naive = cohens_d(high_pairs, low_pairs)
    logger.info("--- Reference: original pairwise-r naive t-test (inflated N) ---")
    logger.info(f"  naive t={t_naive:.3f}  naive p={p_naive:.4g}  d={d_naive:+.3f}  (n_pairs_high={len(high_pairs)}, n_pairs_low={len(low_pairs)})")

    out = {
        'n_high': int(len(high)),
        'n_low':  int(len(low)),
        'loo_r_test': {
            'description': 'Subject-level leave-one-out r vs group-mean, Welch t-test on independent observations',
            'high_mean_r': float(r_high.mean()),
            'low_mean_r':  float(r_low.mean()),
            'mean_diff':   float(r_high.mean() - r_low.mean()),
            'ci95_bootstrap': [ci_lo, ci_hi],
            'welch_t': float(t_loo),
            'welch_p': float(p_loo),
            'mannwhitney_u': float(u_loo),
            'mannwhitney_p': float(pu_loo),
            'cohens_d': float(d_loo),
        },
        'label_permutation_test': {
            'description': 'Two-sided permutation on efficiency labels, statistic = within-group mean pairwise r difference',
            'observed_diff': float(obs_diff),
            'n_perm': int(N_PERM),
            'p_value': float(p_perm),
            'null_mean': float(null.mean()),
            'null_std':  float(null.std()),
        },
        'naive_reference': {
            'description': 'Original s03 pairwise-r t-test (treats ~2450 pairs as independent; reported for comparison only)',
            't': float(t_naive),
            'p': float(p_naive),
            'cohens_d': float(d_naive),
            'n_pairs_high': int(len(high_pairs)),
            'n_pairs_low':  int(len(low_pairs)),
        },
        'notes': (
            'LOO-r and label permutation both respect subject-level independence. '
            'If LOO-r p and permutation p both < 0.05, H3 is supported under a '
            'statistically valid framework. If either exceeds 0.05, the original '
            'H3 finding (t=2.78, p=0.0055) is an artifact of inflated N from '
            'non-independent pairwise observations.'
        ),
    }
    out_path = nd / "h3_robust.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"Saved: {out_path}")

    # Verdict
    sig_loo = p_loo < 0.05
    sig_perm = p_perm < 0.05
    verdict = ("SUPPORTED (both tests)" if (sig_loo and sig_perm)
               else "SUPPORTED (permutation only)" if sig_perm
               else "SUPPORTED (LOO-r only)" if sig_loo
               else "NOT SUPPORTED under independence-safe tests")
    logger.info("=" * 60)
    logger.info(f"H3 verdict: {verdict}")
    logger.info("=" * 60)
    return out


if __name__ == "__main__":
    main()
