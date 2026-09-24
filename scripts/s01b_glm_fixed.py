#!/usr/bin/env python3
"""
=============================================================================
Stage 1b (camera-ready): task activation with a corrected GLM
=============================================================================
Stage 1 (s01) has three defects found during camera-ready checks:

  1. load_task_evs() reads EVERY *.txt in the EVs folder, so the GLM also
     contains Sync.txt and the event-level regressors HCP ships next to the
     block regressors (WM: 0bk_cor, 2bk_err, all_bk_cor, ...; LANGUAGE:
     present_story, question_math, ...; GAMBLING: win_event, ...; SOCIAL:
     mental_resp, ...; RELATIONAL: error). These overlap the blocks in time,
     and WM's all_bk_cor equals 0bk_cor + 2bk_cor, so the design is
     rank-deficient and lstsq's minimum-norm betas split the block effect
     arbitrarily. On subject 100206 this flips the sign of frontoparietal
     2-back > 0-back activation.
  2. EV names are matched by substring, so 'story' also selects
     present_story / question_story / response_story, 'win' selects
     win_event, and the "contrast" averages block and event betas.
  3. _create_hrf() divides the undershoot gamma by 1e12 instead of
     6 * 15! (~7.8e12), giving an undershoot almost as large as the peak.
  4. MOTOR used movement vs. cue. The 3-s cue always immediately precedes a
     movement block, the two regressors are highly correlated, and the
     resulting map barely engages somatomotor cortex.

This stage re-extracts the activation array with, for every task, only the
HCP block EVs in the design (exact file-name match), the SPM canonical
double-gamma HRF, a DCT high-pass drift basis (128 s), and the HCP standard
contrasts (MOTOR: mean of the five movement blocks vs. implicit fixation
baseline). Everything else (parcellation, LR/RL averaging, z-scoring across
parcels) is unchanged. Output goes to a NEW run directory; the submitted run
is left untouched.

Usage:
  PROJECT2_RUN_TIMESTAMP=20260924_glmfix python scripts/s01b_glm_fixed.py
=============================================================================
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import json
import pickle
from pathlib import Path
from multiprocessing import Pool

import numpy as np
from scipy import stats
from scipy.stats import gamma

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
from configs import config  # noqa: E402
from scripts.s01_multi_task_extraction import (  # noqa: E402
    load_cifti_data, extract_parcel_timeseries)

config.ensure_run_directories()
TR = 0.72

# Block EVs entering the design (exact stems) and the contrast, per task.
DESIGN = {
    'WM': {'evs': ['2bk_body', '2bk_faces', '2bk_places', '2bk_tools',
                   '0bk_body', '0bk_faces', '0bk_places', '0bk_tools'],
           'pos': ['2bk_body', '2bk_faces', '2bk_places', '2bk_tools'],
           'neg': ['0bk_body', '0bk_faces', '0bk_places', '0bk_tools']},
    'MOTOR': {'evs': ['cue', 'lf', 'lh', 'rf', 'rh', 't'],
              'pos': ['lf', 'lh', 'rf', 'rh', 't'], 'neg': []},
    'LANGUAGE': {'evs': ['story', 'math'], 'pos': ['story'], 'neg': ['math']},
    'SOCIAL': {'evs': ['mental', 'rnd'], 'pos': ['mental'], 'neg': ['rnd']},
    'RELATIONAL': {'evs': ['relation', 'match'], 'pos': ['relation'], 'neg': ['match']},
    'EMOTION': {'evs': ['fear', 'neut'], 'pos': ['fear'], 'neg': ['neut']},
    'GAMBLING': {'evs': ['win', 'loss'], 'pos': ['win'], 'neg': ['loss']},
}


def spm_hrf(tr, duration=32.0):
    """SPM canonical double-gamma HRF sampled at the TR."""
    t = np.arange(0, duration, tr)
    h = gamma.pdf(t, 6) - gamma.pdf(t, 16) / 6.0
    return h / h.sum()


def dct_basis(n, tr, cutoff=128.0):
    """Discrete cosine drift regressors below 1/cutoff Hz (SPM high-pass)."""
    k = int(np.floor(2 * n * tr / cutoff)) + 1
    t = np.arange(n)
    return np.column_stack([np.cos(np.pi * j * (2 * t + 1) / (2 * n)) for j in range(1, k)]) if k > 1 else np.zeros((n, 0))


def block_regressor(ev, n, tr, hrf, dt=0.1):
    """Boxcar at 0.1-s resolution, convolved, then sampled at the TR."""
    ev = np.atleast_2d(ev)
    t_hi = np.arange(0, n * tr, dt)
    box = np.zeros_like(t_hi)
    for onset, dur, amp in ev[:, :3]:
        box[(t_hi >= onset) & (t_hi < onset + dur)] = amp
    h_hi = spm_hrf(dt)
    conv = np.convolve(box, h_hi)[:len(t_hi)]
    idx = np.clip(np.round(np.arange(n) * tr / dt).astype(int), 0, len(t_hi) - 1)
    return conv[idx]


def run_contrast(ts, evs_dir, task):
    d = DESIGN[task]
    n = ts.shape[0]
    hrf = spm_hrf(TR)
    cols, names = [], []
    for name in d['evs']:
        f = Path(evs_dir) / f"{name}.txt"
        if not f.exists():
            continue
        ev = np.loadtxt(f)
        if ev.size < 3:
            continue
        r = block_regressor(ev, n, TR, hrf)
        if r.std() < 1e-8:
            continue
        cols.append(r)
        names.append(name)
    if not any(p in names for p in d['pos']):
        return None, None
    X = np.column_stack(cols + [dct_basis(n, TR), np.ones(n)])
    beta, *_ = np.linalg.lstsq(X, ts, rcond=None)
    b = dict(zip(names, beta[:len(names)]))
    pos = np.mean([b[p] for p in d['pos'] if p in b], axis=0)
    neg = [b[q] for q in d['neg'] if q in b]
    con = pos - np.mean(neg, axis=0) if neg else pos
    return con, float(np.linalg.cond(X[:, :len(names)]))


def one_subject(subject):
    out = np.full((len(config.TASK_ORDER), 360), np.nan)
    conds = {}
    for j, task in enumerate(config.TASK_ORDER):
        runs = []
        for run in ['LR', 'RL']:
            fpath = config.get_task_fmri_path(subject, task, run)
            if not fpath.exists():
                continue
            ts = extract_parcel_timeseries(load_cifti_data(fpath), 360)
            if ts is None:
                continue
            con, cond = run_contrast(ts, config.get_task_evs_path(subject, task, run), task)
            if con is not None:
                runs.append(con)
                conds[f"{task}_{run}"] = cond
        if runs:
            out[j] = stats.zscore(np.mean(runs, axis=0))
    return subject, out, conds


if __name__ == "__main__":
    subjects = config.SUBJECTS
    with Pool(8) as pool:
        res = pool.map(one_subject, subjects, chunksize=1)
    res = {s: (a, c) for s, a, c in res}
    A = np.stack([res[s][0] for s in subjects])
    with open(config.NEUROSCIENCE_DIR / "activation_matrix.pkl", "wb") as f:
        pickle.dump({'data': A, 'subjects': subjects, 'tasks': config.TASK_ORDER,
                     'n_parcels': 360}, f)
    stats_out = {
        'glm': 'block EVs only, exact names, SPM double-gamma HRF, DCT 128 s high-pass',
        'design': DESIGN,
        'n_nan_cells': int(np.isnan(A[:, :, 0]).sum()),
        'successful_extractions': int(np.isfinite(A[:, :, 0]).sum()),
        'design_condition_number_median': float(np.median([v for s in subjects for v in res[s][1].values()])),
        'design_condition_number_max': float(np.max([v for s in subjects for v in res[s][1].values()])),
    }
    json.dump(stats_out, open(config.NEUROSCIENCE_DIR / "extraction_stats.json", "w"), indent=2)
    print(json.dumps(stats_out, indent=1, default=str))
    print("done")
