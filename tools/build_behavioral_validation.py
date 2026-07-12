#!/usr/bin/env python3
"""
Build behavioral_validation.csv from per-subject WM_Stats.csv files.

Source files: data/<subj>/MNINonLinear/Results/tfMRI_WM_{LR,RL}/EVs/WM_Stats.csv

Each WM_Stats.csv contains rows like:
    Value,ConditionName,Measure
    0.9,0BK_BODY,ACC
    674.0,0BK_BODY,MEDIAN_RT
    ...

We extract accuracy and median reaction time for the 0-back and 2-back
conditions, average across LR+RL runs, and compute an inverse-efficiency score:
    efficiency = accuracy_overall / (rt_overall / rt_overall_mean_across_subjects)

Writes a CSV with one row per subject and the columns expected by
`compute_efficiency_scores()` in scripts/s02_sparsity_analysis.py.
"""

import argparse
import csv
import json
from pathlib import Path
import statistics


def _load_wm_stats(path: Path):
    """Parse a WM_Stats.csv into {condition: {measure: value}}."""
    out = {}
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            if len(row) < 3:
                continue
            val, cond, meas = row[0], row[1], row[2]
            try:
                val = float(val)
            except ValueError:
                continue
            out.setdefault(cond, {})[meas] = val
    return out


def _condition_summary(stats):
    """Aggregate 4x0BK and 4x2BK conditions into run-level means."""
    conds_0bk = [c for c in stats if c.startswith('0BK_')]
    conds_2bk = [c for c in stats if c.startswith('2BK_')]

    def agg(conds, measure):
        vals = [stats[c][measure] for c in conds if measure in stats[c]]
        return statistics.mean(vals) if vals else float('nan')

    return {
        'acc_0bk': agg(conds_0bk, 'ACC'),
        'acc_2bk': agg(conds_2bk, 'ACC'),
        'rt_0bk':  agg(conds_0bk, 'MEDIAN_RT'),
        'rt_2bk':  agg(conds_2bk, 'MEDIAN_RT'),
    }


def _subject_row(subj, data_root):
    """Average WM_Stats over LR and RL runs. Returns dict of fields or None."""
    per_run = []
    for run in ('LR', 'RL'):
        p = data_root / subj / 'MNINonLinear' / 'Results' / f'tfMRI_WM_{run}' / 'EVs' / 'WM_Stats.csv'
        if not p.exists():
            continue
        stats = _load_wm_stats(p)
        summary = _condition_summary(stats)
        if all(v == v for v in summary.values()):  # no NaN
            per_run.append(summary)

    if not per_run:
        return None

    def mean_key(key):
        return statistics.mean(r[key] for r in per_run)

    acc_overall = (mean_key('acc_0bk') + mean_key('acc_2bk')) / 2
    rt_overall = (mean_key('rt_0bk') + mean_key('rt_2bk')) / 2

    return {
        'subject': subj,
        'acc_2bk': round(mean_key('acc_2bk'), 4),
        'acc_0bk': round(mean_key('acc_0bk'), 4),
        'acc_overall': round(acc_overall, 4),
        'rt_2bk': round(mean_key('rt_2bk'), 2),
        'rt_0bk': round(mean_key('rt_0bk'), 2),
        'rt_overall': round(rt_overall, 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    subjects = sorted(d.name for d in data_root.iterdir()
                      if d.is_dir() and d.name.isdigit())

    rows = []
    for s in subjects:
        r = _subject_row(s, data_root)
        if r is not None:
            rows.append(r)
        else:
            print(f"  [skip] {s}: no usable WM_Stats.csv")

    if not rows:
        print("No subjects had usable behavioral data. Aborting.")
        return

    # Compute efficiency score: accuracy / normalized RT
    rt_mean = statistics.mean(r['rt_overall'] for r in rows)
    for r in rows:
        r['efficiency_score'] = round(r['acc_overall'] / (r['rt_overall'] / rt_mean), 6)

    # Median-split labelling
    eff_sorted = sorted(r['efficiency_score'] for r in rows)
    median_eff = eff_sorted[len(eff_sorted) // 2]
    for r in rows:
        r['group'] = 'high' if r['efficiency_score'] >= median_eff else 'low'

    # Write CSV in same column order as the existing behavioral_validation.csv
    fieldnames = ['subject', 'group', 'efficiency_score',
                  'acc_2bk', 'acc_0bk', 'acc_overall',
                  'rt_2bk', 'rt_0bk', 'rt_overall']
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fieldnames})

    print(f"\nWrote {len(rows)} rows to {out}")
    print(f"  high-efficiency: {sum(1 for r in rows if r['group']=='high')}")
    print(f"  low-efficiency:  {sum(1 for r in rows if r['group']=='low')}")
    print(f"  mean efficiency: {statistics.mean(r['efficiency_score'] for r in rows):.4f}")
    print(f"  mean accuracy:   {statistics.mean(r['acc_overall'] for r in rows):.4f}")
    print(f"  mean RT (ms):    {statistics.mean(r['rt_overall'] for r in rows):.1f}")


if __name__ == '__main__':
    main()
