#!/usr/bin/env python3
"""
Update BIBM2026_paper.tex placeholders with values from a completed run.

Reads:
    <run-dir>/neuroscience/hypothesis_fdr_summary.json   (H1/H3/H4 FDR)
    <run-dir>/neuroscience/sparsity_comparison.json      (Gini table vals)
    <run-dir>/neuroscience/modularity_analysis.json      (seg, part p-values)
    <run-dir>/neuroscience/routing_analysis.json         (selectivity, r values)
    <run-dir>/moe_analysis/multi_moe_summary.csv         (for sparsity ratio)
    <run-dir>/neuroscience/extraction_stats.json         (N subjects)

Replaces {{placeholder}} tokens in the .tex file. Writes backup to .tex.bak.
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path


def _load_json(p):
    if not p.exists():
        print(f"[WARN] missing: {p}", file=sys.stderr)
        return None
    with open(p) as f:
        return json.load(f)


def fmt(x, prec=3):
    """Format a float for inline tex."""
    if x is None:
        return "{{?}}"
    try:
        return f"{float(x):.{prec}f}"
    except (ValueError, TypeError):
        return str(x)


def fmt_p(p):
    """Format a p-value with standard conventions."""
    if p is None:
        return "{{?}}"
    p = float(p)
    if p < 0.001:
        return "< .001"
    return f"{p:.3f}"


def collect_substitutions(run_dir: Path):
    """Return dict of placeholder -> replacement string."""
    ns = run_dir / "neuroscience"

    fdr = _load_json(ns / "hypothesis_fdr_summary.json")
    modularity = _load_json(ns / "modularity_analysis.json")
    extraction = _load_json(ns / "extraction_stats.json")
    moe_summary_path = run_dir / "moe_analysis" / "multi_moe_summary.csv"

    subs = {}

    # --- N subjects -------------------------------------------------------
    if extraction:
        n = extraction.get('n_subjects')
        if n is None and 'subjects' in extraction:
            n = len(extraction['subjects'])
        if n is not None:
            subs['N'] = str(n)

    # --- H1/H3/H4 primary FDR values --------------------------------------
    if fdr:
        ph = fdr.get('primary_hypotheses', {})
        for key_src, prefix in [
            ('H1_sparse_activation', 'H1'),
            ('H3_consistent_routing', 'H3'),
            ('H4_modularity', 'H4'),
        ]:
            entry = ph.get(key_src, {})
            if 'cohens_d' in entry:
                subs[f'd{prefix}'] = f"{float(entry['cohens_d']):+.2f}"
            if 'p_value_fdr' in entry:
                subs[f'p{prefix}fdr'] = fmt_p(entry['p_value_fdr'])

    # --- Modularity sub-metrics p-values (FDR-corrected in-stage) ---------
    if modularity and 'comparison' in modularity:
        comp = modularity['comparison']
        if 'segregation' in comp:
            subs['pSegFdr'] = fmt_p(
                comp['segregation'].get('p_value_fdr',
                                        comp['segregation'].get('p_value'))
            )
        if 'mean_participation' in comp:
            subs['pPartFdr'] = fmt_p(
                comp['mean_participation'].get('p_value_fdr',
                                                comp['mean_participation'].get('p_value'))
            )

    # --- Brain vs MoE sparsity ratio --------------------------------------
    if moe_summary_path.exists():
        try:
            import csv
            with open(moe_summary_path) as f:
                rows = list(csv.DictReader(f))
            moe_ginis = [float(r['gini']) for r in rows
                         if r.get('gini') not in (None, '')]
            if moe_ginis:
                moe_mean = sum(moe_ginis) / len(moe_ginis)
                # Brain mean Gini from sparsity_summary.json
                spar_sum = _load_json(ns / "sparsity_summary.json") or {}
                brain_gini = spar_sum.get('mean_gini')
                if brain_gini and moe_mean > 0:
                    subs['sparsityRatio'] = f"{brain_gini / moe_mean:.1f}"
        except Exception as e:
            print(f"[WARN] could not compute sparsity ratio: {e}",
                  file=sys.stderr)

    return subs


def apply_substitutions(tex_path: Path, subs: dict, backup=True):
    text = tex_path.read_text()
    if backup:
        shutil.copy(tex_path, tex_path.with_suffix(tex_path.suffix + ".bak"))

    n_replaced = {}
    for key, val in subs.items():
        pattern = r'\{\{' + re.escape(key) + r'\}\}'
        new_text, n = re.subn(pattern, val, text)
        if n > 0:
            n_replaced[key] = n
        text = new_text

    tex_path.write_text(text)

    # Warn about remaining placeholders
    remaining = set(re.findall(r'\{\{(\w+)\}\}', text))
    return n_replaced, remaining


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True,
                        help='Path to results/run_YYYYMMDD_HHMMSS/')
    parser.add_argument('--tex', required=True,
                        help='Path to BIBM2026_paper.tex')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print substitutions without editing')
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    tex_path = Path(args.tex)

    subs = collect_substitutions(run_dir)

    print("Substitutions collected:")
    for k, v in sorted(subs.items()):
        print(f"  {{{{{k}}}}} -> {v}")

    if args.dry_run:
        print("\n(dry-run, no edits made)")
        return

    n_replaced, remaining = apply_substitutions(tex_path, subs)

    print(f"\nEdited: {tex_path}")
    print(f"Replacements applied: {sum(n_replaced.values())} "
          f"across {len(n_replaced)} placeholder names")
    for k, n in sorted(n_replaced.items()):
        print(f"  {{{{{k}}}}}: {n}x")

    if remaining:
        print(f"\n[WARN] Unfilled placeholders remain in tex: {sorted(remaining)}")
        print("  These need manual filling from result JSONs or Tables.")


if __name__ == "__main__":
    main()
