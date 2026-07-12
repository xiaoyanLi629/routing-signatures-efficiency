#!/usr/bin/env python3
"""
=============================================================================
Stage 6b: Global FDR Correction Across Primary Brain-Analysis Hypotheses
=============================================================================

Applies Benjamini-Hochberg FDR correction across the primary statistical tests
reported in the manuscript (H1, H3, H4). This is a *cross-stage* correction —
each individual stage (s02, s04) already applies FDR within its own family of
post-hoc tests (per-task sparsity, modularity metrics); this stage corrects the
headline claims that go into the Abstract / Table IV.

Inputs (written by s02, s03, s04):
    neuroscience/sparsity_comparison.json   (H1)
    neuroscience/routing_analysis.json      (H3)
    neuroscience/modularity_analysis.json   (H4)

Output:
    neuroscience/hypothesis_fdr_summary.json
"""

import sys
import json
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from configs import config

config.ensure_run_directories()
logger = config.setup_logging('hypothesis_fdr')


def _load(p):
    with open(p) as f:
        return json.load(f)


def main():
    nd = config.NEUROSCIENCE_DIR
    sparsity = _load(nd / "sparsity_comparison.json")
    modularity = _load(nd / "modularity_analysis.json")

    # H3 uses the INDEPENDENCE-SAFE subject-level LOO-r test (s03b) as its
    # primary statistic. The original pairwise-r t-test in s03 treats
    # ~2450 non-independent pairs as independent and is reported only for
    # reference. If h3_robust.json is not present, this stage will fail
    # rather than silently fall back to the inflated-N test.
    h3_path = nd / "h3_robust.json"
    if not h3_path.exists():
        raise FileNotFoundError(
            f"{h3_path} not found; run scripts/s03b_h3_robust.py first so that "
            "the primary H3 test is the independence-safe LOO-r Welch t."
        )
    h3_robust = _load(h3_path)
    h3_loo = h3_robust['loo_r_test']
    h3_perm = h3_robust['label_permutation_test']

    primary = {
        'H1_sparse_activation': {
            'p_value': sparsity['overall']['composite_sparsity']['p_value'],
            'cohens_d': sparsity['overall']['composite_sparsity']['cohens_d'],
            'test': 'High-efficiency group has higher composite sparsity',
        },
        'H3_consistent_routing': {
            # Subject-level LOO-r Welch t-test: each subject contributes one
            # independent r (profile vs. leave-one-out group mean).
            'p_value': h3_loo['welch_p'],
            'cohens_d': h3_loo['cohens_d'],
            'test': ('High-efficiency group has higher inter-subject routing '
                     'consistency (subject-level LOO-r Welch t)'),
            'auxiliary': {
                'label_permutation_p': h3_perm['p_value'],
                'naive_pairwise_p_for_reference_only': h3_robust['naive_reference']['p'],
            },
        },
        'H4_modularity': {
            'p_value': modularity['comparison']['modularity_Q']['p_value'],
            'cohens_d': modularity['comparison']['modularity_Q']['cohens_d'],
            'test': 'Efficiency group difference in network modularity (Q)',
        },
    }

    # Apply FDR-BH across the 3 primary hypotheses.
    config.apply_fdr_correction(primary, p_value_key='p_value')

    logger.info("Global FDR correction across primary brain-analysis hypotheses:")
    for name, entry in primary.items():
        logger.info(
            f"  {name}: d={entry['cohens_d']:+.3f} "
            f"p_raw={entry['p_value']:.4f} p_fdr={entry['p_value_fdr']:.4f} "
            f"sig_fdr={entry['significant_fdr']}"
        )

    out = nd / "hypothesis_fdr_summary.json"
    with open(out, 'w') as f:
        json.dump({
            'primary_hypotheses': primary,
            'correction_method': 'fdr_bh',
            'alpha': config.STATS_PARAMS['alpha'],
            'family_size': len(primary),
            'description': (
                'Benjamini-Hochberg FDR correction across the three primary '
                'statistical claims reported in the manuscript (H1, H3, H4). '
                'Per-task and per-metric sub-tests are FDR-corrected within '
                'their own stage (s02, s04).'
            ),
        }, f, indent=2)
    logger.info(f"Saved: {out}")

    return primary


if __name__ == "__main__":
    main()
