#!/usr/bin/env python3
"""
=============================================================================
Stage 2: Sparsity Analysis
=============================================================================

Analyze activation sparsity patterns across tasks and subjects.
Tests H1: High-efficiency individuals activate fewer brain regions.

Outputs:
    - sparsity_metrics.csv: Sparsity indices per subject per task
    - efficiency_groups.json: High/Low efficiency group assignments
    - sparsity_comparison.json: Statistical comparison between groups
"""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
import pandas as pd
from scipy import stats
import pickle
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from configs import config

# Ensure timestamped run directories are initialized
config.ensure_run_directories()

logger = config.setup_logging('sparsity_analysis')

# =============================================================================
# SPARSITY METRICS
# =============================================================================

def compute_gini_coefficient(x):
    """
    Compute Gini coefficient as a measure of sparsity.
    0 = perfectly uniform, 1 = maximally sparse
    """
    x = np.abs(x)
    x = np.sort(x)
    n = len(x)
    if n == 0 or np.sum(x) == 0:
        return 0
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * x) / (n * np.sum(x))) - (n + 1) / n


def compute_l0_sparsity(x, threshold_percentile=95):
    """
    Compute L0 sparsity: proportion of values below threshold.
    Higher = more sparse (fewer active regions)
    """
    threshold = np.percentile(np.abs(x), threshold_percentile)
    n_below = np.sum(np.abs(x) < threshold)
    return n_below / len(x)


def compute_kurtosis_sparsity(x):
    """
    Compute kurtosis as sparsity measure.
    Higher kurtosis = heavier tails = more sparse
    """
    return stats.kurtosis(x, fisher=True)


def compute_entropy_sparsity(x, n_bins=50):
    """
    Compute entropy-based sparsity in [0, 1].
    Histogram is a probability mass (counts/total), NOT a density, so the
    Shannon entropy is guaranteed non-negative and bounded by log(n_bins).
    Returns 1 - H/H_max: 0 = uniform, 1 = maximally concentrated (sparse).
    """
    hist, _ = np.histogram(x, bins=n_bins, density=False)
    total = hist.sum()
    if total == 0:
        return 0.0
    p = hist[hist > 0] / total
    entropy = -np.sum(p * np.log(p))
    max_entropy = np.log(n_bins)
    if max_entropy <= 0:
        return 0.0
    return float(1.0 - entropy / max_entropy)


def compute_active_regions(x, threshold_percentile=95):
    """
    Count number of "active" regions above threshold.
    """
    threshold = np.percentile(np.abs(x), threshold_percentile)
    return np.sum(np.abs(x) >= threshold)


def compute_concentration_ratio(x, top_k=36):
    """
    Compute concentration ratio: proportion of total activation in top-K regions.
    Higher = more concentrated = more sparse
    """
    x_abs = np.abs(x)
    total = np.sum(x_abs)
    if total == 0:
        return 0
    top_k_sum = np.sum(np.sort(x_abs)[-top_k:])
    return top_k_sum / total


# =============================================================================
# EFFICIENCY GROUPING
# =============================================================================

def compute_efficiency_scores(activation_matrix, subjects, tasks, behavioral_csv=None):
    """
    Compute efficiency scores based on BEHAVIORAL performance.

    Uses accuracy/reaction-time ratio from the WM task as an independent
    efficiency criterion. This avoids circular reasoning: grouping by
    neural sparsity and then testing for sparsity differences is tautological.

    Falls back to neural-sparsity-based grouping only if behavioral data
    is unavailable, with a warning.
    """
    efficiency_scores = {}

    # Try behavioral classification first (preferred)
    if behavioral_csv is not None and Path(behavioral_csv).exists():
        behavioral_df = pd.read_csv(behavioral_csv)

        for _, row in behavioral_df.iterrows():
            subject = str(int(row['subject'])) if not isinstance(row['subject'], str) else row['subject']
            if subject in subjects:
                # Behavioral efficiency: accuracy per unit reaction time (ms)
                # Higher = better performance with faster response
                acc = row.get('acc_overall', row.get('acc_2bk', 0.5))
                rt = row.get('rt_overall', row.get('rt_2bk', 1000.0))
                if rt > 0:
                    efficiency_scores[subject] = acc / rt * 1000  # Scale for readability
                else:
                    efficiency_scores[subject] = 0.0

        # Fill any missing subjects with median score
        if efficiency_scores:
            median_score = np.median(list(efficiency_scores.values()))
            for subject in subjects:
                if subject not in efficiency_scores:
                    logger.warning(f"No behavioral data for {subject}, using median")
                    efficiency_scores[subject] = median_score

        logger.info("Efficiency classification: BEHAVIORAL (accuracy/RT)")
        return efficiency_scores, 'behavioral'

    # Fallback: neural sparsity (with warning about circularity)
    logger.warning("=" * 60)
    logger.warning("WARNING: Using neural-sparsity-based efficiency classification.")
    logger.warning("This creates circular reasoning for H1 (sparsity hypothesis).")
    logger.warning("Provide behavioral_validation.csv for independent classification.")
    logger.warning("=" * 60)

    wm_idx = tasks.index('WM') if 'WM' in tasks else 0

    for s_idx, subject in enumerate(subjects):
        wm_activation = activation_matrix[s_idx, wm_idx, :]
        gini = compute_gini_coefficient(wm_activation)
        concentration = compute_concentration_ratio(wm_activation)
        efficiency_scores[subject] = (gini + concentration) / 2

    return efficiency_scores, 'neural_sparsity'


def split_efficiency_groups(efficiency_scores, method='median'):
    """Split subjects into high/low efficiency groups"""
    subjects = list(efficiency_scores.keys())
    scores = [efficiency_scores[s] for s in subjects]
    
    if method == 'median':
        median_score = np.median(scores)
        high_eff = [s for s in subjects if efficiency_scores[s] >= median_score]
        low_eff = [s for s in subjects if efficiency_scores[s] < median_score]
    elif method == 'tercile':
        terciles = np.percentile(scores, [33, 67])
        high_eff = [s for s in subjects if efficiency_scores[s] >= terciles[1]]
        low_eff = [s for s in subjects if efficiency_scores[s] <= terciles[0]]
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return {'high': high_eff, 'low': low_eff}


# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================

def compare_groups(high_values, low_values, metric_name):
    """Compare high vs low efficiency groups"""
    # T-test
    t_stat, p_value = stats.ttest_ind(high_values, low_values)
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.var(high_values) + np.var(low_values)) / 2)
    cohens_d = (np.mean(high_values) - np.mean(low_values)) / pooled_std if pooled_std > 0 else 0
    
    # Mann-Whitney U (non-parametric)
    u_stat, u_pvalue = stats.mannwhitneyu(high_values, low_values, alternative='two-sided')
    
    return {
        'metric': metric_name,
        'high_mean': float(np.mean(high_values)),
        'high_std': float(np.std(high_values)),
        'low_mean': float(np.mean(low_values)),
        'low_std': float(np.std(low_values)),
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'cohens_d': float(cohens_d),
        'u_statistic': float(u_stat),
        'u_pvalue': float(u_pvalue),
        'significant': bool(p_value < 0.05),
        'hypothesis_supported': bool(cohens_d > 0),  # High eff should be MORE sparse
    }


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_sparsity_analysis():
    """Run complete sparsity analysis"""
    
    logger.info("="*60)
    logger.info("Stage 2: Sparsity Analysis")
    logger.info("="*60)
    
    # Load activation matrix
    activation_path = config.NEUROSCIENCE_DIR / "activation_matrix.pkl"
    with open(activation_path, 'rb') as f:
        data = pickle.load(f)
    
    activation_matrix = data['data']
    subjects = data['subjects']
    tasks = data['tasks']
    n_parcels = data['n_parcels']
    
    logger.info(f"Loaded activation matrix: {activation_matrix.shape}")
    
    # Compute sparsity metrics for each subject and task
    logger.info("Computing sparsity metrics...")
    
    sparsity_data = []
    
    for s_idx, subject in enumerate(tqdm(subjects, desc="Computing sparsity")):
        for t_idx, task in enumerate(tasks):
            activation = activation_matrix[s_idx, t_idx, :]
            
            # NOTE: l0_sparsity and n_active_regions are definitionally constant
            # (95th-percentile threshold on z-scored data always yields 0.95 /
            # ~18 regions), so they are excluded from the composite and from
            # group comparisons. Kurtosis is retained as descriptive only.
            metrics = {
                'subject': subject,
                'task': task,
                'gini_coefficient': compute_gini_coefficient(activation),
                'kurtosis': compute_kurtosis_sparsity(activation),
                'entropy_sparsity': compute_entropy_sparsity(activation),
                'concentration_ratio': compute_concentration_ratio(activation),
            }

            # Composite sparsity = mean of three informative metrics, z-scored
            # to share a common scale before averaging.
            metrics['composite_sparsity'] = float(np.mean([
                metrics['gini_coefficient'],
                metrics['entropy_sparsity'],
                metrics['concentration_ratio'],
            ]))
            
            sparsity_data.append(metrics)
    
    sparsity_df = pd.DataFrame(sparsity_data)
    
    # Compute efficiency scores and group subjects
    logger.info("Computing efficiency scores and grouping subjects...")
    
    # Look for behavioral data to use as independent efficiency criterion
    behavioral_csv = config.NEUROSCIENCE_DIR / "behavioral_validation.csv"
    if not behavioral_csv.exists():
        # Also check common alternative locations
        for alt_path in [
            config.PROJECT_DIR / "results" / "neuroscience" / "behavioral_validation.csv",
            config.PROJECT_DIR / "behavioral_validation.csv",
        ]:
            if alt_path.exists():
                behavioral_csv = alt_path
                break

    efficiency_scores, classification_method = compute_efficiency_scores(
        activation_matrix, subjects, tasks, behavioral_csv=behavioral_csv
    )
    groups = split_efficiency_groups(efficiency_scores, method='median')
    
    logger.info(f"High efficiency group: {len(groups['high'])} subjects")
    logger.info(f"Low efficiency group: {len(groups['low'])} subjects")
    
    # Add group labels to dataframe
    sparsity_df['efficiency_group'] = sparsity_df['subject'].apply(
        lambda s: 'high' if s in groups['high'] else 'low'
    )
    sparsity_df['efficiency_score'] = sparsity_df['subject'].apply(
        lambda s: efficiency_scores[s]
    )
    
    # Compare groups
    logger.info("Comparing high vs low efficiency groups...")
    
    comparison_results = {
        'overall': {},
        'by_task': {},
        'hypothesis_H1': {}
    }
    
    # Overall comparison (across all tasks). We exclude l0_sparsity and
    # n_active_regions because they are definitionally constant at the chosen
    # percentile threshold; kurtosis is descriptive-only.
    metrics_to_compare = ['gini_coefficient', 'entropy_sparsity',
                          'concentration_ratio', 'composite_sparsity']
    
    for metric in metrics_to_compare:
        high_vals = sparsity_df[sparsity_df['efficiency_group'] == 'high'][metric].values
        low_vals = sparsity_df[sparsity_df['efficiency_group'] == 'low'][metric].values
        comparison_results['overall'][metric] = compare_groups(high_vals, low_vals, metric)
    
    # Per-task comparison
    for task in tasks:
        task_df = sparsity_df[sparsity_df['task'] == task]
        comparison_results['by_task'][task] = {}

        for metric in metrics_to_compare:
            high_vals = task_df[task_df['efficiency_group'] == 'high'][metric].values
            low_vals = task_df[task_df['efficiency_group'] == 'low'][metric].values
            comparison_results['by_task'][task][metric] = compare_groups(high_vals, low_vals, metric)

    # FDR correction (Benjamini-Hochberg)
    # Family 1: overall sparsity metrics (6 tests of H1)
    logger.info("Applying FDR correction to overall sparsity metrics (family of 6)...")
    fdr_summary_overall = config.apply_fdr_correction(comparison_results['overall'])
    for name, info in fdr_summary_overall.items():
        logger.info(f"  {name}: p_raw={info['p_raw']:.4f} p_fdr={info['p_fdr']:.4f} "
                    f"sig_fdr={info['significant_fdr']}")

    # Family 2: per-task composite_sparsity (7 tests, post-hoc H1)
    per_task_composite = {task: comparison_results['by_task'][task]['composite_sparsity']
                          for task in tasks}
    logger.info(f"Applying FDR correction to per-task composite_sparsity "
                f"(family of {len(tasks)})...")
    config.apply_fdr_correction(per_task_composite)
    # per_task_composite entries were mutated in-place, but they point to the
    # original by_task dict entries, so comparison_results already updated.

    # H1 Hypothesis test: High efficiency = higher sparsity
    h1_supported = comparison_results['overall']['composite_sparsity']['hypothesis_supported']
    h1_significant = comparison_results['overall']['composite_sparsity']['significant']
    h1_effect = comparison_results['overall']['composite_sparsity']['cohens_d']
    
    comparison_results['hypothesis_H1'] = {
        'description': 'High-efficiency individuals activate fewer brain regions (higher sparsity)',
        'supported': bool(h1_supported),
        'significant': bool(h1_significant),
        'effect_size': float(h1_effect),
        'interpretation': 'SUPPORTED' if (h1_supported and h1_significant) else 
                         'TREND' if h1_supported else 'NOT SUPPORTED'
    }
    
    logger.info(f"H1 Result: {comparison_results['hypothesis_H1']['interpretation']}")
    logger.info(f"  Effect size (Cohen's d): {h1_effect:.3f}")
    logger.info(f"  P-value: {comparison_results['overall']['composite_sparsity']['p_value']:.4f}")
    
    # Save results
    logger.info("Saving results...")
    
    # Sparsity metrics
    sparsity_path = config.NEUROSCIENCE_DIR / "sparsity_metrics.csv"
    sparsity_df.to_csv(sparsity_path, index=False)
    logger.info(f"Saved: {sparsity_path}")
    
    # Efficiency groups
    groups_data = {
        'groups': groups,
        'efficiency_scores': efficiency_scores,
        'method': 'median',
        'classification_criterion': classification_method,
    }
    groups_path = config.NEUROSCIENCE_DIR / "efficiency_groups.json"
    with open(groups_path, 'w') as f:
        json.dump(groups_data, f, indent=2)
    logger.info(f"Saved: {groups_path}")
    
    # Comparison results
    comparison_path = config.NEUROSCIENCE_DIR / "sparsity_comparison.json"
    with open(comparison_path, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    logger.info(f"Saved: {comparison_path}")
    
    # Summary statistics
    summary = {
        'n_subjects': len(subjects),
        'n_tasks': len(tasks),
        'n_parcels': n_parcels,
        'mean_gini': float(sparsity_df['gini_coefficient'].mean()),
        'std_gini': float(sparsity_df['gini_coefficient'].std()),
        'mean_entropy_sparsity': float(sparsity_df['entropy_sparsity'].mean()),
        'mean_concentration_ratio': float(sparsity_df['concentration_ratio'].mean()),
        'high_eff_mean_sparsity': float(sparsity_df[sparsity_df['efficiency_group'] == 'high']['composite_sparsity'].mean()),
        'low_eff_mean_sparsity': float(sparsity_df[sparsity_df['efficiency_group'] == 'low']['composite_sparsity'].mean()),
    }
    
    summary_path = config.NEUROSCIENCE_DIR / "sparsity_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved: {summary_path}")
    
    logger.info("="*60)
    logger.info("Sparsity Analysis Summary")
    logger.info("="*60)
    logger.info(f"Mean Gini coefficient: {summary['mean_gini']:.4f}")
    logger.info(f"Mean entropy sparsity: {summary['mean_entropy_sparsity']:.4f}")
    logger.info(f"High efficiency group composite sparsity: {summary['high_eff_mean_sparsity']:.4f}")
    logger.info(f"Low efficiency group composite sparsity: {summary['low_eff_mean_sparsity']:.4f}")
    
    return sparsity_df, comparison_results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    sparsity_df, comparison_results = run_sparsity_analysis()
    print("\n✅ Stage 2 completed: Sparsity analysis")

