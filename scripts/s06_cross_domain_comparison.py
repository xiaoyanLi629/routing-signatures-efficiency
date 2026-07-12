#!/usr/bin/env python3
"""
=============================================================================
Stage 6: Cross-Domain Comparison
=============================================================================

Compare brain routing patterns with MoE routing patterns.
Tests H7: Brain and MoE routing patterns show structural similarity.

Outputs:
    - brain_moe_comparison.json: Detailed comparison results
    - similarity_metrics.csv: Similarity measures
    - cross_domain_analysis.json: Complete analysis
"""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cosine, correlation
from sklearn.preprocessing import StandardScaler
import pickle
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from configs import config

# Ensure timestamped run directories are initialized
config.ensure_run_directories()

logger = config.setup_logging('cross_domain_comparison')

# =============================================================================
# COMPARISON FUNCTIONS
# =============================================================================

def compute_structural_similarity(brain_matrix, moe_matrix):
    """
    Compute structural similarity between brain task-network matrix
    and MoE category-expert matrix.
    """
    # Ensure same dimensions by averaging/aggregating if needed
    # Brain: (7 tasks × 7 networks), MoE: (7 categories × 8 experts)
    
    # Normalize both matrices
    brain_norm = StandardScaler().fit_transform(brain_matrix)
    moe_norm = StandardScaler().fit_transform(moe_matrix)
    
    # Compute correlation of flattened matrices
    brain_flat = brain_norm.flatten()
    moe_flat = moe_norm.flatten()
    
    # Need same length - use min
    min_len = min(len(brain_flat), len(moe_flat))
    r, p = stats.pearsonr(brain_flat[:min_len], moe_flat[:min_len])
    
    return {'correlation': float(r), 'p_value': float(p)}


def compute_sparsity_correlation(brain_sparsity, moe_sparsity):
    """
    Compare sparsity patterns between brain and MoE.
    """
    # Get sparsity values
    if isinstance(brain_sparsity, pd.DataFrame):
        brain_vals = brain_sparsity.groupby('task')['composite_sparsity'].mean().values
    else:
        brain_vals = list(brain_sparsity.values())
    
    if isinstance(moe_sparsity, pd.DataFrame):
        moe_vals = moe_sparsity.groupby('category')['sparsity'].mean().values
    else:
        moe_vals = list(moe_sparsity.values())
    
    # Ensure same length
    min_len = min(len(brain_vals), len(moe_vals))
    r, p = stats.pearsonr(brain_vals[:min_len], moe_vals[:min_len])
    
    return {
        'correlation': float(r),
        'p_value': float(p),
        'brain_mean': float(np.mean(brain_vals)),
        'moe_mean': float(np.mean(moe_vals)),
    }


def compute_specialization_similarity(brain_selectivity, moe_specialization):
    """
    Compare specialization patterns.
    """
    # Brain selectivity per task
    if isinstance(brain_selectivity, dict):
        brain_vals = list(brain_selectivity.values())
    else:
        brain_vals = brain_selectivity
    
    # MoE specialization per expert
    if isinstance(moe_specialization, dict):
        moe_vals = [v['selectivity'] for v in moe_specialization.values()]
    else:
        moe_vals = moe_specialization
    
    # Compute summary statistics comparison
    brain_mean = np.mean(brain_vals)
    moe_mean = np.mean(moe_vals)
    
    # Test if distributions are similar
    if len(brain_vals) >= 3 and len(moe_vals) >= 3:
        t_stat, p_value = stats.ttest_ind(brain_vals, moe_vals)
    else:
        t_stat, p_value = 0, 1
    
    return {
        'brain_mean_specialization': float(brain_mean),
        'moe_mean_specialization': float(moe_mean),
        'difference': float(brain_mean - moe_mean),
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'similar': bool(p_value > 0.05),  # Not significantly different = similar
    }


def compute_routing_pattern_similarity(brain_routing, moe_routing):
    """
    Compare overall routing pattern structures.
    """
    similarities = []
    
    # Compare variance structure
    if isinstance(brain_routing, np.ndarray):
        brain_var = np.var(brain_routing, axis=1)  # Variance across targets (networks/experts)
    else:
        brain_var = np.array([np.var(list(v.values())) for v in brain_routing.values()])
    
    if isinstance(moe_routing, np.ndarray):
        moe_var = np.var(moe_routing, axis=1)
    else:
        moe_var = np.array([np.var(list(v.values())) for v in moe_routing.values()])
    
    # Correlation of variance patterns
    min_len = min(len(brain_var), len(moe_var))
    if min_len >= 2:
        r, p = stats.pearsonr(brain_var[:min_len], moe_var[:min_len])
        similarities.append({
            'measure': 'variance_pattern',
            'correlation': float(r),
            'p_value': float(p)
        })
    
    return similarities


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_cross_domain_comparison():
    """Run complete cross-domain comparison"""
    
    logger.info("="*60)
    logger.info("Stage 6: Cross-Domain Comparison")
    logger.info("="*60)
    
    # Load brain data
    logger.info("Loading brain analysis results...")
    
    # Task-network matrix
    tn_path = config.NEUROSCIENCE_DIR / "task_network_matrix.pkl"
    with open(tn_path, 'rb') as f:
        tn_data = pickle.load(f)
    brain_tn_matrix = tn_data['overall']
    
    # Sparsity data
    sparsity_path = config.NEUROSCIENCE_DIR / "sparsity_metrics.csv"
    brain_sparsity = pd.read_csv(sparsity_path)
    
    # Routing analysis
    routing_path = config.NEUROSCIENCE_DIR / "routing_analysis.json"
    with open(routing_path, 'r') as f:
        brain_routing = json.load(f)
    
    # Modularity data
    mod_path = config.NEUROSCIENCE_DIR / "modularity_analysis.json"
    with open(mod_path, 'r') as f:
        brain_modularity = json.load(f)
    
    # Load MoE data
    logger.info("Loading MoE analysis results...")
    
    # Category-expert matrix
    ce_path = config.MOE_ANALYSIS_DIR / "category_expert_matrix.pkl"
    with open(ce_path, 'rb') as f:
        ce_data = pickle.load(f)
    moe_ce_matrix = ce_data['matrix']
    
    # Sparsity data
    moe_sparsity_path = config.MOE_ANALYSIS_DIR / "moe_sparsity.csv"
    moe_sparsity = pd.read_csv(moe_sparsity_path)
    
    # Specialization
    spec_path = config.MOE_ANALYSIS_DIR / "expert_specialization.json"
    with open(spec_path, 'r') as f:
        moe_specialization = json.load(f)
    
    # MoE analysis
    moe_path = config.MOE_ANALYSIS_DIR / "moe_analysis.json"
    with open(moe_path, 'r') as f:
        moe_analysis = json.load(f)
    
    # Run comparisons
    logger.info("Computing cross-domain comparisons...")
    
    comparison_results = {}
    
    # 1. Structural similarity
    logger.info("1. Structural similarity...")
    structural = compute_structural_similarity(brain_tn_matrix, moe_ce_matrix)
    comparison_results['structural_similarity'] = structural
    
    # 2. Sparsity correlation
    logger.info("2. Sparsity comparison...")
    sparsity_comp = compute_sparsity_correlation(brain_sparsity, moe_sparsity)
    comparison_results['sparsity_comparison'] = sparsity_comp
    
    # 3. Specialization similarity
    logger.info("3. Specialization comparison...")
    brain_selectivity = brain_routing.get('selectivity', {})
    spec_comp = compute_specialization_similarity(brain_selectivity, moe_specialization)
    comparison_results['specialization_comparison'] = spec_comp
    
    # 4. Routing pattern similarity
    logger.info("4. Routing pattern comparison...")
    pattern_sim = compute_routing_pattern_similarity(brain_tn_matrix, moe_ce_matrix)
    comparison_results['routing_pattern_similarity'] = pattern_sim
    
    # Compute overall similarity score
    similarity_scores = []
    
    if not np.isnan(structural['correlation']):
        similarity_scores.append(np.abs(structural['correlation']))
    
    if not np.isnan(sparsity_comp['correlation']):
        similarity_scores.append(np.abs(sparsity_comp['correlation']))
    
    if spec_comp['similar']:
        similarity_scores.append(0.7)  # Bonus for similar distributions
    
    overall_similarity = np.mean(similarity_scores) if similarity_scores else 0
    
    # H7: Brain and MoE routing patterns show structural similarity
    h7_result = {
        'description': 'Brain and MoE routing patterns show structural similarity',
        'overall_similarity': float(overall_similarity),
        'structural_correlation': float(structural['correlation']),
        'sparsity_correlation': float(sparsity_comp['correlation']),
        'specialization_similar': bool(spec_comp['similar']),
        'supported': bool(overall_similarity > 0.3),
        'interpretation': 'SUPPORTED' if overall_similarity > 0.5 else 
                         'PARTIAL' if overall_similarity > 0.3 else 'NOT SUPPORTED'
    }
    
    comparison_results['hypothesis_H7'] = h7_result
    
    logger.info(f"H7 Result: {h7_result['interpretation']}")
    logger.info(f"  Overall similarity: {overall_similarity:.3f}")
    logger.info(f"  Structural correlation: {structural['correlation']:.3f}")
    logger.info(f"  Sparsity correlation: {sparsity_comp['correlation']:.3f}")
    
    # Create comparison summary table
    summary_data = [
        {
            'comparison': 'Structural Similarity',
            'brain_metric': 'Task-Network Matrix',
            'moe_metric': 'Category-Expert Matrix',
            'correlation': float(structural['correlation']),
            'p_value': float(structural['p_value']),
            'significant': bool(structural['p_value'] < 0.05),
        },
        {
            'comparison': 'Sparsity Pattern',
            'brain_metric': 'Activation Sparsity',
            'moe_metric': 'Routing Sparsity',
            'correlation': float(sparsity_comp['correlation']),
            'p_value': float(sparsity_comp.get('p_value', 1.0)),
            'significant': bool(sparsity_comp.get('p_value', 1) < 0.05),
        },
        {
            'comparison': 'Specialization Level',
            'brain_metric': f"Mean = {spec_comp['brain_mean_specialization']:.3f}",
            'moe_metric': f"Mean = {spec_comp['moe_mean_specialization']:.3f}",
            'correlation': float('nan'),  # Not a correlation
            'p_value': float(spec_comp['p_value']),
            'significant': bool(spec_comp['similar']),
        },
    ]
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save results
    logger.info("Saving results...")
    
    # Comparison results
    comp_path = config.COMPARISON_DIR / "brain_moe_comparison.json"
    with open(comp_path, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    logger.info(f"Saved: {comp_path}")
    
    # Summary table
    summary_path = config.COMPARISON_DIR / "similarity_metrics.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Saved: {summary_path}")
    
    # Complete analysis
    full_analysis = {
        'brain_summary': {
            'n_tasks': brain_tn_matrix.shape[0],
            'n_networks': brain_tn_matrix.shape[1],
            'mean_sparsity': float(brain_sparsity['composite_sparsity'].mean()),
            'mean_selectivity': float(np.mean(list(brain_selectivity.values()))),
        },
        'moe_summary': {
            'n_categories': moe_ce_matrix.shape[0],
            'n_experts': moe_ce_matrix.shape[1],
            'mean_sparsity': float(moe_sparsity['sparsity'].mean()),
            'mean_specialization': float(np.mean([v['selectivity'] for v in moe_specialization.values()])),
        },
        'comparisons': comparison_results,
        'overall_similarity': float(overall_similarity),
    }
    
    analysis_path = config.COMPARISON_DIR / "cross_domain_analysis.json"
    with open(analysis_path, 'w') as f:
        json.dump(full_analysis, f, indent=2)
    logger.info(f"Saved: {analysis_path}")
    
    logger.info("="*60)
    logger.info("Cross-Domain Comparison Summary")
    logger.info("="*60)
    logger.info(f"Overall similarity score: {overall_similarity:.4f}")
    logger.info(f"Structural correlation: {structural['correlation']:.4f}")
    logger.info(f"Sparsity correlation: {sparsity_comp['correlation']:.4f}")
    logger.info(f"Specialization similar: {spec_comp['similar']}")
    
    return comparison_results, full_analysis


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    comparison_results, full_analysis = run_cross_domain_comparison()
    print("\n✅ Stage 6 completed: Cross-domain comparison")

