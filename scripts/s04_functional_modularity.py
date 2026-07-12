#!/usr/bin/env python3
"""
=============================================================================
Stage 4: Functional Modularity Analysis
=============================================================================

Analyze functional connectivity and modularity across subjects.
Tests H4: High-efficiency individuals have clearer functional module boundaries.

Outputs:
    - connectivity_matrices.pkl: Functional connectivity per subject/task
    - modularity_metrics.csv: Modularity scores
    - community_assignments.pkl: Community detection results
    - modularity_analysis.json: Statistical analysis
"""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
import networkx as nx
from networkx.algorithms.community import louvain_communities
import pickle
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from configs import config

# Ensure timestamped run directories are initialized
config.ensure_run_directories()

logger = config.setup_logging('functional_modularity')

# =============================================================================
# CONNECTIVITY ANALYSIS
# =============================================================================

def compute_functional_connectivity(activation_matrix):
    """
    Compute functional connectivity from activation patterns.
    Uses correlation between parcel activation profiles across tasks.
    """
    # activation_matrix shape: (tasks, parcels)
    # Compute correlation between parcels based on their activation across tasks
    
    n_parcels = activation_matrix.shape[1]
    connectivity = np.corrcoef(activation_matrix.T)
    
    # Handle NaN values
    connectivity = np.nan_to_num(connectivity, nan=0)
    np.fill_diagonal(connectivity, 0)
    
    return connectivity


def threshold_connectivity(connectivity, threshold=0.1):
    """Apply threshold to connectivity matrix"""
    thresholded = connectivity.copy()
    thresholded[np.abs(thresholded) < threshold] = 0
    return thresholded


# =============================================================================
# MODULARITY ANALYSIS
# =============================================================================

def compute_modularity(connectivity, community_assignments):
    """
    Compute Newman's modularity Q.
    Q = (1/2m) * sum_ij[(A_ij - k_i*k_j/2m) * delta(c_i, c_j)]
    """
    A = connectivity.copy()
    np.fill_diagonal(A, 0)
    A = np.abs(A)  # Use absolute values
    
    m = np.sum(A) / 2
    if m == 0:
        return 0.0
    
    k = np.sum(A, axis=1)  # Degree of each node
    n = len(A)
    
    Q = 0.0
    for i in range(n):
        for j in range(n):
            if community_assignments[i] == community_assignments[j]:
                Q += A[i, j] - (k[i] * k[j]) / (2 * m)
    
    Q /= (2 * m)
    return Q


def detect_communities_hierarchical(connectivity, n_communities=7):
    """Ward hierarchical clustering with fixed k; kept for back-compat only."""
    similarity = np.abs(connectivity)
    np.fill_diagonal(similarity, 0)
    if np.std(similarity) < 1e-10:
        return np.zeros(len(connectivity), dtype=int)
    distance = 1 - similarity
    distance = np.clip(distance, 0, 2)
    condensed = distance[np.triu_indices(len(distance), k=1)]
    try:
        Z = linkage(condensed, method='ward')
        communities = fcluster(Z, n_communities, criterion='maxclust')
        return communities - 1
    except Exception:
        return np.zeros(len(connectivity), dtype=int)


def detect_communities_louvain(connectivity, resolution=1.0, seed=0):
    """
    Louvain community detection on the absolute-valued connectivity matrix.

    Returns a 1-D integer label array with community id per node. The number
    of communities is decided by the algorithm, not pre-specified.
    """
    W = np.abs(connectivity).copy()
    np.fill_diagonal(W, 0.0)
    # Drop near-zero edges to keep the graph sparse enough for Louvain
    W[W < 1e-6] = 0.0
    if W.sum() == 0:
        return np.zeros(len(W), dtype=int)
    G = nx.from_numpy_array(W)
    try:
        comms = louvain_communities(G, resolution=resolution, seed=seed, weight='weight')
    except Exception:
        return detect_communities_hierarchical(connectivity, n_communities=7)
    labels = np.zeros(len(W), dtype=int)
    for c_idx, nodes in enumerate(comms):
        for n in nodes:
            labels[n] = c_idx
    return labels


def compute_module_segregation(connectivity, community_assignments):
    """
    Compute module segregation index.
    Higher = stronger within-module vs between-module connectivity.
    """
    communities = np.unique(community_assignments)
    n = len(connectivity)
    
    within_module = []
    between_module = []
    
    for i in range(n):
        for j in range(i + 1, n):
            if community_assignments[i] == community_assignments[j]:
                within_module.append(np.abs(connectivity[i, j]))
            else:
                between_module.append(np.abs(connectivity[i, j]))
    
    mean_within = np.mean(within_module) if within_module else 0
    mean_between = np.mean(between_module) if between_module else 0
    
    if mean_within + mean_between == 0:
        return 0.0
    
    segregation = (mean_within - mean_between) / (mean_within + mean_between)
    return segregation


def compute_participation_coefficient(connectivity, community_assignments):
    """
    Compute participation coefficient for each node.
    Higher = more diverse connections across modules.
    Lower = more connections within own module.
    """
    n = len(connectivity)
    communities = np.unique(community_assignments)
    
    participation = np.zeros(n)
    
    for i in range(n):
        k_i = np.sum(np.abs(connectivity[i, :]))  # Total degree
        if k_i == 0:
            continue
        
        sum_sq = 0
        for c in communities:
            # Connections to nodes in community c
            mask = community_assignments == c
            k_ic = np.sum(np.abs(connectivity[i, mask]))
            sum_sq += (k_ic / k_i) ** 2
        
        participation[i] = 1 - sum_sq
    
    return participation


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_modularity_analysis():
    """Run complete modularity analysis"""
    
    logger.info("="*60)
    logger.info("Stage 4: Functional Modularity Analysis")
    logger.info("="*60)
    
    # Load data
    activation_path = config.NEUROSCIENCE_DIR / "activation_matrix.pkl"
    with open(activation_path, 'rb') as f:
        data = pickle.load(f)
    
    activation_matrix = data['data']
    subjects = data['subjects']
    tasks = data['tasks']
    n_parcels = data['n_parcels']
    
    groups_path = config.NEUROSCIENCE_DIR / "efficiency_groups.json"
    with open(groups_path, 'r') as f:
        groups_data = json.load(f)
    groups = groups_data['groups']
    
    logger.info(f"Loaded activation matrix: {activation_matrix.shape}")
    
    # Compute connectivity and modularity for each subject
    logger.info("Computing functional connectivity and modularity...")
    
    connectivity_matrices = {}
    community_assignments = {}
    modularity_data = []
    
    for s_idx, subject in enumerate(tqdm(subjects, desc="Processing subjects")):
        # Get subject's activation across tasks
        subject_activation = activation_matrix[s_idx, :, :]  # (tasks, parcels)
        
        # Compute functional connectivity
        connectivity = compute_functional_connectivity(subject_activation)
        connectivity_matrices[subject] = connectivity
        
        # Detect communities via Louvain (primary), as described in manuscript
        # Methods. Number of communities is determined by the algorithm.
        communities = detect_communities_louvain(connectivity, resolution=1.0, seed=0)
        community_assignments[subject] = communities
        
        # Compute modularity metrics
        Q = compute_modularity(connectivity, communities)
        segregation = compute_module_segregation(connectivity, communities)
        participation = compute_participation_coefficient(connectivity, communities)
        
        modularity_data.append({
            'subject': subject,
            'modularity_Q': Q,
            'segregation': segregation,
            'mean_participation': np.mean(participation),
            'std_participation': np.std(participation),
            'n_communities': len(np.unique(communities)),
            'efficiency_group': 'high' if subject in groups['high'] else 'low'
        })
    
    modularity_df = pd.DataFrame(modularity_data)
    
    # Compare groups
    logger.info("Comparing efficiency groups...")
    
    metrics_to_compare = ['modularity_Q', 'segregation', 'mean_participation']
    comparison_results = {}
    
    for metric in metrics_to_compare:
        high_vals = modularity_df[modularity_df['efficiency_group'] == 'high'][metric].values
        low_vals = modularity_df[modularity_df['efficiency_group'] == 'low'][metric].values
        
        t_stat, p_value = stats.ttest_ind(high_vals, low_vals)
        pooled_std = np.sqrt((np.var(high_vals) + np.var(low_vals)) / 2)
        cohens_d = (np.mean(high_vals) - np.mean(low_vals)) / pooled_std if pooled_std > 0 else 0
        
        comparison_results[metric] = {
            'high_mean': float(np.mean(high_vals)),
            'high_std': float(np.std(high_vals)),
            'low_mean': float(np.mean(low_vals)),
            'low_std': float(np.std(low_vals)),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'cohens_d': float(cohens_d),
            'significant': bool(p_value < 0.05)
        }

    # FDR correction across the 3 network-organization metrics (H4 family)
    logger.info("Applying FDR correction to modularity family (3 metrics)...")
    fdr_summary = config.apply_fdr_correction(comparison_results)
    for name, info in fdr_summary.items():
        logger.info(f"  {name}: p_raw={info['p_raw']:.4f} p_fdr={info['p_fdr']:.4f} "
                    f"sig_fdr={info['significant_fdr']}")

    # H4: Higher modularity in efficient individuals
    mod_d = comparison_results['modularity_Q']['cohens_d']
    mod_p = comparison_results['modularity_Q']['p_value']
    seg_p = comparison_results['segregation']['p_value']
    part_p = comparison_results['mean_participation']['p_value']
    mod_p_fdr = comparison_results['modularity_Q'].get('p_value_fdr', mod_p)
    seg_p_fdr = comparison_results['segregation'].get('p_value_fdr', seg_p)
    part_p_fdr = comparison_results['mean_participation'].get('p_value_fdr', part_p)

    # Determine interpretation based on direction AND significance
    if mod_d > 0 and mod_p < 0.05:
        h4_interp = 'SUPPORTED'
    elif mod_d < 0 and mod_p < 0.05:
        h4_interp = 'REVERSED'
    elif mod_d > 0:
        h4_interp = 'TREND'
    else:
        h4_interp = 'NOT SUPPORTED'

    h4_result = {
        'description': 'High-efficiency individuals have clearer functional module boundaries',
        'modularity_difference': float(mod_d),
        'segregation_difference': float(comparison_results['segregation']['cohens_d']),
        'participation_difference': float(comparison_results['mean_participation']['cohens_d']),
        'supported': bool(mod_d > 0 and mod_p < 0.05),
        'significant': bool(mod_p < 0.05),
        'interpretation': h4_interp,
        'note': (
            f'Modularity Q: d={mod_d:.2f}, p={mod_p:.3f} (FDR {mod_p_fdr:.3f}, '
            f'{"sig" if mod_p_fdr < 0.05 else "n.s."}); '
            f'Segregation: p={seg_p:.3f} (FDR {seg_p_fdr:.3f}, '
            f'{"sig" if seg_p_fdr < 0.05 else "n.s."}); '
            f'Participation: p={part_p:.3f} (FDR {part_p_fdr:.3f}, '
            f'{"sig" if part_p_fdr < 0.05 else "n.s."}). '
            f'{"Only modularity Q reached FDR-corrected significance." if (mod_p_fdr < 0.05 and seg_p_fdr >= 0.05 and part_p_fdr >= 0.05) else ""}'
        )
    }
    
    logger.info(f"H4 Result: {h4_result['interpretation']}")
    logger.info(f"  Modularity effect: {h4_result['modularity_difference']:.3f}")
    logger.info(f"  High group Q: {comparison_results['modularity_Q']['high_mean']:.4f}")
    logger.info(f"  Low group Q: {comparison_results['modularity_Q']['low_mean']:.4f}")
    
    # Compute group-average connectivity matrices
    logger.info("Computing group-average connectivity...")
    
    group_connectivity = {}
    for group_name, group_subjects in groups.items():
        group_conn = [connectivity_matrices[s] for s in group_subjects if s in connectivity_matrices]
        if group_conn:
            group_connectivity[group_name] = np.mean(group_conn, axis=0)
    
    # Save results
    logger.info("Saving results...")
    
    # Connectivity matrices
    conn_path = config.NEUROSCIENCE_DIR / "connectivity_matrices.pkl"
    with open(conn_path, 'wb') as f:
        pickle.dump({
            'individual': connectivity_matrices,
            'group_average': group_connectivity,
            'n_parcels': n_parcels,
        }, f)
    logger.info(f"Saved: {conn_path}")
    
    # Community assignments
    comm_path = config.NEUROSCIENCE_DIR / "community_assignments.pkl"
    with open(comm_path, 'wb') as f:
        pickle.dump(community_assignments, f)
    logger.info(f"Saved: {comm_path}")
    
    # Modularity metrics
    mod_path = config.NEUROSCIENCE_DIR / "modularity_metrics.csv"
    modularity_df.to_csv(mod_path, index=False)
    logger.info(f"Saved: {mod_path}")
    
    # Analysis results
    analysis_results = {
        'comparison': comparison_results,
        'hypothesis_H4': h4_result,
        'summary': {
            'mean_modularity': float(modularity_df['modularity_Q'].mean()),
            'mean_segregation': float(modularity_df['segregation'].mean()),
            'high_group_modularity': float(modularity_df[modularity_df['efficiency_group'] == 'high']['modularity_Q'].mean()),
            'low_group_modularity': float(modularity_df[modularity_df['efficiency_group'] == 'low']['modularity_Q'].mean()),
        }
    }
    
    analysis_path = config.NEUROSCIENCE_DIR / "modularity_analysis.json"
    with open(analysis_path, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    logger.info(f"Saved: {analysis_path}")
    
    logger.info("="*60)
    logger.info("Modularity Analysis Summary")
    logger.info("="*60)
    logger.info(f"Mean modularity Q: {analysis_results['summary']['mean_modularity']:.4f}")
    logger.info(f"Mean segregation: {analysis_results['summary']['mean_segregation']:.4f}")
    logger.info(f"High group mean Q: {analysis_results['summary']['high_group_modularity']:.4f}")
    logger.info(f"Low group mean Q: {analysis_results['summary']['low_group_modularity']:.4f}")
    
    return modularity_df, analysis_results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    modularity_df, analysis_results = run_modularity_analysis()
    print("\n✅ Stage 4 completed: Functional modularity analysis")

