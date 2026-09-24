#!/usr/bin/env python3
"""
=============================================================================
Stage 3: Routing Pattern Analysis
=============================================================================

Analyze task-network routing patterns across subjects.
Tests H2: Different tasks activate different specialized network modules.
Tests H3: High-efficiency individuals show more consistent routing.

Outputs:
    - task_network_matrix.pkl: Task × Network activation matrix
    - routing_patterns.pkl: Individual routing profiles
    - routing_analysis.json: Statistical analysis of routing patterns
"""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import mutual_info_score
import pickle
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from configs import config

# Ensure timestamped run directories are initialized
config.ensure_run_directories()

logger = config.setup_logging('routing_patterns')

# =============================================================================
# NETWORK ASSIGNMENT
# =============================================================================

_PARCEL_NETWORK_CACHE = {'map': None}


def assign_parcels_to_networks(n_parcels=360, n_networks=7):
    """
    Assign each of the 360 Glasser cortical parcels to one of the 7 Yeo-like
    networks (`config.NETWORK_ORDER`) using the Cole-Anticevic 12-network
    CIFTI (mapped to Yeo-7 via `config.COLE_TO_YEO7`).

    Falls back to the old index-based assignment only if the atlas files are
    absent, with a logged warning. Returns a dict {parcel_idx: 'NET'}.
    """
    if _PARCEL_NETWORK_CACHE['map'] is not None:
        return _PARCEL_NETWORK_CACHE['map']

    import nibabel as nib
    glasser = config.GLASSER_INDICES_CIFTI
    cole    = config.COLE_NETWORK_CIFTI
    if not (glasser.exists() and cole.exists()):
        logger.warning(
            "Atlas files missing (%s or %s); falling back to index-based "
            "network assignment. This is not a real Yeo-7 mapping.",
            glasser, cole
        )
        pp = n_parcels // n_networks
        fallback = {i: config.NETWORK_ORDER[min(i // pp, n_networks - 1)]
                    for i in range(n_parcels)}
        _PARCEL_NETWORK_CACHE['map'] = fallback
        return fallback

    g_data = nib.load(str(glasser)).get_fdata().flatten().astype(int)
    n_data = nib.load(str(cole)).get_fdata().flatten().astype(int)
    assignments = {}
    for pid in range(1, n_parcels + 1):
        mask = g_data == pid
        if not mask.any():
            continue
        nets = n_data[mask]
        nets = nets[nets > 0]
        if len(nets) == 0:
            continue
        modal_cole = int(np.bincount(nets).argmax())
        yeo = config.COLE_TO_YEO7.get(modal_cole)
        if yeo is None:
            continue
        assignments[pid - 1] = yeo  # store 0-indexed parcel id
    _PARCEL_NETWORK_CACHE['map'] = assignments
    logger.info("Loaded real HCP-MMP/Cole-Anticevic -> Yeo-7 mapping "
                "(%d parcels assigned)", len(assignments))
    return assignments


def aggregate_to_networks(activation, parcel_assignments):
    """Aggregate parcel-level activation to network level"""
    networks = config.NETWORK_ORDER
    network_activation = {}
    
    for network in networks:
        parcel_indices = [p for p, n in parcel_assignments.items() if n == network]
        if parcel_indices:
            network_activation[network] = np.mean(activation[parcel_indices])
        else:
            network_activation[network] = 0.0
    
    return network_activation


# =============================================================================
# ROUTING METRICS
# =============================================================================

def compute_selectivity_index(task_network_matrix):
    """
    Compute selectivity index for each task.
    Higher = task activates specific networks more selectively.
    NOTE: This (max-mean)/max metric saturates near 1.0 when averaged
    across subjects. Use compute_entropy_selectivity() for a more
    discriminative metric.
    """
    selectivity = {}

    for task_idx, task in enumerate(config.TASK_ORDER):
        activation = task_network_matrix[task_idx, :]
        max_act = np.max(activation)
        mean_act = np.mean(activation)

        if max_act > 0:
            selectivity[task] = (max_act - mean_act) / max_act
        else:
            selectivity[task] = 0.0

    return selectivity


def compute_entropy_selectivity(task_network_matrix):
    """
    Compute entropy-based selectivity for each task.
    Uses normalized entropy of the activation distribution across networks.
    Returns values between 0 (uniform across networks) and 1 (single network).
    More discriminative than (max-mean)/max for cross-domain comparison.
    """
    n_tasks, n_networks = task_network_matrix.shape
    selectivity = {}

    for task_idx, task in enumerate(config.TASK_ORDER):
        activation = np.abs(task_network_matrix[task_idx, :])
        total = np.sum(activation)
        if total > 0:
            proportions = activation / total
            # Shannon entropy (base 2)
            entropy = -np.sum(proportions * np.log2(proportions + 1e-10))
            max_entropy = np.log2(n_networks)
            # Invert: 1 = maximally selective, 0 = uniform
            selectivity[task] = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.0
        else:
            selectivity[task] = 0.0

    return selectivity


def compute_routing_consistency(routing_profiles, groups):
    """
    Compute routing consistency within efficiency groups.
    Higher consistency = more similar routing across subjects.
    """
    consistency = {}
    
    for group_name, group_subjects in groups.items():
        if len(group_subjects) < 2:
            consistency[group_name] = 0.0
            continue
        
        # Get routing matrices for this group
        group_matrices = [routing_profiles[s] for s in group_subjects if s in routing_profiles]
        
        if len(group_matrices) < 2:
            consistency[group_name] = 0.0
            continue
        
        # Compute pairwise correlations
        correlations = []
        for i in range(len(group_matrices)):
            for j in range(i + 1, len(group_matrices)):
                flat_i = group_matrices[i].flatten()
                flat_j = group_matrices[j].flatten()
                r, _ = stats.pearsonr(flat_i, flat_j)
                if not np.isnan(r):
                    correlations.append(r)
        
        consistency[group_name] = np.mean(correlations) if correlations else 0.0
    
    return consistency


def compute_task_similarity_matrix(activation_matrix, tasks):
    """
    Compute similarity between tasks based on activation patterns.
    """
    n_tasks = len(tasks)
    similarity_matrix = np.zeros((n_tasks, n_tasks))
    
    for i in range(n_tasks):
        for j in range(n_tasks):
            # Average activation across subjects
            act_i = np.mean(activation_matrix[:, i, :], axis=0)
            act_j = np.mean(activation_matrix[:, j, :], axis=0)
            
            r, _ = stats.pearsonr(act_i, act_j)
            similarity_matrix[i, j] = r if not np.isnan(r) else 0
    
    return similarity_matrix


def identify_dominant_network(task_network_activation):
    """Identify the dominant network for each task"""
    dominant = {}
    
    for task, network_act in task_network_activation.items():
        if network_act:
            dominant[task] = max(network_act, key=network_act.get)
        else:
            dominant[task] = 'Unknown'
    
    return dominant


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_routing_analysis():
    """Run complete routing pattern analysis"""
    
    logger.info("="*60)
    logger.info("Stage 3: Routing Pattern Analysis")
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
    
    # Assign parcels to networks
    parcel_assignments = assign_parcels_to_networks(n_parcels)
    networks = config.NETWORK_ORDER
    
    # Compute task-network matrices for each subject
    logger.info("Computing task-network routing matrices...")
    
    routing_profiles = {}  # subject -> (tasks × networks) matrix
    task_network_data = []
    
    for s_idx, subject in enumerate(tqdm(subjects, desc="Processing subjects")):
        subject_matrix = np.zeros((len(tasks), len(networks)))
        
        for t_idx, task in enumerate(tasks):
            activation = activation_matrix[s_idx, t_idx, :]
            network_act = aggregate_to_networks(activation, parcel_assignments)
            
            for n_idx, network in enumerate(networks):
                subject_matrix[t_idx, n_idx] = network_act[network]
                
                task_network_data.append({
                    'subject': subject,
                    'task': task,
                    'network': network,
                    'activation': network_act[network],
                    'efficiency_group': 'high' if subject in groups['high'] else 'low'
                })
        
        routing_profiles[subject] = subject_matrix
    
    task_network_df = pd.DataFrame(task_network_data)
    
    # Compute group-level task-network matrices
    logger.info("Computing group-level routing matrices...")
    
    group_matrices = {}
    for group_name, group_subjects in groups.items():
        group_profiles = [routing_profiles[s] for s in group_subjects if s in routing_profiles]
        if group_profiles:
            group_matrices[group_name] = np.mean(group_profiles, axis=0)
    
    # Compute overall task-network matrix
    overall_matrix = np.mean(list(routing_profiles.values()), axis=0)
    
    # Compute selectivity indices.
    # PRIMARY: entropy-based selectivity (bounded [0, 1], not saturated).
    # DESCRIPTIVE: (max-mean)/max, known to saturate near 1 on 7-network
    # vectors and therefore not used for hypothesis decisions.
    logger.info("Computing task selectivity indices...")
    entropy_selectivity = compute_entropy_selectivity(overall_matrix)
    selectivity_maxmean = compute_selectivity_index(overall_matrix)
    # `selectivity` continues to be the name referenced downstream; point it
    # at the entropy-based metric so H2 decisions use the unsaturated one.
    selectivity = entropy_selectivity
    
    # Identify dominant networks per task
    task_network_activation = {}
    for t_idx, task in enumerate(tasks):
        task_network_activation[task] = {
            network: overall_matrix[t_idx, n_idx] 
            for n_idx, network in enumerate(networks)
        }
    
    dominant_networks = identify_dominant_network(task_network_activation)
    
    # Compare with expected networks
    expected_match = {}
    for task in tasks:
        expected = config.TASKS[task]['expected_networks']
        actual = dominant_networks[task]
        expected_match[task] = {
            'expected': expected,
            'actual': actual,
            'match': actual in expected
        }
    
    # Compute routing consistency
    logger.info("Computing routing consistency...")
    consistency = compute_routing_consistency(routing_profiles, groups)
    
    # Compute task similarity matrix
    logger.info("Computing task similarity matrix...")
    task_similarity = compute_task_similarity_matrix(activation_matrix, tasks)
    
    # Statistical comparison of consistency
    logger.info("Statistical analysis...")
    
    # Get within-group correlations for comparison
    high_corrs = []
    low_corrs = []
    
    for group_name, group_subjects in groups.items():
        group_profiles_list = [routing_profiles[s] for s in group_subjects if s in routing_profiles]
        
        for i in range(len(group_profiles_list)):
            for j in range(i + 1, len(group_profiles_list)):
                r, _ = stats.pearsonr(
                    group_profiles_list[i].flatten(), 
                    group_profiles_list[j].flatten()
                )
                if not np.isnan(r):
                    if group_name == 'high':
                        high_corrs.append(r)
                    else:
                        low_corrs.append(r)
    
    # Test H3: High efficiency = more consistent routing
    if high_corrs and low_corrs:
        t_stat, p_value = stats.ttest_ind(high_corrs, low_corrs)
        cohens_d = (np.mean(high_corrs) - np.mean(low_corrs)) / np.sqrt(
            (np.var(high_corrs) + np.var(low_corrs)) / 2
        )
    else:
        t_stat, p_value, cohens_d = 0, 1, 0
    
    h3_result = {
        'description': 'High-efficiency individuals show more consistent routing patterns',
        'high_consistency': float(np.mean(high_corrs)) if high_corrs else 0,
        'low_consistency': float(np.mean(low_corrs)) if low_corrs else 0,
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'cohens_d': float(cohens_d),
        'supported': bool(cohens_d > 0),
        'significant': bool(p_value < 0.05),
        'interpretation': 'SUPPORTED' if (cohens_d > 0 and p_value < 0.05) else 
                         'TREND' if cohens_d > 0 else 'NOT SUPPORTED'
    }
    
    # H2: Task specialization
    mean_selectivity = np.mean(list(selectivity.values()))
    n_matched = sum(1 for t in tasks if expected_match[t]['match'])
    
    # H2 decision now uses the entropy-based selectivity (primary) AND the
    # canonical match rate. A threshold of 0.3 on entropy_selectivity means
    # "activation noticeably concentrated on one network"; 0 would be uniform.
    mean_maxmean = float(np.mean(list(selectivity_maxmean.values()))) if selectivity_maxmean else 0.0
    h2_result = {
        'description': 'Different tasks activate different specialized network modules',
        'mean_selectivity_entropy': float(mean_selectivity),
        'mean_selectivity_maxmean_DEPRECATED': mean_maxmean,
        'expected_network_matches': int(n_matched),
        'total_tasks': int(len(tasks)),
        'match_rate': float(n_matched / len(tasks)),
        'supported': bool(mean_selectivity > 0.3 and n_matched >= len(tasks) // 2),
        'interpretation': ('SUPPORTED' if (mean_selectivity > 0.3 and n_matched >= len(tasks) // 2)
                           else 'PARTIALLY SUPPORTED' if (mean_selectivity > 0.3 and n_matched < len(tasks) // 2)
                           else 'NOT SUPPORTED'),
        'note': (f'Entropy-based selectivity mean={mean_selectivity:.3f} '
                 f'(0 = uniform across networks, 1 = single network). '
                 f'{n_matched}/{len(tasks)} tasks matched canonically predicted networks. '
                 f'(The (max-mean)/max index saturates near 1 on 7-element vectors '
                 f'and is reported descriptively only: {mean_maxmean:.3f}).'),
    }
    
    logger.info(f"H2 Result: {h2_result['interpretation']}")
    logger.info(f"  Mean selectivity: {mean_selectivity:.3f}")
    logger.info(f"  Network match rate: {h2_result['match_rate']:.1%}")
    
    logger.info(f"H3 Result: {h3_result['interpretation']}")
    logger.info(f"  High group consistency: {h3_result['high_consistency']:.3f}")
    logger.info(f"  Low group consistency: {h3_result['low_consistency']:.3f}")
    logger.info(f"  Effect size: {h3_result['cohens_d']:.3f}")
    
    # Save results
    logger.info("Saving results...")
    
    # Task-network matrix
    tn_path = config.NEUROSCIENCE_DIR / "task_network_matrix.pkl"
    with open(tn_path, 'wb') as f:
        pickle.dump({
            'overall': overall_matrix,
            'by_group': group_matrices,
            'tasks': tasks,
            'networks': networks,
        }, f)
    logger.info(f"Saved: {tn_path}")
    
    # Routing profiles
    rp_path = config.NEUROSCIENCE_DIR / "routing_profiles.pkl"
    with open(rp_path, 'wb') as f:
        pickle.dump(routing_profiles, f)
    logger.info(f"Saved: {rp_path}")
    
    # Task-network DataFrame
    tn_df_path = config.NEUROSCIENCE_DIR / "task_network_data.csv"
    task_network_df.to_csv(tn_df_path, index=False)
    logger.info(f"Saved: {tn_df_path}")
    
    # Task similarity
    sim_path = config.NEUROSCIENCE_DIR / "task_similarity_matrix.pkl"
    with open(sim_path, 'wb') as f:
        pickle.dump({
            'matrix': task_similarity,
            'tasks': tasks
        }, f)
    logger.info(f"Saved: {sim_path}")
    
    # Analysis results
    analysis_results = {
        'selectivity': selectivity,
        'entropy_selectivity': entropy_selectivity,
        'dominant_networks': dominant_networks,
        'expected_match': expected_match,
        'consistency': consistency,
        'hypothesis_H2': h2_result,
        'hypothesis_H3': h3_result,
    }
    
    analysis_path = config.NEUROSCIENCE_DIR / "routing_analysis.json"
    with open(analysis_path, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    logger.info(f"Saved: {analysis_path}")
    
    logger.info("="*60)
    logger.info("Routing Analysis Summary")
    logger.info("="*60)
    logger.info(f"Task-network matrix shape: {overall_matrix.shape}")
    logger.info(f"Mean selectivity: {mean_selectivity:.4f}")
    logger.info(f"High group consistency: {consistency.get('high', 0):.4f}")
    logger.info(f"Low group consistency: {consistency.get('low', 0):.4f}")
    
    return routing_profiles, analysis_results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    routing_profiles, analysis_results = run_routing_analysis()
    print("\n✅ Stage 3 completed: Routing pattern analysis")

