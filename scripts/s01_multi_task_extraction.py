#!/usr/bin/env python3
"""
=============================================================================
Stage 1: Multi-Task Activation Extraction
=============================================================================

Extract activation patterns from all 7 HCP cognitive tasks for each subject.
Uses the HCP-MMP 360 parcel parcellation.

Outputs:
    - activation_matrix.pkl: (subjects × tasks × parcels) activation data
    - task_contrasts.pkl: Contrast maps for each task
    - extraction_stats.json: Quality metrics and summary statistics
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
import pandas as pd
import nibabel as nib
from scipy import stats
from scipy.ndimage import gaussian_filter1d
import pickle
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from configs import config

# Ensure timestamped run directories are initialized
config.ensure_run_directories()

# Setup logging
logger = config.setup_logging('multi_task_extraction')

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_cifti_data(filepath):
    """Load CIFTI dtseries file and extract data matrix"""
    try:
        img = nib.load(filepath)
        data = img.get_fdata()
        return data
    except Exception as e:
        logger.warning(f"Failed to load {filepath}: {e}")
        return None


_PARCEL_INDEX_CACHE = {'map': None}


def _load_glasser_parcel_map():
    """Load and cache the Glasser-360 parcel-index vector (length 91282)."""
    if _PARCEL_INDEX_CACHE['map'] is not None:
        return _PARCEL_INDEX_CACHE['map']
    path = config.GLASSER_INDICES_CIFTI
    if not path.exists():
        raise FileNotFoundError(
            f"Glasser parcel index CIFTI not found at {path}. "
            "Run tools/download_atlas.py or restore data/_atlas/."
        )
    img = nib.load(str(path))
    data = img.get_fdata().flatten().astype(int)
    _PARCEL_INDEX_CACHE['map'] = data
    return data


def extract_parcel_timeseries(cifti_data, n_parcels=360):
    """
    Extract parcel-level timeseries from CIFTI data using the real
    HCP-MMP1.0 (Glasser-360) parcel index file bundled with the
    ColeAnticevicNetPartition release.

    Each of the 360 cortical parcels is assigned its own column; grayordinates
    labelled 0 (subcortical / medial wall / unassigned) are dropped. Output
    shape is (n_timepoints, n_parcels).
    """
    if cifti_data is None:
        return None

    parcel_map = _load_glasser_parcel_map()
    n_grayordinates = cifti_data.shape[1]
    if parcel_map.shape[0] != n_grayordinates:
        # Atlas and data grayordinate counts must match; otherwise we cannot
        # align. This typically indicates a resampling mismatch that must be
        # fixed upstream rather than silently hidden.
        raise ValueError(
            f"Atlas grayordinate count {parcel_map.shape[0]} != CIFTI {n_grayordinates}"
        )

    n_timepoints = cifti_data.shape[0]
    parcel_data = np.zeros((n_timepoints, n_parcels), dtype=np.float32)
    for p in range(1, n_parcels + 1):
        mask = parcel_map == p
        if mask.any():
            parcel_data[:, p - 1] = cifti_data[:, mask].mean(axis=1)
    return parcel_data


def load_task_evs(evs_dir, task):
    """Load event timing files for a task"""
    evs = {}
    evs_path = Path(evs_dir)
    
    if not evs_path.exists():
        return evs
    
    for ev_file in evs_path.glob("*.txt"):
        ev_name = ev_file.stem
        try:
            ev_data = np.loadtxt(ev_file)
            if ev_data.size > 0:
                evs[ev_name] = ev_data
        except:
            pass
    
    return evs


def compute_task_activation(timeseries, evs, tr=0.72, task=None):
    """
    Compute task-evoked activation via a condition-wise GLM, then form a
    canonical contrast defined by config.TASK_CONTRASTS[task].

    The contrast for each task is
        contrast = mean(betas over `positive` EVs) - mean(betas over `negative` EVs)
    if `negative` is non-empty, otherwise mean(betas over `positive` EVs).

    Returns a length-n_parcels contrast vector, or None if the contrast
    cannot be formed (e.g. task has no matching positive EVs).
    """
    if timeseries is None or len(evs) == 0:
        return None

    n_timepoints, n_parcels = timeseries.shape

    # Build one regressor per EV
    design_cols = []
    ev_names = []
    for ev_name, ev_data in evs.items():
        try:
            if ev_data is None:
                continue
            ev_data = np.atleast_2d(ev_data)
            if ev_data.size == 0:
                continue
            if ev_data.ndim == 1:
                if len(ev_data) >= 3:
                    ev_data = ev_data.reshape(-1, 3)
                else:
                    continue
            if ev_data.shape[0] == 0 or ev_data.shape[1] < 3:
                continue

            regressor = np.zeros(n_timepoints)
            for row in ev_data:
                if len(row) >= 3:
                    onset, duration, amplitude = row[0], row[1], row[2]
                    onset_tr = int(onset / tr)
                    duration_tr = max(1, int(duration / tr))
                    if onset_tr < n_timepoints:
                        end_tr = min(onset_tr + duration_tr, n_timepoints)
                        regressor[onset_tr:end_tr] = amplitude

            hrf = _create_hrf(tr)
            regressor = np.convolve(regressor, hrf)[:n_timepoints]

            # Skip zero/degenerate regressors that would destabilise OLS
            if np.std(regressor) < 1e-8:
                continue

            design_cols.append(regressor)
            ev_names.append(ev_name)
        except Exception:
            continue

    if len(design_cols) == 0:
        return None

    # Add intercept column
    design_cols.append(np.ones(n_timepoints))
    design_matrix = np.column_stack(design_cols)

    # Vectorised OLS: betas shape (n_ev + 1, n_parcels)
    try:
        beta_all, *_ = np.linalg.lstsq(design_matrix, timeseries, rcond=None)
    except Exception:
        return None
    betas = beta_all[:-1, :]  # drop intercept row

    # Pick positive / negative EVs according to TASK_CONTRASTS
    task_contrast = config.TASK_CONTRASTS.get(task) if task is not None else None
    if task_contrast is None:
        # Fallback: mean of all betas (old behaviour)
        return np.mean(betas, axis=0)

    def _match(ev_candidates, name):
        return any(c.lower() in name.lower() for c in ev_candidates)

    pos_idx = [i for i, n in enumerate(ev_names) if _match(task_contrast['positive'], n)]
    neg_idx = [i for i, n in enumerate(ev_names) if _match(task_contrast['negative'], n)]

    if not pos_idx:
        # No positive EVs matched -> cannot form contrast
        return None

    pos_mean = betas[pos_idx, :].mean(axis=0)
    if neg_idx:
        neg_mean = betas[neg_idx, :].mean(axis=0)
        return pos_mean - neg_mean
    return pos_mean


def _create_hrf(tr, duration=32):
    """Create a simple HRF (hemodynamic response function)"""
    t = np.arange(0, duration, tr)
    # Double gamma HRF approximation
    hrf = (t ** 5) * np.exp(-t) / 120 - (t ** 15) * np.exp(-t) / 1e12
    hrf = hrf / np.max(hrf)
    return hrf


def compute_mean_activation(timeseries):
    """Compute mean activation across time"""
    if timeseries is None:
        return None
    return np.mean(timeseries, axis=0)


def zscore_activation(activation):
    """Z-score normalize activation values"""
    if activation is None:
        return None
    return stats.zscore(activation)


# =============================================================================
# MAIN EXTRACTION FUNCTION
# =============================================================================

def extract_all_tasks():
    """Extract activation patterns from all tasks for all subjects"""
    
    logger.info("="*60)
    logger.info("Stage 1: Multi-Task Activation Extraction")
    logger.info("="*60)
    
    subjects = config.SUBJECTS
    tasks = config.TASK_ORDER
    n_parcels = config.PARCELLATION['n_parcels']
    
    logger.info(f"Subjects: {len(subjects)}")
    logger.info(f"Tasks: {tasks}")
    logger.info(f"Parcels: {n_parcels}")
    
    # Initialize storage
    activation_matrix = np.zeros((len(subjects), len(tasks), n_parcels))
    extraction_stats = {
        'subjects': subjects,
        'tasks': tasks,
        'n_parcels': n_parcels,
        'successful_extractions': 0,
        'failed_extractions': 0,
        'per_subject': {},
        'per_task': {task: {'mean': 0, 'std': 0, 'n_success': 0} for task in tasks},
    }
    
    # Process each subject
    for s_idx, subject in enumerate(tqdm(subjects, desc="Processing subjects")):
        subject_stats = {'tasks_extracted': [], 'tasks_failed': []}
        
        for t_idx, task in enumerate(tasks):
            activation = None
            
            # Try both runs (LR and RL) and average
            run_activations = []
            
            for run in ['LR', 'RL']:
                fmri_path = config.get_task_fmri_path(subject, task, run)
                evs_path = config.get_task_evs_path(subject, task, run)
                
                if fmri_path.exists():
                    # Load CIFTI data
                    cifti_data = load_cifti_data(fmri_path)
                    
                    if cifti_data is not None:
                        # Extract parcel timeseries
                        timeseries = extract_parcel_timeseries(cifti_data, n_parcels)
                        
                        if timeseries is not None:
                            # Load EVs and compute activation
                            evs = load_task_evs(evs_path, task)
                            
                            if evs:
                                act = compute_task_activation(timeseries, evs, task=task)
                            else:
                                act = compute_mean_activation(timeseries)
                            
                            if act is not None:
                                run_activations.append(act)
            
            # Average across runs. REAL DATA ONLY — no simulation fallback:
            # a failed extraction leaves the activation_matrix entry as NaN so
            # downstream stages can mask it explicitly. This prevents any
            # synthetic data from entering the analysis.
            if len(run_activations) > 0:
                activation = np.mean(run_activations, axis=0)
                activation = zscore_activation(activation)
                subject_stats['tasks_extracted'].append(task)
                extraction_stats['successful_extractions'] += 1
                extraction_stats['per_task'][task]['n_success'] += 1
                activation_matrix[s_idx, t_idx, :] = activation
            else:
                logger.warning(
                    f"REAL-DATA EXTRACTION FAILED for {subject}/{task} "
                    f"(no LR/RL run produced a valid activation); leaving "
                    f"NaN in activation_matrix[{s_idx}, {t_idx}]."
                )
                subject_stats['tasks_failed'].append(task)
                extraction_stats['failed_extractions'] += 1
                activation_matrix[s_idx, t_idx, :] = np.nan
        
        extraction_stats['per_subject'][subject] = subject_stats
    
    # Compute per-task statistics
    for t_idx, task in enumerate(tasks):
        task_data = activation_matrix[:, t_idx, :]
        extraction_stats['per_task'][task]['mean'] = float(np.mean(task_data))
        extraction_stats['per_task'][task]['std'] = float(np.std(task_data))
    
    # Save results
    logger.info("Saving results...")
    
    # Save activation matrix
    output_path = config.NEUROSCIENCE_DIR / "activation_matrix.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump({
            'data': activation_matrix,
            'subjects': subjects,
            'tasks': tasks,
            'n_parcels': n_parcels,
        }, f)
    logger.info(f"Saved: {output_path}")
    
    # Save extraction stats
    stats_path = config.NEUROSCIENCE_DIR / "extraction_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(extraction_stats, f, indent=2)
    logger.info(f"Saved: {stats_path}")
    
    # Create summary DataFrame
    summary_data = []
    for s_idx, subject in enumerate(subjects):
        for t_idx, task in enumerate(tasks):
            summary_data.append({
                'subject': subject,
                'task': task,
                'mean_activation': float(np.mean(activation_matrix[s_idx, t_idx, :])),
                'std_activation': float(np.std(activation_matrix[s_idx, t_idx, :])),
                'max_activation': float(np.max(activation_matrix[s_idx, t_idx, :])),
                'min_activation': float(np.min(activation_matrix[s_idx, t_idx, :])),
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = config.NEUROSCIENCE_DIR / "activation_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Saved: {summary_path}")
    
    # Print summary
    logger.info("="*60)
    logger.info("Extraction Summary")
    logger.info("="*60)
    logger.info(f"Total extractions: {len(subjects) * len(tasks)}")
    logger.info(f"Successful: {extraction_stats['successful_extractions']}")
    logger.info(f"Failed (left as NaN): {extraction_stats['failed_extractions']}")
    logger.info(f"Activation matrix shape: {activation_matrix.shape}")
    
    return activation_matrix, extraction_stats


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    activation_matrix, stats = extract_all_tasks()
    print("\n✅ Stage 1 completed: Multi-task activation extraction")
