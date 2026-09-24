"""
=============================================================================
Project Configuration: Routing-Level Signatures of Cognitive Efficiency
IEEE BIBM 2026 Submission
=============================================================================

"Sparse Routing in Biological and Artificial Neural Networks: 
Multi-Task Evidence for Efficient Information Processing"

This configuration file centralizes all paths, parameters, and constants
used throughout the analysis pipeline.
"""

import os
from pathlib import Path
from datetime import datetime

# =============================================================================
# PROJECT PATHS
# =============================================================================

# Base paths
# PROJECT_DIR is this project's root (one level up from configs/)
PROJECT_DIR = Path(__file__).parent.parent
# DATA_ROOT lives inside the project directory to keep data isolated per-project.
# Override via environment variable if needed:
DATA_ROOT = Path(os.environ.get('HCP_DATA_ROOT', str(PROJECT_DIR / "data")))

# =============================================================================
# TIMESTAMPED RUN DIRECTORY
# =============================================================================

# Global run timestamp - initialized once when config is first imported
_RUN_TIMESTAMP = None
_DIRECTORIES_INITIALIZED = False

def get_run_timestamp():
    """Get or create the run timestamp for this session"""
    global _RUN_TIMESTAMP
    if _RUN_TIMESTAMP is None:
        # Check if timestamp is passed via environment variable (for child processes)
        _RUN_TIMESTAMP = os.environ.get('PROJECT2_RUN_TIMESTAMP')
        if _RUN_TIMESTAMP is None:
            _RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    return _RUN_TIMESTAMP

def initialize_run_directories():
    """Initialize timestamped result directories for a new run"""
    global RESULTS_DIR, NEUROSCIENCE_DIR, MOE_ANALYSIS_DIR, COMPARISON_DIR, FIGURES_DIR
    global _DIRECTORIES_INITIALIZED
    
    timestamp = get_run_timestamp()
    
    # Create timestamped results directory
    RESULTS_DIR = PROJECT_DIR / "results" / f"run_{timestamp}"
    NEUROSCIENCE_DIR = RESULTS_DIR / "neuroscience"
    MOE_ANALYSIS_DIR = RESULTS_DIR / "moe_analysis"
    COMPARISON_DIR = RESULTS_DIR / "comparison"
    FIGURES_DIR = RESULTS_DIR / "figures"
    
    # Create directories
    for d in [NEUROSCIENCE_DIR, MOE_ANALYSIS_DIR, COMPARISON_DIR, FIGURES_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    
    _DIRECTORIES_INITIALIZED = True
    return RESULTS_DIR

def ensure_run_directories():
    """Ensure run directories are initialized (called by individual scripts)"""
    global _DIRECTORIES_INITIALIZED
    if not _DIRECTORIES_INITIALIZED:
        # Check if we have a timestamp from parent process
        if os.environ.get('PROJECT2_RUN_TIMESTAMP'):
            initialize_run_directories()
        else:
            # Running script standalone - create new timestamped directory
            initialize_run_directories()

# Output directories (default - will be overwritten by initialize_run_directories)
RESULTS_DIR = PROJECT_DIR / "results"
NEUROSCIENCE_DIR = RESULTS_DIR / "neuroscience"
MOE_ANALYSIS_DIR = RESULTS_DIR / "moe_analysis"
COMPARISON_DIR = RESULTS_DIR / "comparison"
FIGURES_DIR = RESULTS_DIR / "figures"
LOGS_DIR = PROJECT_DIR / "logs"
# MoE model weights. Shared read-only with the TST journal paper via
# models -> ../../moe_models. The BIBM paper's third contribution (Table II,
# cross-domain routing comparison) depends on these.
MODELS_DIR = PROJECT_DIR / "models" / "huggingface_cache"

# Create log directory (shared across runs)
for d in [LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# SUBJECT LIST
# =============================================================================

def get_subject_list():
    """Get list of all available HCP subjects in DATA_ROOT"""
    subjects = []
    if DATA_ROOT.exists():
        for item in DATA_ROOT.iterdir():
            if item.is_dir() and item.name.isdigit():
                subjects.append(item.name)
    else:
        import logging
        logging.getLogger(__name__).warning(
            f"DATA_ROOT not found: {DATA_ROOT}. "
            f"Set HCP_DATA_ROOT environment variable or ensure data directory exists. "
            f"Pipeline stages that require raw fMRI data will fail; "
            f"stages using pre-computed results (activation_matrix.pkl) will still work."
        )
    return sorted(subjects)

SUBJECTS = get_subject_list()

# =============================================================================
# HCP TASK DEFINITIONS
# =============================================================================

TASKS = {
    'WM': {
        'name': 'Working Memory',
        'cognitive_domain': 'Executive Function',
        'expected_networks': ['FPN', 'DAN'],
        'runs': ['LR', 'RL'],
        'contrasts': ['2bk_vs_0bk', '2bk_vs_baseline', '0bk_vs_baseline'],
    },
    'MOTOR': {
        'name': 'Motor',
        'cognitive_domain': 'Motor Control',
        'expected_networks': ['SMN'],
        'runs': ['LR', 'RL'],
        'contrasts': ['lh_vs_avg', 'rh_vs_avg', 'lf_vs_avg', 'rf_vs_avg', 't_vs_avg'],
    },
    'LANGUAGE': {
        'name': 'Language',
        'cognitive_domain': 'Language Processing',
        # Yeo-7 mapping: language-selective regions load mainly onto FPN
        # (temporo-parietal) and VAN. 'Language' and 'TPJ' are outside the
        # Yeo-7 set and have been removed from the expected list.
        'expected_networks': ['FPN', 'VAN'],
        'runs': ['LR', 'RL'],
        'contrasts': ['story_vs_math', 'math_vs_story'],
    },
    'SOCIAL': {
        'name': 'Social Cognition',
        'cognitive_domain': 'Theory of Mind',
        # Theory-of-mind studies implicate DMN; TPJ is part of DMN/VAN in
        # Yeo-7 and is dropped as a standalone label here.
        'expected_networks': ['DMN'],
        'runs': ['LR', 'RL'],
        'contrasts': ['mental_vs_rnd'],
    },
    'RELATIONAL': {
        'name': 'Relational Processing',
        'cognitive_domain': 'Reasoning',
        # Relational reasoning: FPN + DAN in Yeo-7; 'Parietal' is not a
        # Yeo-7 label and has been removed.
        'expected_networks': ['FPN', 'DAN'],
        'runs': ['LR', 'RL'],
        'contrasts': ['rel_vs_match'],
    },
    'EMOTION': {
        'name': 'Emotion',
        'cognitive_domain': 'Emotion Processing',
        # Emotion face-matching engages LIM (ventromedial PFC) in Yeo-7;
        # 'Amygdala' is subcortical and outside the cortical Yeo-7 set.
        'expected_networks': ['LIM'],
        'runs': ['LR', 'RL'],
        'contrasts': ['faces_vs_shapes'],
    },
    'GAMBLING': {
        'name': 'Gambling',
        'cognitive_domain': 'Decision Making',
        # Reward/decision: FPN in Yeo-7; 'Reward' is not a Yeo-7 label.
        'expected_networks': ['FPN'],
        'runs': ['LR', 'RL'],
        'contrasts': ['punish_vs_reward', 'reward_vs_punish'],
    },
}

TASK_ORDER = ['WM', 'MOTOR', 'LANGUAGE', 'SOCIAL', 'RELATIONAL', 'EMOTION', 'GAMBLING']


# =============================================================================
# CANONICAL TASK CONTRASTS (EV-level, matched to HCP EVs/*.txt filenames)
# =============================================================================
#
# For each task, `positive` and `negative` are lists of EV (regressor) names
# whose betas are averaged separately; the task-evoked activation is then
# mean(positive betas) - mean(negative betas). If `negative` is empty, the
# activation is mean(positive betas) (task vs. implicit baseline).
#
# EV names are matched case-insensitively and by substring. These substrings
# are aligned with the HCP S1200 EV files observed in the data directory.

TASK_CONTRASTS = {
    'WM': {
        # 2-back vs. 0-back across all stimulus types (body/faces/places/tools)
        'positive': ['2bk_body', '2bk_faces', '2bk_places', '2bk_tools'],
        'negative': ['0bk_body', '0bk_faces', '0bk_places', '0bk_tools'],
    },
    'MOTOR': {
        # All movement conditions vs. cue
        'positive': ['lf', 'rf', 'lh', 'rh', 't'],
        'negative': ['cue'],
    },
    'LANGUAGE': {
        # Story vs. math
        'positive': ['story'],
        'negative': ['math'],
    },
    'SOCIAL': {
        # Mental (theory-of-mind) vs. random-motion control
        'positive': ['mental'],
        'negative': ['rnd'],
    },
    'RELATIONAL': {
        # Relational reasoning vs. control matching
        'positive': ['relation'],
        'negative': ['match'],
    },
    'EMOTION': {
        # Fear faces vs. neutral shapes
        'positive': ['fear'],
        'negative': ['neut'],
    },
    'GAMBLING': {
        # Reward (win) vs. punishment (loss)
        'positive': ['win'],
        'negative': ['loss'],
    },
}

# =============================================================================
# BRAIN NETWORK DEFINITIONS (YEO 7 NETWORKS + EXTENDED)
# =============================================================================

NETWORKS = {
    'VIS': {
        'name': 'Visual',
        'color': '#781286',
        'description': 'Primary and secondary visual cortex',
    },
    'SMN': {
        'name': 'Somatomotor',
        'color': '#4682B4',
        'description': 'Primary motor and somatosensory cortex',
    },
    'DAN': {
        'name': 'Dorsal Attention',
        'color': '#00760E',
        'description': 'Top-down attention control',
    },
    'VAN': {
        'name': 'Ventral Attention / Salience',
        'color': '#C43AFA',
        'description': 'Salience detection and attention reorienting',
    },
    'LIM': {
        'name': 'Limbic',
        'color': '#DCF8A4',
        'description': 'Emotion and memory processing',
    },
    'FPN': {
        'name': 'Frontoparietal Control',
        'color': '#E69422',
        'description': 'Executive control and cognitive flexibility',
    },
    'DMN': {
        'name': 'Default Mode',
        'color': '#CD3E4E',
        'description': 'Self-referential and internally-directed cognition',
    },
}

NETWORK_ORDER = ['VIS', 'SMN', 'DAN', 'VAN', 'LIM', 'FPN', 'DMN']
NETWORK_COLORS = {net: info['color'] for net, info in NETWORKS.items()}

# =============================================================================
# ATLAS FILES (HCP-MMP1.0 Glasser 360 + Cole-Anticevic 12-network partition)
# =============================================================================
# Provided by the Cole-Anticevic Net Partition repository
# (https://github.com/ColeLab/ColeAnticevicNetPartition). The 360-parcel Glasser
# index file and the 12-network CIFTI dlabel together allow us to assign every
# grayordinate to a real HCP-MMP parcel, and every parcel to a network. We then
# collapse the 12 Cole-Anticevic networks to the 7 Yeo-like networks we use
# throughout the paper.

ATLAS_DIR = DATA_ROOT / "_atlas"
GLASSER_INDICES_CIFTI = ATLAS_DIR / "Glasser360Indices_LR.dscalar.nii"
COLE_NETWORK_CIFTI = ATLAS_DIR / "CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_netassignments_LR.dlabel.nii"

# Cole-Anticevic network id -> Yeo-7-like label
# The mapping groups each Cole-Anticevic network to the closest Yeo-7 network:
#   Visual1/Visual2       -> VIS
#   Somatomotor/Auditory  -> SMN
#   Dorsal-attention      -> DAN
#   Cingulo-Opercular/Language/Ventral-Multimodal -> VAN (ventral attention / salience / language)
#   Orbito-Affective      -> LIM
#   Frontoparietal/Posterior-Multimodal -> FPN
#   Default               -> DMN
COLE_TO_YEO7 = {
    1:  'VIS',   # Visual1
    2:  'VIS',   # Visual2
    3:  'SMN',   # Somatomotor
    4:  'VAN',   # Cingulo-Opercular
    5:  'DAN',   # Dorsal-attention
    6:  'VAN',   # Language
    7:  'FPN',   # Frontoparietal
    8:  'SMN',   # Auditory
    9:  'DMN',   # Default
    10: 'FPN',   # Posterior-Multimodal
    11: 'VAN',   # Ventral-Multimodal
    12: 'LIM',   # Orbito-Affective
}

# =============================================================================
# HCP-MMP PARCELLATION
# =============================================================================

PARCELLATION = {
    'name': 'HCP-MMP1.0',
    'n_parcels': 360,
    'n_parcels_per_hemisphere': 180,
}

# =============================================================================
# MOE MODEL CONFIGURATION
# =============================================================================

MOE_CONFIG = {
    'models': [
        {
            'name': 'google/switch-base-8',
            'short_name': 'Switch-8',
            'type': 'Switch Transformer',
            'n_experts': 8,
            'routing': 'top-1',
            'size': '~220M params',
            'year': 2021,
            'organization': 'Google',
        },
        {
            'name': 'deepseek-ai/deepseek-moe-16b-base',
            'short_name': 'DeepSeek-16B',
            'type': 'DeepSeek MoE',
            'n_experts': 64,  # 64 experts, top-6 routing
            'routing': 'top-6',
            'size': '~16B params',
            'year': 2024,
            'organization': 'DeepSeek',
        },
        {
            'name': 'Qwen/Qwen1.5-MoE-A2.7B',
            'short_name': 'Qwen-MoE',
            'type': 'Qwen MoE',
            'n_experts': 60,  # 60 experts, top-4 routing
            'routing': 'top-4',
            'size': '~14.3B total, 2.7B active',
            'year': 2024,
            'organization': 'Alibaba',
        },
    ],
    # Keep primary model for backward compatibility
    'primary_model': {
        'name': 'google/switch-base-8',
        'type': 'Switch Transformer',
        'n_experts': 8,
        'routing': 'top-1',
        'size': '~220M params',
    },
    'input_categories': [
        {'name': 'scientific', 'analog_task': 'RELATIONAL', 'description': 'Scientific reasoning text'},
        {'name': 'social', 'analog_task': 'SOCIAL', 'description': 'Social interaction dialogues'},
        {'name': 'emotional', 'analog_task': 'EMOTION', 'description': 'Emotional content'},
        {'name': 'procedural', 'analog_task': 'MOTOR', 'description': 'Step-by-step instructions'},
        {'name': 'narrative', 'analog_task': 'LANGUAGE', 'description': 'Story and narrative text'},
        {'name': 'mathematical', 'analog_task': 'WM', 'description': 'Math problems and logic'},
        {'name': 'risk_decision', 'analog_task': 'GAMBLING', 'description': 'Risk and decision scenarios'},
    ],
}

# =============================================================================
# ANALYSIS PARAMETERS
# =============================================================================

ANALYSIS_PARAMS = {
    # Sparsity analysis
    'sparsity': {
        'threshold_percentile': 95,  # Top 5% activation
        'gini_normalize': True,
    },
    
    # Efficiency grouping
    'grouping': {
        'method': 'median',  # 'median', 'tercile', 'kmeans'
        'efficiency_metric': 'composite',  # Based on WM task performance
    },
    
    # Functional connectivity
    'connectivity': {
        'method': 'correlation',
        'fisher_z': True,
        'threshold': 0.1,
    },
    
    # Modularity
    'modularity': {
        'algorithm': 'louvain',
        'resolution': 1.0,
        'n_iterations': 100,
    },
}

# =============================================================================
# STATISTICAL PARAMETERS
# =============================================================================

STATS_PARAMS = {
    'alpha': 0.05,
    'correction': 'fdr_bh',
    'n_permutations': 5000,
    'bootstrap_samples': 10000,
    'effect_size': 'cohens_d',
    'confidence_level': 0.95,
}

# =============================================================================
# VISUALIZATION PARAMETERS
# =============================================================================

# Unified Color Scheme
UNIFIED_COLORS = {
    # Group colors
    'HIGH_EFF': '#1a5276',        # Deep Blue - High Efficiency
    'LOW_EFF': '#c0392b',         # Deep Red - Low Efficiency
    'HIGH_EFF_LIGHT': '#5dade2',  # Light Blue
    'LOW_EFF_LIGHT': '#f1948a',   # Light Red
    
    # System colors
    'BRAIN': '#2ecc71',           # Green - Brain
    'AI': '#9b59b6',              # Purple - AI
    
    # Neutral
    'NEUTRAL': '#566573',         # Gray
    'ACCENT': '#f39c12',          # Gold
    
    # Sequential colormaps
    'SEQUENTIAL_CMAP': 'plasma',
    'DIVERGING_CMAP': 'RdBu_r',
    'CATEGORICAL_CMAP': 'Set2',
    'HEATMAP_CMAP': 'YlOrRd',
    'CORRELATION_CMAP': 'coolwarm',
}

# ColorBrewer colormaps
COLORMAPS = {
    'sequential': 'plasma',       # For sequential data
    'diverging': 'RdBu_r',        # For diverging data (centered at 0)
    'categorical': 'Set2',        # For categorical comparisons
    'correlation': 'coolwarm',    # For correlation matrices
    'heatmap': 'YlOrRd',          # For heatmaps
    'activation': 'viridis',      # For brain activation
    'routing': 'Spectral',        # For routing patterns
}

# Figure parameters
FIGURE_PARAMS = {
    'dpi': 300,
    'formats': ['pdf'],
    'font_family': 'Arial',
    'font_size': {
        'title': 16,
        'subtitle': 14,
        'label': 12,
        'tick': 10,
        'legend': 10,
        'annotation': 9,
    },
    'figsize': {
        'single': (8, 6),
        'double': (14, 6),
        'triple': (18, 6),
        'square': (10, 10),
        'large': (16, 12),
        'wide': (16, 8),
        'tall': (8, 14),
        'summary': (16, 20),
    },
    'style': {
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 1.2,
        'axes.labelpad': 10,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'legend.frameon': False,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.edgecolor': 'none',
    },
}

# =============================================================================
# FILE PATH FUNCTIONS
# =============================================================================

def get_task_fmri_path(subject, task, run='LR'):
    """Get path to task fMRI data (HCP MSMAll-aligned, ICA-FIX cleaned)."""
    return (DATA_ROOT / subject / "MNINonLinear" / "Results" /
            f"tfMRI_{task}_{run}" /
            f"tfMRI_{task}_{run}_Atlas_MSMAll.dtseries.nii")

def get_task_evs_path(subject, task, run='LR'):
    """Get path to task EVs directory"""
    return (DATA_ROOT / subject / "MNINonLinear" / "Results" /
            f"tfMRI_{task}_{run}" / "EVs")

def get_rest_fmri_path(subject, run='REST1_LR'):
    """Get path to resting-state fMRI data"""
    run_name = f"rfMRI_{run}"
    return (DATA_ROOT / subject / "MNINonLinear" / "Results" /
            run_name / f"{run_name}_Atlas_MSMAll.dtseries.nii")

def get_structural_path(subject, modality='T1w'):
    """Get path to structural MRI"""
    return (DATA_ROOT / subject / "MNINonLinear" / 
            f"{modality}_restore.nii.gz")

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

import logging

def setup_logging(name, level=logging.INFO):
    """Setup logging for a module"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # File handler
    fh = logging.FileHandler(LOGS_DIR / f"{name}.log")
    fh.setLevel(level)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_efficiency_groups(behavioral_data, method='median'):
    """
    Split subjects into high/low efficiency groups based on WM task performance
    """
    if method == 'median':
        median_val = behavioral_data['efficiency_score'].median()
        high_eff = behavioral_data[behavioral_data['efficiency_score'] >= median_val]['subject'].tolist()
        low_eff = behavioral_data[behavioral_data['efficiency_score'] < median_val]['subject'].tolist()
    elif method == 'tercile':
        terciles = behavioral_data['efficiency_score'].quantile([0.33, 0.67])
        high_eff = behavioral_data[behavioral_data['efficiency_score'] >= terciles[0.67]]['subject'].tolist()
        low_eff = behavioral_data[behavioral_data['efficiency_score'] <= terciles[0.33]]['subject'].tolist()
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return {'high': high_eff, 'low': low_eff}

def compute_gini_coefficient(x):
    """
    Compute Gini coefficient as a measure of sparsity
    0 = perfectly uniform, 1 = maximally sparse
    """
    import numpy as np
    x = np.abs(x)
    x = np.sort(x)
    n = len(x)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * x) / (n * np.sum(x))) - (n + 1) / n

def compute_selectivity_index(activation_matrix, axis=0):
    """
    Compute selectivity index for each region/task
    Higher = more selective (responds to fewer tasks/regions)
    """
    import numpy as np
    max_activation = np.max(activation_matrix, axis=axis)
    mean_activation = np.mean(activation_matrix, axis=axis)
    selectivity = (max_activation - mean_activation) / (max_activation + 1e-8)
    return selectivity


def apply_fdr_correction(comparison_dict, p_value_key='p_value',
                          alpha=None, method='fdr_bh'):
    """Apply FDR (Benjamini-Hochberg) correction to a dict of comparison results.

    Mutates each sub-dict to add 'p_value_fdr' and 'significant_fdr' fields.
    The family = all entries in `comparison_dict` that have `p_value_key`.

    Args:
        comparison_dict: {name: {'p_value': float, ...}, ...}
        p_value_key: key holding the raw p-value (default 'p_value')
        alpha: significance threshold (default STATS_PARAMS['alpha'])
        method: multipletests method (default 'fdr_bh')

    Returns:
        dict mapping name -> {'p_raw', 'p_fdr', 'significant_fdr'} for logging.
    """
    from statsmodels.stats.multitest import multipletests
    if alpha is None:
        alpha = STATS_PARAMS['alpha']

    names = []
    pvals = []
    for name, entry in comparison_dict.items():
        if isinstance(entry, dict) and p_value_key in entry:
            names.append(name)
            pvals.append(float(entry[p_value_key]))

    if not pvals:
        return {}

    reject, pvals_corrected, _, _ = multipletests(pvals, alpha=alpha, method=method)

    summary = {}
    for name, p_raw, p_fdr, rej in zip(names, pvals, pvals_corrected, reject):
        comparison_dict[name]['p_value_fdr'] = float(p_fdr)
        comparison_dict[name]['significant_fdr'] = bool(rej)
        comparison_dict[name]['fdr_method'] = method
        comparison_dict[name]['fdr_family_size'] = len(pvals)
        summary[name] = {'p_raw': p_raw, 'p_fdr': float(p_fdr),
                          'significant_fdr': bool(rej)}
    return summary

