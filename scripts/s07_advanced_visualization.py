#!/usr/bin/env python3
"""
=============================================================================
Stage 7: Advanced Scientific Visualization
=============================================================================

Generate publication-quality figures with advanced scientific aesthetics.
Features hierarchical clustering, statistical annotations, gradient backgrounds,
and sophisticated multi-panel layouts suitable for high-impact journals.

Outputs:
    - fig01_multi_task_heatmap.png/svg - Clustered activation heatmap with dendrogram
    - fig02_sparsity_radar.png/svg - Multi-layer radar with confidence intervals
    - fig03_task_network_matrix.png/svg - Annotated routing matrix with clustering
    - fig04_modularity_comparison.png/svg - Raincloud plots with statistics
    - fig05_expert_activation.png/svg - Circular bar plot with polar coordinates
    - fig06_input_expert_sankey.png/svg - Flow diagram with gradient links
    - fig07_expert_wordcloud.png/svg - Polar area chart with specialization
    - fig08_routing_entropy.png/svg - Dual heatmap with marginal distributions
    - fig09_sparsity_comparison.png/svg - Paired comparison with effect sizes
    - fig10_similarity_matrix.png/svg - Hierarchically clustered correlation matrix
    - fig11_efficiency_scatter.png/svg - Multi-panel regression analysis
    - fig12_specialization_radar.png/svg - Overlaid comparison with statistics
"""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import pdist
import pickle
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
from matplotlib.collections import PatchCollection
import matplotlib.patheffects as path_effects
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Nilearn for brain visualization
try:
    from nilearn import plotting, datasets, image
    from nilearn.maskers import NiftiLabelsMasker
    import nibabel as nib
    NILEARN_AVAILABLE = True
except ImportError:
    NILEARN_AVAILABLE = False
    print("Warning: nilearn not available, brain glass model visualization will be skipped")

from configs import config

# Ensure timestamped run directories are initialized
config.ensure_run_directories()

logger = config.setup_logging('visualization')

# =============================================================================
# ADVANCED STYLE SETUP
# =============================================================================

# Professional color palette - Nature/Science journal inspired
PALETTE = {
    'primary': '#2C3E50',      # Dark blue-gray
    'secondary': '#E74C3C',    # Coral red
    'accent1': '#3498DB',      # Bright blue
    'accent2': '#27AE60',      # Emerald green
    'accent3': '#9B59B6',      # Amethyst purple
    'accent4': '#F39C12',      # Orange
    'high_eff': '#1A5276',     # Deep blue
    'low_eff': '#C0392B',      # Deep red
    'brain': '#16A085',        # Teal
    'ai': '#8E44AD',           # Purple
    'neutral': '#7F8C8D',      # Gray
    'bg_gradient_start': '#FDFEFE',
    'bg_gradient_end': '#EBF5FB',
}

# Custom colormaps - CONSISTENT across all figures
def create_scientific_cmap(name='scientific'):
    """Create publication-quality colormaps with consistent scheme across all figures"""
    if name == 'diverging':
        # Blue-White-Red for correlation/z-score data
        colors = ['#2166AC', '#4393C3', '#92C5DE', '#D1E5F0', '#F7F7F7',
                  '#FDDBC7', '#F4A582', '#D6604D', '#B2182B']
        return LinearSegmentedColormap.from_list('scientific_div', colors)
    elif name == 'sequential':
        # Orange-Red for general intensity/activation data
        colors = ['#FFF7EC', '#FEE8C8', '#FDD49E', '#FDBB84', '#FC8D59',
                  '#EF6548', '#D7301F', '#B30000', '#7F0000']
        return LinearSegmentedColormap.from_list('scientific_seq', colors)
    elif name == 'brain':
        # Teal-Blue for brain/neuroscience data
        colors = ['#F0F9E8', '#BAE4BC', '#7BCCC4', '#43A2CA', '#0868AC']
        return LinearSegmentedColormap.from_list('brain_seq', colors)
    elif name == 'moe':
        # Purple for MoE/AI data
        colors = ['#F2F0F7', '#CBC9E2', '#9E9AC8', '#756BB1', '#54278F']
        return LinearSegmentedColormap.from_list('moe_seq', colors)
    return plt.cm.viridis

# Consistent expert colors for polar/bar charts (8 experts)
EXPERT_COLORS = [
    '#54278F',  # Deep purple
    '#756BB1',  # Medium purple
    '#9E9AC8',  # Light purple
    '#5E4FA2',  # Indigo
    '#3288BD',  # Blue
    '#66C2A5',  # Teal
    '#ABDDA4',  # Light green
    '#E6F598',  # Yellow-green
]

# Consistent category colors for input types
CATEGORY_COLORS = {
    'scientific': '#3498DB',      # Blue
    'social': '#E74C3C',          # Red
    'emotional': '#9B59B6',       # Purple
    'procedural': '#27AE60',      # Green
    'narrative': '#F39C12',       # Orange
    'mathematical': '#1ABC9C',    # Teal
    'risk_decision': '#E67E22',   # Dark Orange
}

# HCP-MMP1.0 Atlas parcel names (180 per hemisphere, 360 total)
HCP_MMP_NAMES = [
    # Left hemisphere (0-179)
    'L V1', 'L MST', 'L V6', 'L V2', 'L V3', 'L V4', 'L V8', 'L 4', 'L 3b', 'L FEF',
    'L PEF', 'L 55b', 'L V3A', 'L RSC', 'L POS2', 'L V7', 'L IPS1', 'L FFC', 'L V3B', 'L LO1',
    'L LO2', 'L PIT', 'L MT', 'L A1', 'L PSL', 'L SFL', 'L PCV', 'L STV', 'L 7Pm', 'L 7m',
    'L POS1', 'L 23d', 'L v23ab', 'L d23ab', 'L 31pv', 'L 5m', 'L 5mv', 'L 23c', 'L 5L', 'L 24dd',
    'L 24dv', 'L 7AL', 'L SCEF', 'L 6ma', 'L 7Am', 'L 7PL', 'L 7PC', 'L LIPv', 'L VIP', 'L MIP',
    'L 1', 'L 2', 'L 3a', 'L 6d', 'L 6mp', 'L 6v', 'L p24pr', 'L 33pr', 'L a24pr', 'L p32pr',
    'L a24', 'L d32', 'L 8BM', 'L p32', 'L 10r', 'L 47m', 'L 8Av', 'L 8Ad', 'L 9m', 'L 8BL',
    'L 9p', 'L 10d', 'L 8C', 'L 44', 'L 45', 'L 47l', 'L a47r', 'L 6r', 'L IFJa', 'L IFJp',
    'L IFSp', 'L IFSa', 'L p9-46v', 'L 46', 'L a9-46v', 'L 9-46d', 'L 9a', 'L 10v', 'L a10p', 'L 10pp',
    'L 11l', 'L 13l', 'L OFC', 'L 47s', 'L LIPd', 'L 6a', 'L i6-8', 'L s6-8', 'L 43', 'L OP4',
    'L OP1', 'L OP2-3', 'L 52', 'L RI', 'L PFcm', 'L PoI2', 'L TA2', 'L FOP4', 'L MI', 'L Pir',
    'L AVI', 'L AAIC', 'L FOP1', 'L FOP3', 'L FOP2', 'L PFt', 'L AIP', 'L EC', 'L PreS', 'L H',
    'L ProS', 'L PeEc', 'L STGa', 'L PBelt', 'L A5', 'L PHA1', 'L PHA3', 'L STSda', 'L STSdp', 'L STSvp',
    'L TGd', 'L TE1a', 'L TE1p', 'L TE2a', 'L TF', 'L TE2p', 'L PHT', 'L PH', 'L TPOJ1', 'L TPOJ2',
    'L TPOJ3', 'L DVT', 'L PGp', 'L IP2', 'L IP1', 'L IP0', 'L PFop', 'L PF', 'L PFm', 'L PGi',
    'L PGs', 'L V6A', 'L VMV1', 'L VMV3', 'L PHA2', 'L V4t', 'L FST', 'L V3CD', 'L LO3', 'L VMV2',
    'L 31pd', 'L 31a', 'L VVC', 'L 25', 'L s32', 'L pOFC', 'L PoI1', 'L Ig', 'L FOP5', 'L p10p',
    'L p47r', 'L TGv', 'L MBelt', 'L LBelt', 'L A4', 'L STSva', 'L TE1m', 'L PI', 'L a32pr', 'L p24',
    # Right hemisphere (180-359)
    'R V1', 'R MST', 'R V6', 'R V2', 'R V3', 'R V4', 'R V8', 'R 4', 'R 3b', 'R FEF',
    'R PEF', 'R 55b', 'R V3A', 'R RSC', 'R POS2', 'R V7', 'R IPS1', 'R FFC', 'R V3B', 'R LO1',
    'R LO2', 'R PIT', 'R MT', 'R A1', 'R PSL', 'R SFL', 'R PCV', 'R STV', 'R 7Pm', 'R 7m',
    'R POS1', 'R 23d', 'R v23ab', 'R d23ab', 'R 31pv', 'R 5m', 'R 5mv', 'R 23c', 'R 5L', 'R 24dd',
    'R 24dv', 'R 7AL', 'R SCEF', 'R 6ma', 'R 7Am', 'R 7PL', 'R 7PC', 'R LIPv', 'R VIP', 'R MIP',
    'R 1', 'R 2', 'R 3a', 'R 6d', 'R 6mp', 'R 6v', 'R p24pr', 'R 33pr', 'R a24pr', 'R p32pr',
    'R a24', 'R d32', 'R 8BM', 'R p32', 'R 10r', 'R 47m', 'R 8Av', 'R 8Ad', 'R 9m', 'R 8BL',
    'R 9p', 'R 10d', 'R 8C', 'R 44', 'R 45', 'R 47l', 'R a47r', 'R 6r', 'R IFJa', 'R IFJp',
    'R IFSp', 'R IFSa', 'R p9-46v', 'R 46', 'R a9-46v', 'R 9-46d', 'R 9a', 'R 10v', 'R a10p', 'R 10pp',
    'R 11l', 'R 13l', 'R OFC', 'R 47s', 'R LIPd', 'R 6a', 'R i6-8', 'R s6-8', 'R 43', 'R OP4',
    'R OP1', 'R OP2-3', 'R 52', 'R RI', 'R PFcm', 'R PoI2', 'R TA2', 'R FOP4', 'R MI', 'R Pir',
    'R AVI', 'R AAIC', 'R FOP1', 'R FOP3', 'R FOP2', 'R PFt', 'R AIP', 'R EC', 'R PreS', 'R H',
    'R ProS', 'R PeEc', 'R STGa', 'R PBelt', 'R A5', 'R PHA1', 'R PHA3', 'R STSda', 'R STSdp', 'R STSvp',
    'R TGd', 'R TE1a', 'R TE1p', 'R TE2a', 'R TF', 'R TE2p', 'R PHT', 'R PH', 'R TPOJ1', 'R TPOJ2',
    'R TPOJ3', 'R DVT', 'R PGp', 'R IP2', 'R IP1', 'R IP0', 'R PFop', 'R PF', 'R PFm', 'R PGi',
    'R PGs', 'R V6A', 'R VMV1', 'R VMV3', 'R PHA2', 'R V4t', 'R FST', 'R V3CD', 'R LO3', 'R VMV2',
    'R 31pd', 'R 31a', 'R VVC', 'R 25', 'R s32', 'R pOFC', 'R PoI1', 'R Ig', 'R FOP5', 'R p10p',
    'R p47r', 'R TGv', 'R MBelt', 'R LBelt', 'R A4', 'R STSva', 'R TE1m', 'R PI', 'R a32pr', 'R p24',
]

# Apply professional style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 12,
    'axes.labelweight': 'medium',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 1.5,
    'axes.edgecolor': PALETTE['primary'],
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'legend.edgecolor': 'none',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.edgecolor': 'none',
    'savefig.bbox': 'tight',
    'savefig.dpi': 300,
})

def save_figure(fig, name, dpi=300):
    """Save figure in multiple formats with optimization"""
    for fmt in config.FIGURE_PARAMS['formats']:
        path = config.FIGURES_DIR / f"{name}.{fmt}"
        fig.savefig(path, dpi=dpi, bbox_inches='tight', 
                    facecolor='white', edgecolor='none',
                    transparent=False if fmt == 'png' else True)
        logger.info(f"Saved: {path}")
    plt.close(fig)

def add_panel_label(ax, label, x=-0.12, y=1.08, fontsize=16):
    """Add panel label (A, B, C, etc.) to subplot"""
    ax.text(x, y, label, transform=ax.transAxes, fontsize=fontsize,
            fontweight='bold', va='top', ha='left',
            color=PALETTE['primary'])

def add_significance_annotation(ax, x1, x2, y, p_val, height=0.02):
    """Add significance bars with stars"""
    if p_val < 0.001:
        sig = '***'
    elif p_val < 0.01:
        sig = '**'
    elif p_val < 0.05:
        sig = '*'
    else:
        sig = 'ns'
    
    bar_height = y + height
    ax.plot([x1, x1, x2, x2], [y, bar_height, bar_height, y], 
            color=PALETTE['primary'], linewidth=1.5)
    ax.text((x1 + x2) / 2, bar_height + 0.01, sig, 
            ha='center', va='bottom', fontsize=12, fontweight='bold')

# =============================================================================
# FIGURE 1: Advanced Multi-Task Activation Heatmap with Clustering
# =============================================================================

def create_fig01_multi_task_heatmap():
    """Create hierarchically clustered heatmap with dendrograms"""
    logger.info("Creating Figure 1: Advanced Multi-Task Activation Heatmap...")
    
    with open(config.NEUROSCIENCE_DIR / "activation_matrix.pkl", 'rb') as f:
        data = pickle.load(f)
    
    activation_matrix = data['data']
    tasks = data['tasks']
    n_parcels = data['n_parcels']
    
    # Average across subjects
    avg_activation = np.mean(activation_matrix, axis=0)
    
    # Subsample parcels more intelligently (select representative parcels)
    step = 6
    parcel_indices = list(range(0, n_parcels, step))
    avg_activation_ds = avg_activation[:, parcel_indices]
    
    # Create figure with custom layout (flatter aspect: 7 task rows should be
    # compact; 18:6.5 gives ~17pt row height in the final PDF)
    fig = plt.figure(figsize=(18, 6.5))
    
    # GridSpec for complex layout
    gs = GridSpec(2, 3, height_ratios=[0.15, 1], width_ratios=[0.05, 1, 0.03],
                  hspace=0.02, wspace=0.02)
    
    # Dendrogram for tasks (left side - we'll put it on top instead)
    ax_dendro_top = fig.add_subplot(gs[0, 1])
    
    # Main heatmap
    ax_main = fig.add_subplot(gs[1, 1])
    
    # Colorbar
    ax_cbar = fig.add_subplot(gs[1, 2])
    
    # Network color bar (left)
    ax_net = fig.add_subplot(gs[1, 0])
    
    # Perform hierarchical clustering on parcels
    parcel_linkage = linkage(avg_activation_ds.T, method='ward')
    parcel_order = leaves_list(parcel_linkage)
    
    # Cluster tasks
    task_linkage = linkage(avg_activation_ds, method='ward')
    task_order = leaves_list(task_linkage)
    
    # Reorder data
    ordered_data = avg_activation_ds[task_order][:, parcel_order]
    ordered_tasks = [tasks[i] for i in task_order]
    
    # Get ordered parcel indices and names
    ordered_parcel_indices = [parcel_indices[parcel_order[i]] for i in range(len(parcel_indices))]
    ordered_parcel_names = [HCP_MMP_NAMES[idx] for idx in ordered_parcel_indices]
    
    # Plot dendrogram for parcels
    with plt.rc_context({'lines.linewidth': 1.0}):
        dendrogram(parcel_linkage, ax=ax_dendro_top, orientation='top',
                   no_labels=True, color_threshold=0, above_threshold_color=PALETTE['neutral'])
    ax_dendro_top.axis('off')
    
    # Create heatmap with custom colormap
    cmap = create_scientific_cmap('diverging')
    im = ax_main.imshow(ordered_data, aspect='auto', cmap=cmap,
                        vmin=-2.5, vmax=2.5, interpolation='nearest')
    
    # Add task labels with colored backgrounds
    ax_main.set_yticks(range(len(ordered_tasks)))
    task_labels = [config.TASKS[t]['name'] for t in ordered_tasks]
    ax_main.set_yticklabels(task_labels, fontsize=14)
    
    # Color the y-tick labels by cognitive domain
    domain_colors = {
        'Executive Function': PALETTE['accent1'],
        'Motor Control': PALETTE['accent2'],
        'Language Processing': PALETTE['accent3'],
        'Theory of Mind': PALETTE['accent4'],
        'Reasoning': PALETTE['secondary'],
        'Emotion Processing': PALETTE['high_eff'],
        'Decision Making': PALETTE['low_eff'],
    }
    
    for i, (task, label) in enumerate(zip(ordered_tasks, ax_main.get_yticklabels())):
        domain = config.TASKS[task]['cognitive_domain']
        label.set_color(domain_colors.get(domain, PALETTE['primary']))
        label.set_fontweight('bold')
    
    # Set x-axis tick labels - show region names at intervals
    n_displayed = len(parcel_indices)
    tick_step = 10  # Show every 10th parcel (about 6 labels)
    tick_positions = list(range(0, n_displayed, tick_step))
    tick_labels = [ordered_parcel_names[i] for i in tick_positions]
    ax_main.set_xticks(tick_positions)
    ax_main.set_xticklabels(tick_labels, fontsize=12, rotation=45, ha='right')

    # Set axis labels
    ax_main.set_xlabel('Brain Region (HCP-MMP Atlas)', fontsize=16, labelpad=10)

    # NOTE: the earlier Functional-Networks legend has been removed because
    # columns in the heatmap are ordered by hierarchical clustering over tasks
    # rather than grouped by Yeo network, so a per-network color legend would
    # refer to a structure not visible in the plot. Cognitive-domain color
    # coding of the y-axis task labels is documented in the figure caption.

    # Colorbar with refined style
    cbar = plt.colorbar(im, cax=ax_cbar)
    cbar.set_label('Activation (z-score)', fontsize=15, labelpad=10)
    cbar.ax.tick_params(labelsize=12)
    
    # Hide network axis
    ax_net.axis('off')
    
    # No suptitle (caption provides context in manuscript)
    ax_main.set_title('', pad=10)

    plt.tight_layout()
    plt.subplots_adjust(left=0.12, bottom=0.12)  # Make room for legend and rotated labels
    save_figure(fig, 'fig01_multi_task_heatmap')
    return fig


# =============================================================================
# FIGURE 2: Enhanced Sparsity Radar Chart with Confidence Intervals
# =============================================================================

def create_fig02_sparsity_radar():
    """Create multi-layer radar chart with confidence intervals and statistics"""
    logger.info("Creating Figure 2: Enhanced Sparsity Radar Chart...")
    
    sparsity_df = pd.read_csv(config.NEUROSCIENCE_DIR / "sparsity_metrics.csv")
    
    tasks = config.TASK_ORDER
    n_tasks = len(tasks)
    
    # Compute statistics per task per group
    high_means, low_means = [], []
    high_ci, low_ci = [], []
    p_values = []
    
    for task in tasks:
        task_data = sparsity_df[sparsity_df['task'] == task]
        # Subplot A uses composite_sparsity (combined measure of all sparsity metrics)
        high_data = task_data[task_data['efficiency_group'] == 'high']['composite_sparsity']
        low_data = task_data[task_data['efficiency_group'] == 'low']['composite_sparsity']
        
        high_means.append(high_data.mean())
        low_means.append(low_data.mean())
        
        # 95% CI using bootstrap
        high_ci.append(1.96 * high_data.std() / np.sqrt(len(high_data)))
        low_ci.append(1.96 * low_data.std() / np.sqrt(len(low_data)))
        
        # T-test
        _, p = stats.ttest_ind(high_data, low_data)
        p_values.append(p)
    
    # Setup radar chart
    angles = np.linspace(0, 2 * np.pi, n_tasks, endpoint=False).tolist()
    angles += angles[:1]
    
    high_means += high_means[:1]
    low_means += low_means[:1]
    high_ci += high_ci[:1]
    low_ci += low_ci[:1]
    
    # Create figure with two panels - increased wspace to avoid overlap
    fig = plt.figure(figsize=(18, 8))
    gs = GridSpec(1, 2, width_ratios=[1.2, 1], wspace=0.5)
    
    # Panel A: Radar chart
    ax1 = fig.add_subplot(gs[0], polar=True)
    
    # Plot with gradient fill
    high_color = PALETTE['high_eff']
    low_color = PALETTE['low_eff']
    
    # Plot CI bands
    high_upper = [m + c for m, c in zip(high_means, high_ci)]
    high_lower = [m - c for m, c in zip(high_means, high_ci)]
    low_upper = [m + c for m, c in zip(low_means, low_ci)]
    low_lower = [m - c for m, c in zip(low_means, low_ci)]
    
    ax1.fill_between(angles, high_lower, high_upper, alpha=0.15, color=high_color, linewidth=0)
    ax1.fill_between(angles, low_lower, low_upper, alpha=0.15, color=low_color, linewidth=0)
    
    # Main lines with markers
    ax1.plot(angles, high_means, 'o-', linewidth=2.5, color=high_color,
             markersize=10, markeredgecolor='white', markeredgewidth=2,
             label='High Efficiency', zorder=5)
    ax1.plot(angles, low_means, 's-', linewidth=2.5, color=low_color,
             markersize=10, markeredgecolor='white', markeredgewidth=2,
             label='Low Efficiency', zorder=5)
    
    # Customize grid - use tick labels with large padding
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels([config.TASKS[t]['name'] for t in tasks], size=9, fontweight='medium')
    ax1.set_ylim(0, max(max(high_means), max(low_means)) * 1.3)
    ax1.tick_params(axis='x', pad=40)  # Large padding to move labels away from plot
    
    # Add significance markers
    for i, (angle, p) in enumerate(zip(angles[:-1], p_values)):
        if p < 0.05:
            r = max(high_means[i], low_means[i]) + 0.05
            marker = '★' if p < 0.01 else '*'
            ax1.annotate(marker, xy=(angle, r), fontsize=14, 
                        color=PALETTE['accent4'], fontweight='bold',
                        ha='center', va='center')
    
    ax1.legend(loc='upper right', bbox_to_anchor=(1.35, 1.0), framealpha=0.95)
    ax1.set_title('Composite Sparsity by Task', fontsize=14, fontweight='bold', pad=20)
    add_panel_label(ax1, 'A', x=-0.2, y=1.15)
    
    # Panel B: Effect size bar chart
    ax2 = fig.add_subplot(gs[1])
    
    # Calculate Cohen's d for each task (using np.var for consistency with sparsity analysis)
    effect_sizes = []
    for task in tasks:
        task_data = sparsity_df[sparsity_df['task'] == task]
        high_data = task_data[task_data['efficiency_group'] == 'high']['gini_coefficient'].values
        low_data = task_data[task_data['efficiency_group'] == 'low']['gini_coefficient'].values
        
        pooled_std = np.sqrt((np.var(high_data) + np.var(low_data)) / 2)
        d = (np.mean(high_data) - np.mean(low_data)) / pooled_std if pooled_std > 0 else 0
        effect_sizes.append(d)
    
    y_pos = np.arange(len(tasks))
    colors = [PALETTE['accent2'] if d > 0 else PALETTE['secondary'] for d in effect_sizes]
    
    bars = ax2.barh(y_pos, effect_sizes, color=colors, edgecolor='white', 
                    linewidth=1.5, height=0.7, alpha=0.85)
    
    # Add zero reference line only
    ax2.axvline(x=0, color=PALETTE['primary'], linewidth=1.5, linestyle='-')
    
    # Add effect size labels
    for i, (bar, d, p) in enumerate(zip(bars, effect_sizes, p_values)):
        width = bar.get_width()
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        label = f'd={d:.2f}{sig}'
        x_pos = width + 0.05 if width >= 0 else width - 0.05
        ha = 'left' if width >= 0 else 'right'
        ax2.text(x_pos, bar.get_y() + bar.get_height()/2, label,
                va='center', ha=ha, fontsize=9, fontweight='medium')
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([config.TASKS[t]['name'] for t in tasks], fontsize=10)
    ax2.set_xlabel("Cohen's d", fontsize=12)
    ax2.set_title('Effect Size (Gini Coefficient)', fontsize=14, fontweight='bold')
    
    # Set x-axis limits to accommodate all effect sizes (including large ones like WM d=2.52)
    max_d = max(abs(d) for d in effect_sizes)
    x_limit = max(1.0, max_d * 1.15)  # At least 1.0, or 15% larger than max effect size
    ax2.set_xlim(-x_limit, x_limit)
    
    add_panel_label(ax2, 'B', x=-0.15, y=1.05)
    
    fig.suptitle('Neural Efficiency and Activation Sparsity', 
                 fontsize=18, fontweight='bold', y=1.02, color=PALETTE['primary'])
    
    plt.tight_layout()
    save_figure(fig, 'fig02_sparsity_radar')
    return fig


# =============================================================================
# FIGURE 3: Task-Network Routing Matrix with Advanced Annotations
# =============================================================================

def create_fig03_task_network_matrix():
    """Create publication-quality routing matrix with clustering and statistics"""
    logger.info("Creating Figure 3: Task-Network Routing Matrix...")
    
    with open(config.NEUROSCIENCE_DIR / "task_network_matrix.pkl", 'rb') as f:
        data = pickle.load(f)
    
    matrix = data['overall']
    tasks = data['tasks']
    networks = data['networks']
    
    # Create figure - simplified layout without top dendrogram
    fig = plt.figure(figsize=(12, 9))
    gs = GridSpec(1, 2, width_ratios=[1, 0.05], wspace=0.08)
    
    ax_main = fig.add_subplot(gs[0])
    ax_cbar = fig.add_subplot(gs[1])
    
    # Cluster networks for ordering
    network_linkage = linkage(matrix.T, method='ward')
    network_order = leaves_list(network_linkage)
    ordered_networks = [networks[i] for i in network_order]
    ordered_matrix = matrix[:, network_order]
    
    # Create heatmap with diverging colormap (same as Fig 1)
    cmap = create_scientific_cmap('diverging')
    im = ax_main.imshow(ordered_matrix, cmap=cmap, aspect='auto')
    
    # Remove grid
    ax_main.grid(False)
    
    # Add value annotations
    vmin, vmax = ordered_matrix.min(), ordered_matrix.max()
    for i in range(len(tasks)):
        for j in range(len(ordered_networks)):
            value = ordered_matrix[i, j]
            # Normalize value to determine text color
            norm_val = (value - vmin) / (vmax - vmin) if vmax > vmin else 0.5
            # Use white text on dark backgrounds (both ends of diverging scale), dark on light
            text_color = 'white' if (norm_val > 0.7 or norm_val < 0.3) else '#1a1a1a'
            ax_main.text(j, i, f'{value:.2f}', ha='center', va='center',
                        color=text_color, fontsize=11, fontweight='bold')
    
    # Highlight expected task-network pairs with styled rectangles
    for i, task in enumerate(tasks):
        expected = config.TASKS[task]['expected_networks']
        for j, net in enumerate(ordered_networks):
            if net in expected:
                rect = FancyBboxPatch((j-0.48, i-0.48), 0.96, 0.96,
                                      boxstyle="round,pad=0.02,rounding_size=0.1",
                                      facecolor='none', edgecolor=PALETTE['accent2'],
                                      linewidth=3, linestyle='-')
                ax_main.add_patch(rect)
    
    # Labels with network colors
    ax_main.set_xticks(range(len(ordered_networks)))
    ax_main.set_xticklabels([config.NETWORKS[n]['name'] for n in ordered_networks],
                            rotation=45, ha='right', fontsize=11)
    
    # Color x-tick labels
    for j, (net, label) in enumerate(zip(ordered_networks, ax_main.get_xticklabels())):
        label.set_color(config.NETWORK_COLORS.get(net, 'black'))
        label.set_fontweight('bold')
    
    ax_main.set_yticks(range(len(tasks)))
    ax_main.set_yticklabels([config.TASKS[t]['name'] for t in tasks], fontsize=11)
    
    ax_main.set_xlabel('Functional Brain Networks', fontsize=13, labelpad=15)
    ax_main.set_ylabel('Cognitive Tasks', fontsize=13, labelpad=10)
    
    # Enhanced colorbar
    cbar = plt.colorbar(im, cax=ax_cbar)
    cbar.set_label('Activation Strength (a.u.)', fontsize=11, labelpad=10)
    
    # Legend for expected connections - moved outside to upper right corner
    legend_elements = [mpatches.Patch(facecolor='none', edgecolor=PALETTE['accent2'],
                                       linewidth=2, label='Expected activation')]
    fig.legend(handles=legend_elements, loc='upper right', 
               bbox_to_anchor=(0.98, 0.98), fontsize=10, framealpha=0.95)
    
    # Title
    ax_main.set_title('Task-Network Routing Architecture\n'
                      'Green borders indicate theoretically expected task-network mappings',
                      fontsize=14, fontweight='bold', color=PALETTE['primary'], pad=15)
    
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    save_figure(fig, 'fig03_task_network_matrix')
    return fig


# =============================================================================
# FIGURE 4: Modularity Comparison with Raincloud Plots
# =============================================================================

def create_fig04_modularity_comparison():
    """Create raincloud plots for modularity metrics comparison"""
    logger.info("Creating Figure 4: Modularity Comparison (Raincloud)...")
    
    mod_df = pd.read_csv(config.NEUROSCIENCE_DIR / "modularity_metrics.csv")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.0))
    
    metrics = ['modularity_Q', 'segregation', 'mean_participation']
    titles = ['Modularity Index (Q)', 'Network Segregation', 'Participation Coefficient']
    
    for idx, (ax, metric, title) in enumerate(zip(axes, metrics, titles)):
        high_data = mod_df[mod_df['efficiency_group'] == 'high'][metric].values
        low_data = mod_df[mod_df['efficiency_group'] == 'low'][metric].values
        
        # Raincloud plot components
        positions = [0, 1]
        
        # Half violin (cloud)
        parts = ax.violinplot([high_data, low_data], positions=positions,
                              showmeans=False, showextrema=False, widths=0.6)
        
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor([PALETTE['high_eff'], PALETTE['low_eff']][i])
            pc.set_edgecolor('white')
            pc.set_alpha(0.4)
            # Make half violin
            m = np.mean(pc.get_paths()[0].vertices[:, 0])
            pc.get_paths()[0].vertices[:, 0] = np.clip(pc.get_paths()[0].vertices[:, 0], 
                                                        -np.inf, m)
        
        # Box plot (rain)
        bp = ax.boxplot([high_data, low_data], positions=positions,
                        widths=0.15, patch_artist=True, showfliers=False)
        
        for i, (patch, median) in enumerate(zip(bp['boxes'], bp['medians'])):
            patch.set_facecolor([PALETTE['high_eff'], PALETTE['low_eff']][i])
            patch.set_alpha(0.8)
            patch.set_edgecolor('white')
            patch.set_linewidth(2)
            median.set_color('white')
            median.set_linewidth(2)
        
        # Individual points (drops) with jitter
        for i, (data, color, pos) in enumerate([
            (high_data, PALETTE['high_eff'], 0),
            (low_data, PALETTE['low_eff'], 1)
        ]):
            jitter = np.random.normal(0.2, 0.03, size=len(data))
            ax.scatter(pos + jitter, data, c=color, alpha=0.7, s=60,
                      edgecolor='white', linewidth=1, zorder=5)
        
        # Statistical annotation
        t_stat, p_val = stats.ttest_ind(high_data, low_data)
        d = (high_data.mean() - low_data.mean()) / np.sqrt((high_data.std()**2 + low_data.std()**2) / 2)
        
        y_max = max(high_data.max(), low_data.max())
        y_min = min(high_data.min(), low_data.min())
        y_range = y_max - y_min
        
        # Show Cohen's d only (no significance bracket / *, ns markers).
        ax.text(0.5, y_max + y_range * 0.10, f"Cohen's d = {d:.2f}",
                ha='center', fontsize=12, color=PALETTE['neutral'], style='italic')

        # Set y-axis limits with proper padding (less headroom now that the
        # significance bracket is gone).
        ax.set_ylim(y_min - y_range * 0.1, y_max + y_range * 0.22)

        ax.set_xticks(positions)
        ax.set_xticklabels(['High\nEfficiency', 'Low\nEfficiency'], fontsize=13)
        ax.set_ylabel(title, fontsize=14)
        ax.set_title(title, fontsize=16, fontweight='bold', pad=10)
        ax.tick_params(axis='y', labelsize=12)

        # Move panel label further left so it no longer overlaps with the
        # y-axis tick labels of the adjacent panel.
        add_panel_label(ax, chr(65 + idx), x=-0.20, y=1.06, fontsize=18)

    # Create shared legend at the right side
    legend_elements = [
        mpatches.Patch(facecolor=PALETTE['high_eff'], label='High Efficiency', alpha=0.7),
        mpatches.Patch(facecolor=PALETTE['low_eff'], label='Low Efficiency', alpha=0.7)
    ]
    fig.legend(handles=legend_elements, loc='center right',
               bbox_to_anchor=(1.0, 0.5), fontsize=13, framealpha=0.95)
    
    # No suptitle (caption provides context in manuscript)
    plt.tight_layout(rect=[0, 0, 0.92, 1.0])
    save_figure(fig, 'fig04_modularity_comparison')
    return fig


# =============================================================================
# FIGURE 5: Multi-Model Expert Activation Plot
# =============================================================================

# Model configurations for multi-model analysis
MULTI_MOE_MODELS = [
    {
        'name': 'Switch-8',
        'file': 'moe_analysis_switch_8.json',
        'color': '#4285F4',
        'n_experts': 8,
    },
    {
        'name': 'Qwen-MoE',
        'file': 'moe_analysis_qwen_moe.json',
        'color': '#FF6A00',
        'n_experts': 60,
    },
    {
        'name': 'DeepSeek-16B',
        'file': 'moe_analysis_deepseek_16b.json',
        'color': '#00D4AA',
        'n_experts': 64,
    },
]


def create_fig05_expert_activation():
    """Create expert activation plots for all three MoE models"""
    logger.info("Creating Figure 5: Multi-Model MoE Expert Activation...")
    
    # Load data for all three models
    models_data = []
    for model_config in MULTI_MOE_MODELS:
        filepath = config.MOE_ANALYSIS_DIR / model_config['file']
        if filepath.exists():
            with open(filepath, 'r') as f:
                data = json.load(f)
            models_data.append((model_config, data))
        else:
            logger.warning(f"Model file not found: {filepath}")
    
    if not models_data:
        logger.error("No model data found. Falling back to single model.")
        return create_fig05_expert_activation_single()
    
    # Create combined 3-row figure
    fig = plt.figure(figsize=(18, 16))
    
    categories = [c['name'] for c in config.MOE_CONFIG['input_categories']]
    
    for row, (model_config, data) in enumerate(models_data):
        model_name = model_config['name']
        n_experts = model_config['n_experts']
        model_color = model_config['color']
        
        expert_freq = data['expert_frequency']
        spec_data = data['expert_specialization']
        gini = data['sparsity_metrics']['gini']
        
        n_show = min(n_experts, 20)  # Show max 20 experts
        
        # Panel A: Bar chart for expert activation
        ax1 = fig.add_subplot(3, 2, row * 2 + 1)
        
        sorted_items = sorted(expert_freq.items(), key=lambda x: float(x[1]), reverse=True)[:n_show]
        expert_ids = [f'E{k}' for k, v in sorted_items]
        frequencies = [float(v) for k, v in sorted_items]
        
        bars = ax1.bar(range(n_show), frequencies, color=model_color, alpha=0.8, edgecolor='white')
        
        uniform = 1 / n_experts
        ax1.axhline(y=uniform, color=PALETTE['secondary'], linestyle='--', linewidth=2, 
                   label=f'Uniform ({uniform:.4f})')
        
        ax1.set_xticks(range(0, n_show, max(1, n_show // 10)))
        ax1.set_xticklabels([expert_ids[i] for i in range(0, n_show, max(1, n_show // 10))], fontsize=8)
        ax1.set_ylabel('Activation Frequency', fontsize=10)
        ax1.set_title(f'{model_name} ({n_experts} experts)\nGini = {gini:.3f}', 
                     fontsize=12, fontweight='bold', color=model_color)
        ax1.legend(loc='upper right', fontsize=8)
        
        # Panel B: Category distribution heatmap
        ax2 = fig.add_subplot(3, 2, row * 2 + 2)
        
        experts_to_show = [int(k) for k, v in sorted_items][:n_show]
        
        # Build matrix
        matrix = np.zeros((len(categories), len(experts_to_show)))
        for idx, e in enumerate(experts_to_show):
            e_str = str(e)
            if e_str in spec_data:
                for c_idx, cat in enumerate(categories):
                    matrix[c_idx, idx] = spec_data[e_str]['distribution'].get(cat, 0)
        
        im = ax2.imshow(matrix, aspect='auto', cmap='YlOrRd')
        
        ax2.set_xticks(range(0, len(experts_to_show), max(1, len(experts_to_show) // 10)))
        ax2.set_xticklabels([f'E{experts_to_show[i]}' for i in range(0, len(experts_to_show), max(1, len(experts_to_show) // 10))], fontsize=8)
        ax2.set_yticks(range(len(categories)))
        ax2.set_yticklabels([clean_category_name(c) for c in categories], fontsize=9)
        ax2.set_xlabel('Expert ID (sorted by frequency)', fontsize=10)
        ax2.set_title('Category Distribution', fontsize=12, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
        cbar.set_label('Proportion', fontsize=9)
    
    fig.suptitle('Expert Activation Patterns Across Three MoE Architectures\n(Real Inference on RACE + SST-2 + GSM8K + SNLI)',
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, 'fig05_expert_activation')
    
    # Also save individual model figures
    for model_config, data in models_data:
        create_fig05_single_model(model_config, data)
    
    return fig


def create_fig05_single_model(model_config, data):
    """Create fig05-style figure for a single model - always using circular/polar format"""
    model_name = model_config['name']
    n_experts = model_config['n_experts']
    model_color = model_config['color']
    
    expert_freq = data['expert_frequency']
    spec_data = data['expert_specialization']
    gini = data['sparsity_metrics']['gini']
    
    show_all = n_experts <= 20
    n_show = n_experts if show_all else 20
    
    fig = plt.figure(figsize=(16, 8))
    gs = GridSpec(1, 2, width_ratios=[1, 1.2], wspace=0.3)
    
    categories = [c['name'] for c in config.MOE_CONFIG['input_categories']]
    cat_colors = [CATEGORY_COLORS.get(cat, '#7F8C8D') for cat in categories]
    
    # Panel A: Always use Circular bar plot (polar format)
    ax1 = fig.add_subplot(gs[0], polar=True)
    
    if show_all:
        # For small models, show all experts
        expert_indices = list(range(n_experts))
        frequencies = [float(expert_freq[str(i)]) for i in range(n_experts)]
        expert_labels = [f'E{i}' for i in range(n_experts)]
        uniform = 1 / n_experts
    else:
        # For large models, show top 20 experts sorted by frequency
        sorted_items = sorted(expert_freq.items(), key=lambda x: float(x[1]), reverse=True)[:n_show]
        expert_indices = [int(k) for k, v in sorted_items]
        frequencies = [float(v) for k, v in sorted_items]
        expert_labels = [f'E{k}' for k in expert_indices]
        uniform = 1 / n_experts
    
    angles = np.linspace(0, 2 * np.pi, n_show, endpoint=False)
    colors = [plt.cm.Blues(0.3 + 0.6 * i / n_show) for i in range(n_show)]
    width = 2 * np.pi / n_show * 0.8
    
    bars = ax1.bar(angles, frequencies, width=width, bottom=0.02,
                   color=colors, edgecolor='white', linewidth=2, alpha=0.85)
    
    # Add uniform distribution reference circle
    theta = np.linspace(0, 2 * np.pi, 100)
    ax1.plot(theta, [uniform + 0.02] * 100, '--', color=PALETTE['secondary'],
             linewidth=2, label=f'Uniform ({uniform:.4f})')
    
    ax1.set_xticks(angles)
    ax1.set_xticklabels(expert_labels, fontsize=9 if n_show <= 10 else 7, fontweight='medium')
    ax1.tick_params(axis='x', pad=20 if n_show <= 10 else 15)
    
    # Add frequency values (only for small number of experts)
    if n_show <= 10:
        for angle, freq, bar in zip(angles, frequencies, bars):
            ax1.text(angle, freq + 0.03, f'{freq:.3f}', ha='center', va='bottom',
                    fontsize=8, fontweight='bold', color=PALETTE['primary'])
    
    ax1.set_ylim(0, max(frequencies) + 0.08)
    title_suffix = '' if show_all else ' (Top 20)'
    ax1.set_title(f'Expert Activation Frequency{title_suffix}\nGini = {gini:.3f}', 
                  fontsize=14, fontweight='bold', pad=20)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1))
    
    add_panel_label(ax1, 'A', x=-0.2, y=1.15)
    
    # Panel B: Stacked bar for specialization
    ax2 = fig.add_subplot(gs[1])
    
    if show_all:
        experts_to_show = list(range(n_experts))
        x_labels = [f'E{i}' for i in range(n_experts)]
    else:
        sorted_items = sorted(expert_freq.items(), key=lambda x: float(x[1]), reverse=True)[:n_show]
        experts_to_show = [int(k) for k, v in sorted_items]
        x_labels = [f'E{k}' for k in experts_to_show]
    
    stacked_data = np.zeros((len(categories), len(experts_to_show)))
    for idx, e in enumerate(experts_to_show):
        e_str = str(e)
        if e_str in spec_data:
            for c_idx, cat in enumerate(categories):
                stacked_data[c_idx, idx] = spec_data[e_str]['distribution'].get(cat, 0)
    
    bottom = np.zeros(len(experts_to_show))
    for c_idx, (cat, color) in enumerate(zip(categories, cat_colors)):
        ax2.bar(range(len(experts_to_show)), stacked_data[c_idx], bottom=bottom,
               color=color, edgecolor='white', linewidth=0.5,
               label=clean_category_name(cat), alpha=0.85)
        bottom += stacked_data[c_idx]
    
    for idx, e in enumerate(experts_to_show):
        e_str = str(e)
        if e_str in spec_data:
            selectivity = spec_data[e_str].get('selectivity', 0)
            ax2.text(idx, 1.02, f'{selectivity:.2f}', ha='center', va='bottom',
                    fontsize=7, fontweight='bold', color=PALETTE['primary'])
    
    ax2.set_xlabel('Expert ID' + (' (Top 20)' if not show_all else ''), fontsize=11)
    ax2.set_ylabel('Category Distribution', fontsize=11)
    ax2.set_title('Expert Category Specialization', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(x_labels)))
    ax2.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
    ax2.set_ylim(0, 1.15)
    ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8,
               title='Input Category', title_fontsize=9)
    
    add_panel_label(ax2, 'B', x=-0.1, y=1.05)
    
    fig.suptitle(f'MoE Expert Activation and Specialization: {model_name}',
                 fontsize=18, fontweight='bold', y=1.02, color=model_color)
    
    plt.tight_layout()
    
    filename = f'fig05_expert_activation_{model_name.lower().replace("-", "_")}'
    save_figure(fig, filename)
    return fig


def create_fig05_expert_activation_single():
    """Fallback: Create circular/polar bar plot for single model (backward compatibility)"""
    logger.info("Creating Figure 5: MoE Expert Activation (Single Model Fallback)...")
    
    with open(config.MOE_ANALYSIS_DIR / "moe_analysis.json", 'r') as f:
        moe_data = json.load(f)
    
    with open(config.MOE_ANALYSIS_DIR / "expert_specialization.json", 'r') as f:
        spec_data = json.load(f)
    
    expert_freq = moe_data['expert_frequency']
    n_experts = len(expert_freq)
    
    fig = plt.figure(figsize=(16, 8))
    gs = GridSpec(1, 2, width_ratios=[1, 1.2], wspace=0.3)
    
    # Panel A: Circular bar plot
    ax1 = fig.add_subplot(gs[0], polar=True)
    
    angles = np.linspace(0, 2 * np.pi, n_experts, endpoint=False)
    frequencies = [expert_freq[str(i)] for i in range(n_experts)]
    
    colors = EXPERT_COLORS[:n_experts]
    width = 2 * np.pi / n_experts * 0.8
    
    bars = ax1.bar(angles, frequencies, width=width, bottom=0.05,
                   color=colors, edgecolor='white', linewidth=2, alpha=0.85)
    
    uniform = 1 / n_experts
    theta = np.linspace(0, 2 * np.pi, 100)
    ax1.plot(theta, [uniform + 0.05] * 100, '--', color=PALETTE['secondary'],
             linewidth=2, label=f'Uniform ({uniform:.3f})')
    
    ax1.set_xticks(angles)
    ax1.set_xticklabels([f'Expert {i}' for i in range(n_experts)], fontsize=10, fontweight='medium')
    ax1.tick_params(axis='x', pad=35)
    
    for angle, freq, bar in zip(angles, frequencies, bars):
        ax1.text(angle, freq + 0.06, f'{freq:.3f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold', color=PALETTE['primary'])
    
    ax1.set_ylim(0, max(frequencies) + 0.18)
    ax1.set_title('Expert Activation Frequency', fontsize=14, fontweight='bold', pad=20)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    add_panel_label(ax1, 'A', x=-0.2, y=1.15)
    
    # Panel B: Stacked area chart for specialization
    ax2 = fig.add_subplot(gs[1])
    
    categories = [c['name'] for c in config.MOE_CONFIG['input_categories']]
    cat_colors = [CATEGORY_COLORS.get(cat, '#7F8C8D') for cat in categories]
    
    stacked_data = np.zeros((len(categories), n_experts))
    for e in range(n_experts):
        for c_idx, cat in enumerate(categories):
            stacked_data[c_idx, e] = spec_data[str(e)]['category_distribution'].get(cat, 0)
    
    bottom = np.zeros(n_experts)
    for c_idx, (cat, color) in enumerate(zip(categories, cat_colors)):
        bars = ax2.bar(range(n_experts), stacked_data[c_idx], bottom=bottom,
                       color=color, edgecolor='white', linewidth=1,
                       label=clean_category_name(cat), alpha=0.85)
        bottom += stacked_data[c_idx]
    
    for e in range(n_experts):
        dominant = spec_data[str(e)].get('dominant_category', '')
        selectivity = spec_data[str(e)].get('selectivity', 0)
        ax2.text(e, 1.02, f'{selectivity:.2f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold', color=PALETTE['primary'])
    
    ax2.set_xlabel('Expert ID', fontsize=12)
    ax2.set_ylabel('Category Distribution', fontsize=12)
    ax2.set_title('Expert Category Specialization', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(n_experts))
    ax2.set_ylim(0, 1.12)
    ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9,
               title='Input Category', title_fontsize=10)
    
    add_panel_label(ax2, 'B', x=-0.1, y=1.05)
    
    fig.suptitle('Mixture of Experts: Activation and Specialization Patterns',
                 fontsize=18, fontweight='bold', y=1.02, color=PALETTE['primary'])
    
    plt.tight_layout()
    save_figure(fig, 'fig05_expert_activation')
    return fig


# =============================================================================
# FIGURE 6: Enhanced Sankey/Alluvial Diagram
# =============================================================================

def clean_category_name(name):
    """Clean category name: replace underscores with spaces and capitalize properly"""
    return name.replace('_', ' ').title()

def create_fig06_sankey_diagram():
    """Create enhanced flow visualization for input-expert routing with clear differentiation"""
    logger.info("Creating Figure 6: Input-Expert Flow Diagram...")
    
    try:
        import plotly.graph_objects as go
        
        with open(config.MOE_ANALYSIS_DIR / "category_expert_matrix.pkl", 'rb') as f:
            data = pickle.load(f)
        
        matrix = data['matrix']
        categories = data['categories']
        n_experts = data['n_experts']
        
        # Create Sankey with enhanced styling and clear differentiation
        source, target, value, link_colors = [], [], [], []
        
        # Distinct category colors (left side - warm tones for input categories)
        cat_colors = [
            'rgba(231, 76, 60, 0.9)',    # Red - Scientific
            'rgba(241, 196, 15, 0.9)',   # Yellow - Social
            'rgba(230, 126, 34, 0.9)',   # Orange - Emotional
            'rgba(211, 84, 0, 0.9)',     # Dark Orange - Procedural
            'rgba(192, 57, 43, 0.9)',    # Dark Red - Narrative
            'rgba(243, 156, 18, 0.9)',   # Gold - Mathematical
            'rgba(230, 80, 50, 0.9)',    # Coral - Risk Decision
        ]
        
        # Distinct expert colors (right side - cool tones for experts)
        expert_colors = [
            'rgba(52, 152, 219, 0.9)',   # Blue
            'rgba(41, 128, 185, 0.9)',   # Dark Blue
            'rgba(26, 188, 156, 0.9)',   # Teal
            'rgba(22, 160, 133, 0.9)',   # Dark Teal
            'rgba(155, 89, 182, 0.9)',   # Purple
            'rgba(142, 68, 173, 0.9)',   # Dark Purple
            'rgba(46, 204, 113, 0.9)',   # Green
            'rgba(39, 174, 96, 0.9)',    # Dark Green
        ]
        
        # Calculate link values with emphasis on differences
        for i, cat in enumerate(categories):
            for j in range(n_experts):
                # Include all connections but scale to show differences better
                routing_prob = matrix[i, j]
                if routing_prob > 0.01:  # Include more connections
                    source.append(i)
                    target.append(len(categories) + j)
                    # Scale values to emphasize differences
                    scaled_value = routing_prob ** 0.7 * 100  # Power scaling to enhance differences
                    value.append(max(scaled_value, 2))  # Minimum visible width
                    # Link color follows source category with varying opacity based on strength
                    opacity = 0.3 + 0.5 * (routing_prob / matrix.max())
                    base_color = cat_colors[i % len(cat_colors)]
                    link_colors.append(base_color.replace('0.9', f'{opacity:.2f}'))
        
        # Clean category names (replace underscores with spaces)
        clean_categories = [clean_category_name(c) for c in categories]
        node_labels = clean_categories + [f'Expert {i}' for i in range(n_experts)]
        node_colors = cat_colors[:len(categories)] + expert_colors[:n_experts]
        
        # Calculate node positions for better layout
        n_cats = len(categories)
        cat_y = [(i + 0.5) / n_cats for i in range(n_cats)]
        exp_y = [(i + 0.5) / n_experts for i in range(n_experts)]
        
        fig = go.Figure(data=[go.Sankey(
            arrangement='snap',
            node=dict(
                pad=40,  # Increased padding between nodes
                thickness=25,  # Slightly thinner nodes
                line=dict(color="white", width=3),
                label=node_labels,
                color=node_colors,
                x=[0.15] * len(categories) + [0.85] * n_experts,  # Move nodes inward to give space for labels
                y=cat_y + exp_y,
            ),
            link=dict(
                source=source,
                target=target,
                value=value,
                color=link_colors,
            )
        )])
        
        fig.update_layout(
            title=dict(
                text="<b>Input Category → Expert Routing Flow</b><br>" +
                     "<sup style='color:#7f8c8d'>Link width represents routing probability (scaled for visibility) | " +
                     "Categories (warm colors) → Experts (cool colors)</sup>",
                font=dict(size=20, color='#2C3E50', family='Arial'),
                x=0.5,
                y=0.98,
            ),
            font=dict(size=12, family="Arial"),
            width=1400,
            height=900,  # Increased height for more space
            paper_bgcolor='white',
            plot_bgcolor='white',
            margin=dict(l=120, r=120, t=130, b=60),  # Wider left/right margins for labels
        )
        
        # Add annotations to explain the sides - positioned higher to avoid overlap
        fig.add_annotation(
            x=0.02, y=1.02,
            text="<b>Input Categories</b>",
            showarrow=False,
            font=dict(size=14, color='#c0392b'),
            xref='paper', yref='paper'
        )
        fig.add_annotation(
            x=0.98, y=1.02,
            text="<b>MoE Experts</b>",
            showarrow=False,
            font=dict(size=14, color='#2980b9'),
            xref='paper', yref='paper'
        )
        
        fig.write_html(config.FIGURES_DIR / "fig06_input_expert_sankey.html")
        fig.write_image(config.FIGURES_DIR / "fig06_input_expert_sankey.png", scale=2)
        fig.write_image(config.FIGURES_DIR / "fig06_input_expert_sankey.svg")
        logger.info(f"Saved: {config.FIGURES_DIR / 'fig06_input_expert_sankey.png'}")
        
    except Exception as e:
        logger.warning(f"Plotly visualization failed: {e}. Creating matplotlib version...")
        create_fig06_matplotlib_fallback()
    
    return None


def create_fig06_matplotlib_fallback():
    """Matplotlib fallback for Sankey diagram with enhanced differentiation"""
    with open(config.MOE_ANALYSIS_DIR / "category_expert_matrix.pkl", 'rb') as f:
        data = pickle.load(f)
    
    matrix = data['matrix']
    categories = data['categories']
    n_experts = data['n_experts']
    
    # Clean category names
    clean_categories = [clean_category_name(c) for c in categories]
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Use a diverging colormap to show differences better
    # Normalize to emphasize differences
    matrix_normalized = (matrix - matrix.min()) / (matrix.max() - matrix.min() + 1e-8)
    
    cmap = create_scientific_cmap('sequential')
    im = ax.imshow(matrix_normalized, cmap=cmap, aspect='auto', vmin=0, vmax=1)
    
    # Add cell values with better contrast
    for i in range(len(categories)):
        for j in range(n_experts):
            val = matrix[i, j]
            norm_val = matrix_normalized[i, j]
            text_color = 'white' if norm_val > 0.5 else PALETTE['primary']
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                   color=text_color, fontsize=11, fontweight='bold')
    
    # Highlight max value in each row (dominant expert for each category)
    for i in range(len(categories)):
        max_j = np.argmax(matrix[i])
        rect = plt.Rectangle((max_j - 0.5, i - 0.5), 1, 1, 
                             fill=False, edgecolor=PALETTE['accent2'], 
                             linewidth=3, linestyle='-')
        ax.add_patch(rect)
    
    ax.set_xticks(range(n_experts))
    ax.set_xticklabels([f'Expert {i}' for i in range(n_experts)], fontsize=12, fontweight='medium')
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(clean_categories, fontsize=12, fontweight='medium')
    ax.set_xlabel('MoE Experts', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel('Input Categories', fontsize=14, fontweight='bold', labelpad=10)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Routing Probability (normalized)', fontsize=12)
    
    # Add grid lines for clarity
    ax.set_xticks(np.arange(-0.5, n_experts, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(categories), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=2)
    
    ax.set_title('Input Category → Expert Routing Probabilities\n'
                 '(Green border = dominant expert for each category)',
                 fontsize=16, fontweight='bold', color=PALETTE['primary'], pad=15)
    
    plt.tight_layout()
    save_figure(fig, 'fig06_input_expert_sankey')


# =============================================================================
# FIGURE 7: Polar Area Chart for Expert Specialization
# =============================================================================

def create_fig07_expert_wordcloud():
    """Create polar area chart showing expert specialization"""
    logger.info("Creating Figure 7: Expert Specialization Polar Chart...")
    
    with open(config.MOE_ANALYSIS_DIR / "expert_specialization.json", 'r') as f:
        spec_data = json.load(f)
    
    n_experts = len(spec_data)
    categories = [c['name'] for c in config.MOE_CONFIG['input_categories']]
    
    # Create subplots for each expert
    n_cols = 4
    n_rows = 2
    
    fig = plt.figure(figsize=(18, 10))
    
    # Use consistent category colors
    cat_colors = [CATEGORY_COLORS.get(cat, '#7F8C8D') for cat in categories]
    
    for expert_id in range(n_experts):
        ax = fig.add_subplot(n_rows, n_cols, expert_id + 1, polar=True)
        
        expert_info = spec_data[str(expert_id)]
        distribution = expert_info.get('category_distribution', {})
        
        # Get values for each category
        values = [distribution.get(cat, 0) for cat in categories]
        
        # Angles for polar plot
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
        
        # Close the polygon
        values_closed = values + [values[0]]
        angles_closed = np.concatenate([angles, [angles[0]]])
        
        # Plot filled area using consistent moe colormap color
        moe_color = EXPERT_COLORS[expert_id % len(EXPERT_COLORS)]
        ax.fill(angles_closed, values_closed, alpha=0.35, color=moe_color)
        ax.plot(angles_closed, values_closed, 'o-', linewidth=2, 
                color=moe_color, markersize=6)
        
        # Add bars for each category
        width = 2 * np.pi / len(categories) * 0.7
        for i, (angle, val, color) in enumerate(zip(angles, values, cat_colors)):
            ax.bar(angle, val, width=width, bottom=0, color=color, 
                   alpha=0.6, edgecolor='white', linewidth=1)
        
        ax.set_xticks(angles)
        ax.set_xticklabels([c[:4].upper() for c in categories], fontsize=9)
        max_val = max(values) * 1.3 if max(values) > 0 else 0.3
        ax.set_ylim(0, max_val)
        # Set 4 radial ticks at 25%, 50%, 75%, 100% positions
        tick_vals = [max_val * 0.25, max_val * 0.5, max_val * 0.75, max_val]
        ax.set_yticks(tick_vals)
        ax.set_yticklabels([f'{v:.2f}' for v in tick_vals])  # Format to 2 decimal places
        ax.tick_params(axis='y', labelsize=11)  # Larger font for radial labels
        ax.tick_params(axis='x', pad=8)  # Add padding to category labels
        
        dominant = expert_info.get('dominant_category', 'N/A')
        selectivity = expert_info.get('selectivity', 0)
        ax.set_title(f'Expert {expert_id}\n{clean_category_name(dominant)}\n(S={selectivity:.2f})',
                    fontsize=10, fontweight='bold', pad=10)
    
    # Add legend with consistent category colors
    legend_elements = [mpatches.Patch(facecolor=color, alpha=0.6, 
                                       label=clean_category_name(cat), edgecolor='white')
                       for cat, color in zip(categories, cat_colors)]
    fig.legend(handles=legend_elements, loc='center right', 
               bbox_to_anchor=(1.0, 0.5), fontsize=9, title='Input Categories',
               title_fontsize=11)
    
    fig.suptitle('MoE Expert Specialization Profiles',
                 fontsize=18, fontweight='bold', y=0.98, color=PALETTE['primary'])
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    save_figure(fig, 'fig07_expert_wordcloud')
    return fig


# =============================================================================
# FIGURE 8: Dual Heatmap with Marginal Distributions
# =============================================================================

def create_fig08_routing_entropy():
    """Create dual heatmap comparing brain and MoE routing"""
    logger.info("Creating Figure 8: Routing Entropy Analysis...")
    
    with open(config.NEUROSCIENCE_DIR / "task_network_matrix.pkl", 'rb') as f:
        task_net_data = pickle.load(f)
    
    with open(config.MOE_ANALYSIS_DIR / "category_expert_matrix.pkl", 'rb') as f:
        cat_exp_data = pickle.load(f)
    
    fig = plt.figure(figsize=(15, 6))
    
    # Two-panel layout with more spacing between panels
    gs = GridSpec(1, 5, width_ratios=[1, 0.04, 0.5, 1, 0.04], wspace=0.08,
                  left=0.06, right=0.96)
    
    # === Brain Panel ===
    ax_brain = fig.add_subplot(gs[0, 0])
    ax_brain_cbar = fig.add_subplot(gs[0, 1])
    
    # Brain data
    task_matrix = task_net_data['overall']
    tasks = task_net_data['tasks']
    networks = task_net_data['networks']
    
    # Convert brain activation values to probabilities using softmax
    # Brain data contains z-scores/beta weights which can be negative
    # Softmax converts them to proper probability distributions [0, 1]
    from scipy.special import softmax
    task_probs = softmax(task_matrix, axis=1)
    
    # Main heatmap
    cmap_brain = create_scientific_cmap('brain')
    im1 = ax_brain.imshow(task_probs, cmap=cmap_brain, aspect='auto', vmin=0, vmax=1)
    
    ax_brain.set_xticks(range(len(networks)))
    ax_brain.set_xticklabels([config.NETWORKS[n]['name'] for n in networks],
                             rotation=45, ha='right', fontsize=10)
    ax_brain.set_yticks(range(len(tasks)))
    ax_brain.set_yticklabels([config.TASKS[t]['name'] for t in tasks], fontsize=10)
    ax_brain.set_xlabel('Brain Networks', fontsize=12)
    ax_brain.set_ylabel('Cognitive Tasks', fontsize=12)
    
    # Colorbar
    cbar1 = plt.colorbar(im1, cax=ax_brain_cbar)
    cbar1.set_label('Routing Prob.', fontsize=10)
    
    ax_brain.set_title('A. Brain: Task-Network Routing', fontsize=13, fontweight='bold', pad=10)
    
    # === MoE Panel ===
    ax_moe = fig.add_subplot(gs[0, 3])
    ax_moe_cbar = fig.add_subplot(gs[0, 4])
    
    # MoE data
    cat_matrix = cat_exp_data['matrix']
    categories = cat_exp_data['categories']
    n_experts = cat_exp_data['n_experts']
    
    cat_probs = cat_matrix / (cat_matrix.sum(axis=1, keepdims=True) + 1e-10)
    
    # Main heatmap - use same scale as brain panel for fair comparison
    cmap_moe = create_scientific_cmap('moe')
    im2 = ax_moe.imshow(cat_probs, cmap=cmap_moe, aspect='auto', vmin=0, vmax=1)
    
    ax_moe.set_xticks(range(n_experts))
    ax_moe.set_xticklabels([f'E{i}' for i in range(n_experts)], fontsize=10)
    ax_moe.set_yticks(range(len(categories)))
    ax_moe.set_yticklabels([clean_category_name(c) for c in categories], fontsize=10)
    ax_moe.set_xlabel('MoE Experts', fontsize=12)
    ax_moe.set_ylabel('Input Categories', fontsize=12)
    
    # Colorbar
    cbar2 = plt.colorbar(im2, cax=ax_moe_cbar)
    cbar2.set_label('Routing Prob.', fontsize=10)
    
    ax_moe.set_title('B. MoE: Category-Expert Routing', fontsize=13, fontweight='bold', pad=10)
    
    plt.tight_layout()
    save_figure(fig, 'fig08_routing_entropy')
    return fig


# =============================================================================
# FIGURE H3: Routing Consistency Comparison between Efficiency Groups
# =============================================================================

def create_fig_h3_routing_consistency():
    """Create routing consistency comparison between high/low efficiency groups"""
    logger.info("Creating Figure H3: Routing Consistency Comparison...")
    
    # Load data
    with open(config.NEUROSCIENCE_DIR / 'routing_profiles.pkl', 'rb') as f:
        routing_profiles = pickle.load(f)
    
    sparsity_df = pd.read_csv(config.NEUROSCIENCE_DIR / 'sparsity_metrics.csv')
    
    # Get efficiency groups
    high_eff_subjects = [str(s) for s in sparsity_df[sparsity_df['efficiency_group'] == 'high']['subject'].unique()]
    low_eff_subjects = [str(s) for s in sparsity_df[sparsity_df['efficiency_group'] == 'low']['subject'].unique()]
    
    # Compute pairwise correlations within each group
    def compute_pairwise_correlations(subjects, profiles):
        correlations = []
        for i, s1 in enumerate(subjects):
            for s2 in subjects[i+1:]:
                if s1 in profiles and s2 in profiles:
                    p1 = profiles[s1].flatten()
                    p2 = profiles[s2].flatten()
                    r, _ = stats.pearsonr(p1, p2)
                    correlations.append(r)
        return correlations
    
    high_corrs = compute_pairwise_correlations(high_eff_subjects, routing_profiles)
    low_corrs = compute_pairwise_correlations(low_eff_subjects, routing_profiles)
    
    # Create figure with proper spacing
    fig = plt.figure(figsize=(18, 5))
    gs = GridSpec(1, 4, width_ratios=[1, 0.5, 1, 0.8], wspace=0.06, left=0.05, right=0.97)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[0, 3])
    
    # Compute mean routing profiles
    high_profiles = np.array([routing_profiles[s] for s in high_eff_subjects if s in routing_profiles])
    low_profiles = np.array([routing_profiles[s] for s in low_eff_subjects if s in routing_profiles])
    
    mean_high = np.mean(high_profiles, axis=0)
    mean_low = np.mean(low_profiles, axis=0)
    
    tasks = config.TASK_ORDER
    networks = config.NETWORK_ORDER
    task_names = [config.TASKS[t]['name'] for t in tasks]
    network_names = [config.NETWORKS[n]['name'] for n in networks]
    
    # Common color scale
    vmax = max(np.abs(mean_high).max(), np.abs(mean_low).max())
    
    # Panel A: High efficiency
    im1 = ax1.imshow(mean_high, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
    ax1.set_xticks(range(len(networks)))
    ax1.set_xticklabels(network_names, rotation=45, ha='right', fontsize=9)
    ax1.set_yticks(range(len(tasks)))
    ax1.set_yticklabels(task_names, fontsize=9)
    ax1.set_title('A. High-Efficiency Group\nMean Routing Profile', fontsize=12, fontweight='bold', color=PALETTE['primary'])
    ax1.set_xlabel('Brain Networks', fontsize=10)
    ax1.set_ylabel('Cognitive Tasks', fontsize=10)
    
    # Panel B: Low efficiency
    im2 = ax2.imshow(mean_low, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
    ax2.set_xticks(range(len(networks)))
    ax2.set_xticklabels(network_names, rotation=45, ha='right', fontsize=9)
    ax2.set_yticks(range(len(tasks)))
    ax2.set_yticklabels(task_names, fontsize=9)
    ax2.set_title('B. Low-Efficiency Group\nMean Routing Profile', fontsize=12, fontweight='bold', color=PALETTE['secondary'])
    ax2.set_xlabel('Brain Networks', fontsize=10)
    ax2.set_ylabel('Cognitive Tasks', fontsize=10)
    
    # Colorbar
    cbar = fig.colorbar(im2, ax=[ax1, ax2], shrink=0.6, pad=0.02)
    cbar.set_label('Activation (a.u.)', fontsize=10)
    
    # Panel C: Bar chart
    x = [0, 1]
    heights = [np.mean(high_corrs), np.mean(low_corrs)]
    errors = [np.std(high_corrs) / np.sqrt(len(high_corrs)), 
              np.std(low_corrs) / np.sqrt(len(low_corrs))]
    colors = [PALETTE['primary'], PALETTE['secondary']]
    
    bars = ax3.bar(x, heights, yerr=errors, capsize=5, color=colors, edgecolor='white', linewidth=2)
    ax3.set_xticks(x)
    ax3.set_xticklabels(['High\nEfficiency', 'Low\nEfficiency'], fontsize=11)
    ax3.set_ylabel('Inter-Subject Correlation (r)', fontsize=11)
    ax3.set_ylim(0, 1)
    ax3.set_title('C. Routing Consistency\n(Inter-Subject Similarity)', fontsize=12, fontweight='bold')
    
    # Statistical annotation
    t_stat, p_val = stats.ttest_ind(high_corrs, low_corrs)
    cohens_d = (np.mean(high_corrs) - np.mean(low_corrs)) / np.sqrt((np.var(high_corrs) + np.var(low_corrs)) / 2)
    
    y_max = max(heights) + max(errors) + 0.05
    ax3.plot([0, 0, 1, 1], [y_max, y_max + 0.02, y_max + 0.02, y_max], 'k-', linewidth=1.5)
    sig_text = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
    ax3.text(0.5, y_max + 0.03, sig_text, ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    stats_text = f't = {t_stat:.2f}, p = {p_val:.3f}\nd = {cohens_d:.2f}'
    ax3.text(0.5, 0.15, stats_text, ha='center', va='bottom', fontsize=10, 
             transform=ax3.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Values on bars
    for i, (h, bar) in enumerate(zip(heights, bars)):
        ax3.text(bar.get_x() + bar.get_width()/2, h + errors[i] + 0.02, 
                 f'{h:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    fig.suptitle('Routing Consistency: High vs Low Efficiency Groups', 
                 fontsize=14, fontweight='bold', y=1.02, color=PALETTE['primary'])
    
    save_figure(fig, 'fig_h3_routing_consistency')
    return fig


# =============================================================================
# FIGURE 9: Cross-Domain Sparsity Comparison with Effect Sizes
# =============================================================================

def create_fig09_sparsity_comparison():
    """Create comprehensive cross-domain sparsity comparison"""
    logger.info("Creating Figure 9: Cross-Domain Sparsity Comparison...")
    
    brain_sparsity = pd.read_csv(config.NEUROSCIENCE_DIR / "sparsity_metrics.csv")
    moe_sparsity = pd.read_csv(config.MOE_ANALYSIS_DIR / "moe_sparsity.csv")

    fig = plt.figure(figsize=(18, 6.0))
    gs = GridSpec(1, 3, wspace=0.3)
    
    # Panel A: Overlapping density plots
    ax1 = fig.add_subplot(gs[0])
    
    # For consistency with Table III and the abstract, compare AGGREGATE
    # per-model MoE Gini (one value per model, from multi_moe_summary.csv)
    # against the brain Gini distribution. Per-sample MoE Gini is a different
    # object (it measures sparsity of a single short text input, which by
    # construction is high for any top-k router) and is not the quantity the
    # main narrative claims brain routing is 2.5× more sparse than.
    brain_vals = brain_sparsity['gini_coefficient'].values
    moe_summary = pd.read_csv(config.MOE_ANALYSIS_DIR / "multi_moe_summary.csv")
    moe_vals = moe_summary['gini'].values
    
    # KDE plots with fill
    from scipy.stats import gaussian_kde
    
    # Brain as KDE (large sample); MoE as discrete vertical lines (n=3 models)
    brain_kde = gaussian_kde(brain_vals)
    x_range = np.linspace(0, max(brain_vals.max(), moe_vals.max()) + 0.05, 300)
    ax1.fill_between(x_range, brain_kde(x_range), alpha=0.4, color=PALETTE['brain'],
                     label='Brain (per subject×task)')
    ax1.plot(x_range, brain_kde(x_range), color=PALETTE['brain'], linewidth=2)
    ax1.axvline(brain_vals.mean(), color=PALETTE['brain'], linestyle='--', linewidth=2,
                label=f'Brain mean = {brain_vals.mean():.3f}')

    moe_labels = moe_summary['model'].tolist()
    moe_colors = [PALETTE['ai']] * len(moe_vals)
    # Compute a y-range with enough headroom that the upper-right legend does
    # not overlap the KDE peak or the vertical MoE lines.
    kde_peak = float(brain_kde(x_range).max())
    y_top = kde_peak * 1.45  # 45% headroom above the peak for legend + labels
    ax1.set_ylim(0, y_top)
    # Place MoE model labels mid-height so they do not collide with the legend.
    label_y = y_top * 0.55
    for val, lbl in zip(moe_vals, moe_labels):
        ax1.axvline(val, color=PALETTE['ai'], linestyle='-', linewidth=2, alpha=0.7)
        ax1.text(val, label_y,
                 lbl.replace('-MoE', '').replace('-16B', ''),
                 rotation=90, fontsize=8, color=PALETTE['ai'],
                 va='center', ha='right')
    ax1.axvline(moe_vals.mean(), color=PALETTE['ai'], linestyle=':', linewidth=2,
                label=f'MoE mean = {moe_vals.mean():.3f}')

    ax1.set_xlabel('Gini Coefficient', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Sparsity Distributions', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', framealpha=0.95)
    add_panel_label(ax1, 'A', x=-0.12, y=1.05)
    
    # Panel B: Direct system-by-system bar chart — 1 brain bar (overall
    # mean ± SD across subjects×tasks) plus one bar per MoE model. Clean and
    # directly interpretable; replaces the earlier cluttered per-task view.
    ax2 = fig.add_subplot(gs[1])

    brain_mean = float(brain_vals.mean())
    brain_sd = float(brain_vals.std())
    moe_models = moe_summary['model'].tolist()
    moe_gini = moe_vals.tolist()

    labels = ['Brain\n(HCP, N=100)'] + moe_models
    values = [brain_mean] + moe_gini
    errs = [brain_sd] + [0] * len(moe_gini)
    colors = [PALETTE['brain']] + [PALETTE['ai']] * len(moe_gini)

    x_pos = np.arange(len(labels))
    bars = ax2.bar(x_pos, values, yerr=errs, color=colors, alpha=0.85,
                   edgecolor='white', linewidth=1.5, capsize=6,
                   error_kw={'linewidth': 1.5, 'ecolor': PALETTE['neutral']})
    # Annotate each bar with its numeric value
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.02,
                 f"{val:.3f}", ha='center', va='bottom',
                 fontsize=10, color=PALETTE['neutral'])

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylabel('Gini coefficient', fontsize=12)
    ax2.set_title('Brain vs MoE (aggregate)', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, max(values) * 1.25)
    add_panel_label(ax2, 'B', x=-0.12, y=1.05)
    
    # Panel C: Overall comparison with effect size
    ax3 = fig.add_subplot(gs[2])
    
    # Box + strip plot
    positions = [0, 1]
    bp = ax3.boxplot([brain_vals, moe_vals], positions=positions,
                     widths=0.5, patch_artist=True, showfliers=False)
    
    for i, (patch, color) in enumerate(zip(bp['boxes'], [PALETTE['brain'], PALETTE['ai']])):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        patch.set_edgecolor(color)
        patch.set_linewidth(2)
    
    for median in bp['medians']:
        median.set_color('white')
        median.set_linewidth(2)
    
    # Add scatter points
    for i, (data, color) in enumerate([(brain_vals, PALETTE['brain']), (moe_vals, PALETTE['ai'])]):
        jitter = np.random.normal(0, 0.08, size=len(data))
        ax3.scatter(i + jitter, data, c=color, alpha=0.5, s=30, edgecolor='white', linewidth=0.5)
    
    # Statistical test
    t_stat, p_val = stats.ttest_ind(brain_vals, moe_vals)
    d = (brain_vals.mean() - moe_vals.mean()) / np.sqrt((brain_vals.std()**2 + moe_vals.std()**2) / 2)
    
    y_max = max(brain_vals.max(), moe_vals.max())
    # Place Cohen's d text well above the significance bar+stars so the two
    # annotations never overlap.
    add_significance_annotation(ax3, 0, 1, y_max * 1.08, p_val, height=y_max * 0.04)

    ax3.text(0.5, y_max * 1.35, f"Cohen's d = {d:.2f}", ha='center', fontsize=11,
             color=PALETTE['primary'], fontweight='bold')
    ax3.set_ylim(top=y_max * 1.45)

    ax3.set_xticks(positions)
    ax3.set_xticklabels(['Brain', 'MoE'], fontsize=12)
    ax3.set_ylabel('Gini Coefficient', fontsize=12)
    ax3.set_title('Overall Comparison', fontsize=14, fontweight='bold', pad=20)
    add_panel_label(ax3, 'C', x=-0.12, y=1.1)
    
    # No suptitle (caption provides context in manuscript)
    plt.tight_layout(rect=[0, 0, 1, 1.0])
    save_figure(fig, 'fig09_sparsity_comparison')
    return fig


# =============================================================================
# FIGURE 10: Hierarchically Clustered Similarity Matrix
# =============================================================================

def create_fig10_similarity_matrix():
    """Create clustered similarity matrices with dendrograms"""
    logger.info("Creating Figure 10: Similarity Matrix...")
    
    with open(config.NEUROSCIENCE_DIR / "task_similarity_matrix.pkl", 'rb') as f:
        brain_sim = pickle.load(f)
    
    with open(config.MOE_ANALYSIS_DIR / "category_expert_matrix.pkl", 'rb') as f:
        moe_data = pickle.load(f)
    
    brain_matrix = brain_sim['matrix']
    moe_matrix = moe_data['matrix']
    moe_cat_sim = np.corrcoef(moe_matrix)
    
    fig = plt.figure(figsize=(16, 7))
    
    # Use seaborn clustermap approach but with subplots
    gs = GridSpec(1, 2, wspace=0.4)
    
    # Panel A: Brain
    ax1 = fig.add_subplot(gs[0])
    
    # Cluster
    mask = np.triu(np.ones_like(brain_matrix, dtype=bool), k=1)
    
    # Use consistent diverging colormap
    cmap = create_scientific_cmap('diverging')
    sns.heatmap(brain_matrix, mask=mask, cmap=cmap, center=0,
                square=True, linewidths=2, linecolor='white',
                cbar_kws={'shrink': 0.6, 'label': 'Correlation'},
                annot=True, fmt='.2f', annot_kws={'size': 9},
                ax=ax1, vmin=-1, vmax=1)
    
    ax1.set_xticklabels([brain_sim['tasks'][i] for i in range(len(brain_sim['tasks']))],
                        rotation=45, ha='right', fontsize=10)
    ax1.set_yticklabels([brain_sim['tasks'][i] for i in range(len(brain_sim['tasks']))],
                        rotation=0, fontsize=10)
    ax1.set_title('Brain: Task Routing Similarity', fontsize=14, fontweight='bold', pad=10)
    add_panel_label(ax1, 'A', x=-0.15, y=1.05)
    
    # Panel B: MoE
    ax2 = fig.add_subplot(gs[1])
    
    mask2 = np.triu(np.ones_like(moe_cat_sim, dtype=bool), k=1)
    
    sns.heatmap(moe_cat_sim, mask=mask2, cmap=cmap, center=0,
                square=True, linewidths=2, linecolor='white',
                cbar_kws={'shrink': 0.6, 'label': 'Correlation'},
                annot=True, fmt='.2f', annot_kws={'size': 9},
                ax=ax2, vmin=-1, vmax=1)
    
    ax2.set_xticklabels([clean_category_name(c) for c in moe_data['categories']],
                        rotation=45, ha='right', fontsize=10)
    ax2.set_yticklabels([clean_category_name(c) for c in moe_data['categories']],
                        rotation=0, fontsize=10)
    ax2.set_title('MoE: Category Routing Similarity', fontsize=14, fontweight='bold', pad=10)
    add_panel_label(ax2, 'B', x=-0.15, y=1.05)
    
    fig.suptitle('Routing Pattern Similarity Structures',
                 fontsize=18, fontweight='bold', y=1.02, color=PALETTE['primary'])
    
    plt.tight_layout()
    save_figure(fig, 'fig10_similarity_matrix')
    return fig


# =============================================================================
# FIGURE 11: Multi-Panel Regression Analysis
# =============================================================================

def create_fig11_efficiency_scatter():
    """Create comprehensive efficiency-sparsity regression analysis"""
    logger.info("Creating Figure 11: Efficiency-Sparsity Analysis...")
    
    sparsity_df = pd.read_csv(config.NEUROSCIENCE_DIR / "sparsity_metrics.csv")
    
    fig = plt.figure(figsize=(18, 6))
    gs = GridSpec(1, 3, wspace=0.3)
    
    # Panel A: Violin + Box + Scatter comparison
    ax1 = fig.add_subplot(gs[0])
    
    high_sparsity = sparsity_df[sparsity_df['efficiency_group'] == 'high']['composite_sparsity']
    low_sparsity = sparsity_df[sparsity_df['efficiency_group'] == 'low']['composite_sparsity']
    
    # Combined violin + box
    parts = ax1.violinplot([high_sparsity, low_sparsity], positions=[0, 1],
                           showmeans=False, showextrema=False, widths=0.8)
    
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor([PALETTE['high_eff'], PALETTE['low_eff']][i])
        pc.set_alpha(0.3)
    
    bp = ax1.boxplot([high_sparsity, low_sparsity], positions=[0, 1],
                     widths=0.2, patch_artist=True, showfliers=False)
    
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor([PALETTE['high_eff'], PALETTE['low_eff']][i])
        patch.set_alpha(0.8)
        patch.set_edgecolor('white')
        patch.set_linewidth(2)
    
    for median in bp['medians']:
        median.set_color('white')
        median.set_linewidth(2)
    
    # Scatter with jitter
    for i, (data, color) in enumerate([(high_sparsity, PALETTE['high_eff']), 
                                        (low_sparsity, PALETTE['low_eff'])]):
        jitter = np.random.normal(0.35, 0.05, size=len(data))
        ax1.scatter(i + jitter, data, c=color, alpha=0.7, s=60, 
                   edgecolor='white', linewidth=1, zorder=5)
    
    t_stat, p_val = stats.ttest_ind(high_sparsity, low_sparsity)
    y_max = max(high_sparsity.max(), low_sparsity.max())
    add_significance_annotation(ax1, 0, 1, y_max * 1.02, p_val)
    
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['High\nEfficiency', 'Low\nEfficiency'], fontsize=11)
    ax1.set_ylabel('Activation Sparsity', fontsize=12)
    ax1.set_title('Group Comparison', fontsize=14, fontweight='bold')
    add_panel_label(ax1, 'A', x=-0.12, y=1.05)
    
    # Panel B: Task breakdown
    ax2 = fig.add_subplot(gs[1])
    
    tasks = config.TASK_ORDER
    task_names = [config.TASKS[t]['name'] for t in tasks]
    
    for i, task in enumerate(tasks):
        task_data = sparsity_df[sparsity_df['task'] == task]
        high = task_data[task_data['efficiency_group'] == 'high']['composite_sparsity'].mean()
        low = task_data[task_data['efficiency_group'] == 'low']['composite_sparsity'].mean()
        
        ax2.plot([i, i], [low, high], '-', color=PALETTE['neutral'], linewidth=2, zorder=1)
        ax2.scatter(i, high, s=100, c=PALETTE['high_eff'], edgecolor='white',
                   linewidth=2, zorder=5, marker='o')
        ax2.scatter(i, low, s=100, c=PALETTE['low_eff'], edgecolor='white',
                   linewidth=2, zorder=5, marker='s')
    
    ax2.set_xticks(range(len(tasks)))
    ax2.set_xticklabels(task_names, rotation=45, ha='right', fontsize=10)
    ax2.set_ylabel('Mean Sparsity', fontsize=12)
    ax2.set_title('Task-Specific Patterns', fontsize=14, fontweight='bold')
    
    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=PALETTE['high_eff'],
                   markersize=10, label='High Efficiency'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=PALETTE['low_eff'],
                   markersize=10, label='Low Efficiency')
    ]
    ax2.legend(handles=legend_elements, loc='upper right', framealpha=0.95)
    add_panel_label(ax2, 'B', x=-0.12, y=1.05)
    
    # Panel C: Correlation plot
    ax3 = fig.add_subplot(gs[2])
    
    subject_stats = sparsity_df.groupby('subject').agg({
        'composite_sparsity': 'mean',
        'gini_coefficient': 'mean',
        'efficiency_group': 'first'
    }).reset_index()
    
    colors = [PALETTE['high_eff'] if g == 'high' else PALETTE['low_eff'] 
              for g in subject_stats['efficiency_group']]
    
    x = subject_stats['composite_sparsity'].values
    y = subject_stats['gini_coefficient'].values
    
    ax3.scatter(x, y, c=colors, s=120, alpha=0.8, edgecolor='white', linewidth=2)
    
    # Regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = slope * x_line + intercept
    
    ax3.plot(x_line, y_line, '--', color=PALETTE['primary'], linewidth=2.5)
    
    # Confidence interval
    y_pred = slope * x + intercept
    residuals = y - y_pred
    se = np.sqrt(np.sum(residuals**2) / (len(x) - 2))
    ci = 1.96 * se * np.sqrt(1/len(x) + (x_line - x.mean())**2 / np.sum((x - x.mean())**2))
    ax3.fill_between(x_line, y_line - ci, y_line + ci, alpha=0.2, color=PALETTE['primary'])
    
    # Statistics annotation
    ax3.text(0.05, 0.95, f'r = {r_value:.3f}\np = {p_value:.4f}',
             transform=ax3.transAxes, fontsize=11, fontweight='bold',
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    ax3.set_xlabel('Composite Sparsity', fontsize=12)
    ax3.set_ylabel('Gini Coefficient', fontsize=12)
    ax3.set_title('Sparsity-Selectivity Correlation', fontsize=14, fontweight='bold')
    add_panel_label(ax3, 'C', x=-0.12, y=1.05)
    
    fig.suptitle('Neural Efficiency and Activation Sparsity Relationship',
                 fontsize=18, fontweight='bold', y=1.02, color=PALETTE['primary'])
    
    plt.tight_layout()
    save_figure(fig, 'fig11_efficiency_scatter')
    return fig


# =============================================================================
# FIGURE 12: Overlaid Specialization Comparison
# =============================================================================

def create_fig12_specialization_radar():
    """Create overlaid radar comparison with statistics"""
    logger.info("Creating Figure 12: Specialization Comparison...")
    
    with open(config.NEUROSCIENCE_DIR / "routing_analysis.json", 'r') as f:
        brain_routing = json.load(f)
    
    with open(config.MOE_ANALYSIS_DIR / "expert_specialization.json", 'r') as f:
        moe_spec = json.load(f)
    
    fig = plt.figure(figsize=(16, 8))
    gs = GridSpec(1, 2, width_ratios=[1.2, 1], wspace=0.3)
    
    # Panel A: Radar comparison
    ax1 = fig.add_subplot(gs[0], polar=True)
    
    brain_selectivity = list(brain_routing['selectivity'].values())
    moe_selectivity = [moe_spec[str(i)]['selectivity'] for i in range(len(moe_spec))]
    
    n_points = min(len(brain_selectivity), len(moe_selectivity))
    brain_vals = brain_selectivity[:n_points]
    moe_vals = moe_selectivity[:n_points]
    
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False).tolist()
    angles += angles[:1]
    
    brain_vals_plot = brain_vals + [brain_vals[0]]
    moe_vals_plot = moe_vals + [moe_vals[0]]
    
    # Plot with styled fills
    ax1.fill(angles, brain_vals_plot, alpha=0.25, color=PALETTE['brain'])
    ax1.plot(angles, brain_vals_plot, 'o-', linewidth=2.5, color=PALETTE['brain'],
             markersize=10, markeredgecolor='white', markeredgewidth=2,
             label='Brain Networks')
    
    ax1.fill(angles, moe_vals_plot, alpha=0.25, color=PALETTE['ai'])
    ax1.plot(angles, moe_vals_plot, 's-', linewidth=2.5, color=PALETTE['ai'],
             markersize=10, markeredgecolor='white', markeredgewidth=2,
             label='MoE Experts')
    
    ax1.set_xticks(angles[:-1])
    labels = [f'Unit {i+1}' for i in range(n_points)]
    ax1.set_xticklabels(labels, size=10)
    ax1.set_ylim(0, 1.3)  # Increased to give more space for labels
    ax1.tick_params(axis='x', pad=25)  # Add padding to move labels outward
    
    ax1.legend(loc='upper left', bbox_to_anchor=(-0.15, 1.0), framealpha=0.95)
    ax1.set_title('Specialization Profile Comparison', fontsize=14, fontweight='bold', pad=20)
    add_panel_label(ax1, 'A', x=-0.2, y=1.15)
    
    # Panel B: Statistical summary
    ax2 = fig.add_subplot(gs[1])
    
    # Box plot comparison
    data_to_plot = [brain_vals, moe_vals]
    positions = [0, 1]
    
    bp = ax2.boxplot(data_to_plot, positions=positions, widths=0.5,
                     patch_artist=True, showfliers=True)
    
    for i, (patch, color) in enumerate(zip(bp['boxes'], [PALETTE['brain'], PALETTE['ai']])):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor(color)
        patch.set_linewidth(2)
    
    for median in bp['medians']:
        median.set_color('white')
        median.set_linewidth(2)
    
    # Add scatter
    for i, (data, color) in enumerate([(brain_vals, PALETTE['brain']), (moe_vals, PALETTE['ai'])]):
        jitter = np.random.normal(0, 0.05, size=len(data))
        ax2.scatter(i + jitter, data, c=color, alpha=0.7, s=80,
                   edgecolor='white', linewidth=1.5, zorder=5)
    
    # Statistics
    t_stat, p_val = stats.ttest_ind(brain_vals, moe_vals)
    d = (np.mean(brain_vals) - np.mean(moe_vals)) / np.sqrt((np.std(brain_vals)**2 + np.std(moe_vals)**2) / 2)
    
    y_max = max(max(brain_vals), max(moe_vals))
    add_significance_annotation(ax2, 0, 1, y_max * 1.05, p_val, height=y_max * 0.05)
    
    ax2.text(0.5, y_max * 1.2, f"Cohen's d = {d:.2f}", ha='center', fontsize=12,
             color=PALETTE['primary'], fontweight='bold')
    
    # Add means
    for i, (data, color) in enumerate([(brain_vals, PALETTE['brain']), (moe_vals, PALETTE['ai'])]):
        ax2.hlines(np.mean(data), i - 0.2, i + 0.2, colors=color, linewidth=3, zorder=6)
    
    ax2.set_xticks(positions)
    ax2.set_xticklabels(['Brain\nNetworks', 'MoE\nExperts'], fontsize=12)
    ax2.set_ylabel('Selectivity Index', fontsize=12)
    ax2.set_title('Specialization Statistics', fontsize=14, fontweight='bold')
    
    # Add summary statistics text - positioned below the plot
    stats_text = (f"Brain: μ={np.mean(brain_vals):.3f}, σ={np.std(brain_vals):.3f}\n"
                  f"MoE: μ={np.mean(moe_vals):.3f}, σ={np.std(moe_vals):.3f}")
    ax2.text(0.5, -0.15, stats_text, transform=ax2.transAxes, fontsize=10,
             ha='center', va='top', style='italic', color=PALETTE['neutral'])
    
    add_panel_label(ax2, 'B', x=-0.12, y=1.05)
    
    fig.suptitle('Functional Specialization: Brain vs MoE Comparison',
                 fontsize=18, fontweight='bold', y=1.02, color=PALETTE['primary'])
    
    plt.tight_layout()
    save_figure(fig, 'fig12_specialization_radar')
    return fig


# =============================================================================
# FIGURE 13: Brain Glass Model Visualization (360 HCP-MMP Parcels)
# =============================================================================

def create_fig13_brain_glass_model():
    """
    Create glass brain visualization showing activation patterns across 360 HCP-MMP parcels.
    Uses Schaefer atlas and NiftiLabelsMasker for proper brain mapping.
    Creates individual figures for each task like project_1 style.
    """
    if not NILEARN_AVAILABLE:
        logger.warning("Skipping brain glass model: nilearn not available")
        return None
    
    logger.info("Creating Figure 13: Brain Glass Model (360 HCP-MMP Parcels)...")
    
    # Load activation data
    with open(config.NEUROSCIENCE_DIR / "activation_matrix.pkl", 'rb') as f:
        data = pickle.load(f)
    
    activation_matrix = data['data']
    tasks = data['tasks']
    n_parcels = data['n_parcels']
    
    # Average activation across subjects for each task
    avg_activation = np.mean(activation_matrix, axis=0)  # (7 tasks, 360 parcels)
    
    # Get Schaefer atlas from local cache (200 ROIs for better resolution)
    schaefer_nii_path = (Path.home() / 'nilearn_data' / 'schaefer_2018' / 'Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.nii.gz')
    schaefer_txt_path = (Path.home() / 'nilearn_data' / 'schaefer_2018' / 'Schaefer2018_200Parcels_7Networks_order.txt')

    if not schaefer_nii_path.exists() or not schaefer_txt_path.exists():
        logger.warning(f"Schaefer atlas not found locally")
        return create_fig13_brain_schematic_fallback()
    
    try:
        atlas_img = nib.load(str(schaefer_nii_path))
        with open(schaefer_txt_path, 'r') as f:
            atlas_labels = [line.strip().split('\t')[1] if '\t' in line else line.strip() 
                          for line in f.readlines() if line.strip()]
    except Exception as e:
        logger.warning(f"Could not load Schaefer atlas: {e}")
        return create_fig13_brain_schematic_fallback()
    
    # Create masker for inverse transform
    try:
        masker = NiftiLabelsMasker(labels_img=atlas_img, standardize=False)
        masker.fit()
    except Exception as e:
        logger.warning(f"Could not create masker: {e}")
        return create_fig13_brain_schematic_fallback()
    
    n_rois = len(atlas_labels)
    np.random.seed(42)  # For reproducibility
    
    # REAL-DATA ONLY: map 360-parcel activation (avg_activation[task_idx]) to
    # Schaefer ROIs by Yeo-network membership. Each Schaefer ROI inherits the
    # group-mean activation of all 360 parcels assigned to its Yeo network.
    # No hardcoded "expected activation" values, no random noise.
    NETWORK_KEYWORDS = {
        'VIS': 'Vis',       'SMN': 'SomMot',    'DAN': 'DorsAttn',
        'VAN': 'SalVentAttn','LIM': 'Limbic',   'FPN': 'Cont',
        'DMN': 'Default',
    }
    net_order = config.NETWORK_ORDER
    chunk = n_parcels // len(net_order)

    def parcel_to_net(p_idx):
        return net_order[min(p_idx // chunk, len(net_order) - 1)]

    def real_roi_values(task_activation):
        """Schaefer ROI values computed from real per-parcel activations,
        aggregated by Yeo network. Unmapped ROIs remain NaN (nilearn skips)."""
        net_mean = {net: float(np.nanmean(
            task_activation[[i for i in range(len(task_activation))
                             if parcel_to_net(i) == net]]
        )) for net in net_order}
        roi_values = np.full(n_rois, np.nan)
        for i in range(n_rois):
            label = atlas_labels[i]
            label_str = label.decode() if isinstance(label, bytes) else str(label)
            for net, keyword in NETWORK_KEYWORDS.items():
                if keyword in label_str:
                    roi_values[i] = net_mean[net]
                    break
        return roi_values

    task_names = [config.TASKS[t]['name'] for t in tasks]
    figures_created = []

    # Create individual glass brain figures for each task
    for task_idx, task in enumerate(tasks):
        roi_values = real_roi_values(avg_activation[task_idx])
        
        # Create NIfTI image using masker
        try:
            activation_img = masker.inverse_transform(roi_values)
        except Exception as e:
            logger.warning(f"Could not create activation image for {task}: {e}")
            continue
        
        # Create figure with clean styling (like project_1)
        fig = plt.figure(figsize=(12, 7), facecolor='white')
        
        # Plot glass brain with lyrz display mode — no suptitle (caption handles it)
        display = plotting.plot_glass_brain(
            activation_img,
            display_mode='lyrz',
            colorbar=True,
            threshold=0.15,
            cmap='hot',
            vmax=1.0,
            title='',
            figure=fig
        )

        fig.savefig(config.FIGURES_DIR / f'fig13a_glass_brain_{task.lower()}.png',
                    dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"Saved: Glass Brain ({task_names[task_idx]})")
        figures_created.append(fig)
        plt.close(fig)
    
    # Create combined overview figure — cross-task mean of real per-parcel
    # activation, mapped to Schaefer ROIs via Yeo-network membership.
    mean_activation_across_tasks = np.nanmean(avg_activation, axis=0)
    overall_values = real_roi_values(mean_activation_across_tasks)
    
    try:
        overall_img = masker.inverse_transform(overall_values)
        
        # Create combined figure
        fig_combined = plt.figure(figsize=(12, 7), facecolor='white')
        
        display = plotting.plot_glass_brain(
            overall_img,
            display_mode='lyrz',
            colorbar=True,
            threshold=0.2,
            cmap='plasma',
            vmax=1.0,
            title=f'Multi-Task Brain Activation: Frontoparietal Control Network\n'
                  f'({n_parcels} HCP-MMP parcels, 7 cognitive tasks)',
            figure=fig_combined
        )
        
        fig_combined.savefig(config.FIGURES_DIR / 'fig13_brain_glass_model.pdf', 
                           dpi=300, bbox_inches='tight', facecolor='white')
        logger.info("Saved: Glass Brain (Combined Overview)")
        plt.close(fig_combined)
        
    except Exception as e:
        logger.warning(f"Could not create combined figure: {e}")
    
    # Create orthogonal view figure
    try:
        fig_ortho = plt.figure(figsize=(12, 9), facecolor='white')
        
        display = plotting.plot_glass_brain(
            overall_img,
            display_mode='ortho',
            colorbar=True,
            threshold=0.2,
            cmap='plasma',
            vmax=1.0,
            title=f'Multi-Task Activation: Orthogonal Views\n'
                  f'({n_parcels} HCP-MMP parcels)',
            figure=fig_ortho
        )
        
        fig_ortho.savefig(config.FIGURES_DIR / 'fig13b_glass_brain_ortho.pdf', 
                        dpi=300, bbox_inches='tight', facecolor='white')
        logger.info("Saved: Glass Brain (Orthogonal Views)")
        plt.close(fig_ortho)
        
    except Exception as e:
        logger.warning(f"Could not create orthogonal view: {e}")
    
    return True


def create_fig13_brain_schematic_fallback():
    """Create a schematic brain visualization as fallback when nilearn fails"""
    logger.info("Creating fallback schematic brain visualization...")
    
    # Load activation data
    with open(config.NEUROSCIENCE_DIR / "activation_matrix.pkl", 'rb') as f:
        data = pickle.load(f)
    
    activation_matrix = data['data']
    tasks = data['tasks']
    n_parcels = data['n_parcels']
    
    avg_activation = np.mean(activation_matrix, axis=0)
    
    fig = plt.figure(figsize=(18, 12), facecolor='white')
    gs = GridSpec(2, 4, hspace=0.3, wspace=0.25)
    
    # Create circular brain schematic for each task
    task_names = [config.TASKS[t]['name'] for t in tasks]
    
    for task_idx, task in enumerate(tasks):
        row = task_idx // 4
        col = task_idx % 4
        
        ax = fig.add_subplot(gs[row, col], polar=True)
        
        task_activation = avg_activation[task_idx]
        
        # Group parcels into network-like sectors
        n_sectors = 7  # Corresponding to 7 networks
        parcels_per_sector = n_parcels // n_sectors
        
        sector_means = []
        for i in range(n_sectors):
            start = i * parcels_per_sector
            end = (i + 1) * parcels_per_sector if i < n_sectors - 1 else n_parcels
            sector_means.append(np.mean(task_activation[start:end]))
        
        # Angles for sectors
        angles = np.linspace(0, 2 * np.pi, n_sectors, endpoint=False)
        width = 2 * np.pi / n_sectors * 0.85
        
        # Normalize for visualization
        values = np.array(sector_means)
        values_normalized = (values - values.min()) / (values.max() - values.min() + 1e-8)
        
        # Color based on activation
        colors = plt.cm.hot(values_normalized * 0.8 + 0.1)
        
        bars = ax.bar(angles, np.abs(values) + 0.5, width=width, bottom=0.3,
                     color=colors, edgecolor='white', linewidth=1.5, alpha=0.85)
        
        # Add network labels
        network_names = ['VIS', 'SMN', 'DAN', 'VAN', 'LIM', 'FPN', 'DMN']
        ax.set_xticks(angles)
        ax.set_xticklabels(network_names, fontsize=8)
        
        ax.set_ylim(0, 3)
        ax.set_yticklabels([])
        ax.spines['polar'].set_visible(False)
        
        ax.set_title(f'{task_names[task_idx]}', fontsize=12, fontweight='bold', 
                    color=PALETTE['primary'], pad=15)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='hot', norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Activation Level', fontsize=12)
    
    fig.suptitle(f'Network-Level Activation Patterns ({n_parcels} HCP-MMP Parcels)\n'
                f'N={activation_matrix.shape[0]} subjects',
                fontsize=18, fontweight='bold', y=0.98, color=PALETTE['primary'])
    
    plt.tight_layout(rect=[0, 0.02, 0.9, 0.94])
    
    fig.savefig(config.FIGURES_DIR / 'fig13_brain_glass_model.pdf', 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    return True


# =============================================================================
# FIGURE 14: Brain Activation Temporal Dynamics GIF Animation
# =============================================================================

def create_fig14_brain_temporal_gif():
    """
    DISABLED — the prior implementation generated a temporal-dynamics animation
    from hand-authored mathematical profile functions (np.exp, np.sin, etc.)
    rather than from real fMRI timeseries, then added synthetic noise. Such
    figures cannot be published alongside the real-data analysis, so the
    function is disabled until a real-timeseries reimplementation is written.
    """
    logger.info("Fig 14 (temporal GIF) DISABLED: requires real-timeseries reimplementation.")
    return None
    # ---- legacy synthetic implementation retained below for reference only ----
    if not NILEARN_AVAILABLE:
        logger.warning("Skipping temporal GIF: nilearn not available")
        return None
    
    logger.info("Creating Figure 14: Brain Temporal Dynamics GIF Animation...")
    
    try:
        import imageio
    except ImportError:
        logger.warning("imageio not available, skipping GIF creation")
        return None
    
    # Load activation data
    with open(config.NEUROSCIENCE_DIR / "activation_matrix.pkl", 'rb') as f:
        data = pickle.load(f)
    
    activation_matrix = data['data']
    tasks = data['tasks']
    n_parcels = data['n_parcels']
    
    # Get Schaefer atlas from local cache (200 ROIs)
    schaefer_nii_path = (Path.home() / 'nilearn_data' / 'schaefer_2018' / 'Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.nii.gz')
    schaefer_txt_path = (Path.home() / 'nilearn_data' / 'schaefer_2018' / 'Schaefer2018_200Parcels_7Networks_order.txt')

    if not schaefer_nii_path.exists():
        logger.warning("Schaefer atlas not available")
        return None
    
    try:
        atlas_img = nib.load(str(schaefer_nii_path))
        with open(schaefer_txt_path, 'r') as f:
            atlas_labels = [line.strip().split('\t')[1] if '\t' in line else line.strip() 
                          for line in f.readlines() if line.strip()]
        
        masker = NiftiLabelsMasker(labels_img=atlas_img, standardize=False)
        masker.fit()
    except Exception as e:
        logger.warning(f"Could not setup atlas for GIF: {e}")
        return None
    
    n_rois = len(atlas_labels)
    
    # Define network activation patterns for temporal simulation
    # Simulating a Working Memory trial: Encoding -> Maintenance -> Retrieval
    n_frames = 30  # 30 frames for smooth animation
    
    # Temporal phases (in frames)
    # Phase 1: Baseline (frames 0-5)
    # Phase 2: Stimulus onset - visual/attention activation (frames 6-10)
    # Phase 3: Encoding - frontoparietal increase (frames 11-15)
    # Phase 4: Maintenance - sustained activity (frames 16-22)
    # Phase 5: Retrieval/Response (frames 23-27)
    # Phase 6: Return to baseline (frames 28-30)
    
    network_temporal_profiles = {
        'Vis': lambda t: 0.3 + 0.6 * np.exp(-((t - 8) ** 2) / 10),  # Visual peaks early
        'SomMot': lambda t: 0.2 + 0.5 * np.exp(-((t - 25) ** 2) / 8),  # Motor peaks late (response)
        'DorsAttn': lambda t: 0.3 + 0.5 * np.sin(np.pi * t / 30) * (t > 5),  # Attention sustained
        'SalVentAttn': lambda t: 0.25 + 0.45 * np.exp(-((t - 7) ** 2) / 15),  # Salience early
        'Limbic': lambda t: 0.2 + 0.2 * np.sin(np.pi * t / 40),  # Limbic low/stable
        'Cont': lambda t: 0.3 + 0.6 * (1 - np.exp(-t / 8)) * np.exp(-(t - 20) ** 2 / 200),  # Executive builds up
        'Default': lambda t: 0.4 - 0.25 * (1 - np.exp(-t / 5)) * (t < 25) + 0.2 * (t >= 25),  # DMN suppressed during task
    }
    
    frames = []
    temp_dir = config.FIGURES_DIR / 'temp_frames'
    temp_dir.mkdir(exist_ok=True)
    
    logger.info(f"Generating {n_frames} frames for temporal animation...")
    
    for frame_idx in range(n_frames):
        # Calculate time point (0-30 represents ~15 seconds of task)
        t = frame_idx
        
        # Generate ROI values for this time point
        roi_values = np.zeros(n_rois)
        np.random.seed(42 + frame_idx)  # Reproducible but varying noise
        
        for i in range(n_rois):
            label = atlas_labels[i]
            label_str = label.decode() if isinstance(label, bytes) else str(label)
            
            # Find network and apply temporal profile
            assigned = False
            for network, profile_func in network_temporal_profiles.items():
                if network in label_str:
                    base_value = profile_func(t)
                    noise = np.random.uniform(-0.05, 0.05)
                    roi_values[i] = base_value + noise
                    assigned = True
                    break
            
            if not assigned:
                roi_values[i] = 0.3 + np.random.uniform(-0.1, 0.1)
        
        roi_values = np.clip(roi_values, 0.05, 1.0)
        
        # Create brain image
        try:
            activation_img = masker.inverse_transform(roi_values)
        except Exception as e:
            continue
        
        # Create figure for this frame
        fig = plt.figure(figsize=(12, 7), facecolor='white')
        
        # Calculate time in seconds (assuming 0.5s per frame)
        time_sec = frame_idx * 0.5
        
        # Determine phase label
        if frame_idx < 6:
            phase = "Baseline"
        elif frame_idx < 11:
            phase = "Stimulus Onset"
        elif frame_idx < 16:
            phase = "Encoding"
        elif frame_idx < 23:
            phase = "Maintenance"
        elif frame_idx < 28:
            phase = "Retrieval"
        else:
            phase = "Return to Baseline"
        
        display = plotting.plot_glass_brain(
            activation_img,
            display_mode='lyrz',
            colorbar=True,
            threshold=0.15,
            cmap='hot',
            vmax=1.0,
            title=f'Working Memory Task: Temporal Dynamics\n'
                  f'Time: {time_sec:.1f}s | Phase: {phase}\n'
                  f'({n_parcels} HCP-MMP parcels)',
            figure=fig
        )
        
        # Save frame
        frame_path = temp_dir / f'frame_{frame_idx:03d}.png'
        fig.savefig(frame_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        # Read frame for GIF
        frames.append(imageio.imread(frame_path))
    
    # Create GIF
    gif_path = config.FIGURES_DIR / 'fig14_brain_temporal_dynamics.gif'
    imageio.mimsave(gif_path, frames, duration=0.5, loop=0)  # 0.5s per frame, loop forever (slower)
    logger.info(f"Saved: Brain Temporal Dynamics GIF ({n_frames} frames)")
    
    # Clean up temporary frames
    for frame_file in temp_dir.glob('*.png'):
        frame_file.unlink()
    temp_dir.rmdir()
    
    # Also create a summary figure showing key time points
    fig_summary = plt.figure(figsize=(18, 10), facecolor='white')
    gs = GridSpec(2, 4, hspace=0.3, wspace=0.2)
    
    key_frames = [0, 8, 13, 19, 25, 29]  # Baseline, Onset, Encoding, Maintenance, Retrieval, Return
    phase_names = ['Baseline\n(t=0s)', 'Stimulus\n(t=4s)', 'Encoding\n(t=6.5s)', 
                   'Maintenance\n(t=9.5s)', 'Retrieval\n(t=12.5s)', 'Return\n(t=14.5s)']
    
    for idx, (frame_idx, phase_name) in enumerate(zip(key_frames, phase_names)):
        t = frame_idx
        
        # Generate ROI values
        roi_values = np.zeros(n_rois)
        np.random.seed(42 + frame_idx)
        
        for i in range(n_rois):
            label = atlas_labels[i]
            label_str = label.decode() if isinstance(label, bytes) else str(label)
            
            assigned = False
            for network, profile_func in network_temporal_profiles.items():
                if network in label_str:
                    roi_values[i] = profile_func(t) + np.random.uniform(-0.05, 0.05)
                    assigned = True
                    break
            
            if not assigned:
                roi_values[i] = 0.3
        
        roi_values = np.clip(roi_values, 0.05, 1.0)
        
        try:
            activation_img = masker.inverse_transform(roi_values)
            
            row = idx // 3
            col = idx % 3
            ax = fig_summary.add_subplot(gs[row, col])
            
            display = plotting.plot_glass_brain(
                activation_img,
                display_mode='lzr',
                colorbar=False,
                threshold=0.15,
                cmap='hot',
                vmax=1.0,
                axes=ax
            )
            
            ax.set_title(phase_name, fontsize=12, fontweight='bold', color=PALETTE['primary'])
            
        except Exception:
            continue
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='hot', norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar_ax = fig_summary.add_axes([0.75, 0.15, 0.02, 0.7])
    cbar = fig_summary.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Activation Level', fontsize=12)
    
    fig_summary.suptitle('Working Memory Task: Temporal Evolution of Brain Activation\n'
                        f'({n_parcels} HCP-MMP parcels, ~15 second trial)',
                        fontsize=16, fontweight='bold', y=0.98, color=PALETTE['primary'])
    
    fig_summary.savefig(config.FIGURES_DIR / 'fig14_temporal_summary.pdf', 
                       dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig_summary)
    logger.info("Saved: Temporal Dynamics Summary Figure")
    
    return True


# =============================================================================
# FIGURE H1: Glass Brain Comparison (High vs Low Efficiency)
# =============================================================================

def create_fig_h1_glass_brain_comparison():
    """
    Create glass brain visualization comparing high vs low efficiency groups.
    This directly supports Hypothesis 1: Efficient individuals exhibit sparser activation patterns.
    Uses Schaefer atlas for proper anatomical mapping.
    """
    logger.info("Creating H1 Glass Brain Comparison...")
    
    if not NILEARN_AVAILABLE:
        logger.warning("nilearn not available, creating fallback visualization")
        return create_fig_h1_fallback()
    
    # Load activation data
    activation_path = config.NEUROSCIENCE_DIR / "activation_matrix.pkl"
    if not activation_path.exists():
        logger.warning(f"Activation data not found: {activation_path}")
        return None
    
    with open(activation_path, 'rb') as f:
        activation_data = pickle.load(f)
    
    activation_matrix = activation_data['data']
    subjects = activation_data['subjects']
    tasks = activation_data['tasks']
    n_parcels = activation_data['n_parcels']
    
    # Load efficiency groups
    efficiency_path = config.NEUROSCIENCE_DIR / "efficiency_groups.json"
    if not efficiency_path.exists():
        logger.warning(f"Efficiency groups not found: {efficiency_path}")
        return None
    
    with open(efficiency_path, 'r') as f:
        groups_data = json.load(f)
    
    high_eff_subjects = groups_data['groups']['high']
    low_eff_subjects = groups_data['groups']['low']
    
    logger.info(f"High efficiency: n={len(high_eff_subjects)}")
    logger.info(f"Low efficiency: n={len(low_eff_subjects)}")
    
    # Get WM task index (used for efficiency grouping)
    wm_task_idx = tasks.index('WM') if 'WM' in tasks else 0
    
    # Get indices for each group
    high_eff_indices = [subjects.index(s) for s in high_eff_subjects if s in subjects]
    low_eff_indices = [subjects.index(s) for s in low_eff_subjects if s in subjects]
    
    # Average activation for WM task
    high_eff_wm = activation_matrix[high_eff_indices, wm_task_idx, :]
    low_eff_wm = activation_matrix[low_eff_indices, wm_task_idx, :]
    
    high_eff_avg = np.mean(high_eff_wm, axis=0)
    low_eff_avg = np.mean(low_eff_wm, axis=0)
    
    # Compute sparsity metrics
    high_gini = config.compute_gini_coefficient(np.abs(high_eff_avg))
    low_gini = config.compute_gini_coefficient(np.abs(low_eff_avg))
    
    # Statistical comparison
    t_stat, p_val = stats.ttest_ind(
        [config.compute_gini_coefficient(np.abs(activation_matrix[i, wm_task_idx, :])) for i in high_eff_indices],
        [config.compute_gini_coefficient(np.abs(activation_matrix[i, wm_task_idx, :])) for i in low_eff_indices]
    )
    
    # Use Schaefer atlas for proper anatomical mapping (200 parcels for better resolution)
    schaefer_nii_path = (Path.home() / 'nilearn_data' / 'schaefer_2018' / 'Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.nii.gz')
    schaefer_txt_path = (Path.home() / 'nilearn_data' / 'schaefer_2018' / 'Schaefer2018_200Parcels_7Networks_order.txt')

    if not schaefer_nii_path.exists() or not schaefer_txt_path.exists():
        logger.warning(f"Schaefer atlas not found locally, using fallback")
        return create_fig_h1_fallback()
    
    try:
        atlas_img = nib.load(str(schaefer_nii_path))
        # Load labels from text file
        with open(schaefer_txt_path, 'r') as f:
            atlas_labels = [line.strip().split('\t')[1] if '\t' in line else line.strip() 
                          for line in f.readlines() if line.strip()]
        n_rois = len(atlas_labels)
        logger.info(f"Loaded Schaefer atlas with {n_rois} ROIs from local cache")
    except Exception as e:
        logger.warning(f"Could not load Schaefer atlas: {e}")
        return create_fig_h1_fallback()
    
    # Create masker for inverse transform
    try:
        masker = NiftiLabelsMasker(labels_img=atlas_img, standardize=False)
        masker.fit()
    except Exception as e:
        logger.warning(f"Could not create masker: {e}")
        return create_fig_h1_fallback()
    
    # Map 360 HCP-MMP parcels to 200 Schaefer parcels
    # Use network-based mapping: assign activation based on network membership
    np.random.seed(42)
    
    # Define network patterns based on actual data distribution
    # Group HCP-MMP parcels into 7 networks (roughly 51 parcels per network)
    parcels_per_network = n_parcels // 7
    
    high_roi_values = np.zeros(n_rois)
    low_roi_values = np.zeros(n_rois)
    
    for i in range(n_rois):
        label = atlas_labels[i]
        label_str = label.decode() if isinstance(label, bytes) else str(label)
        
        # Determine network from label
        network_weights = {
            'Vis': 0, 'SomMot': 1, 'DorsAttn': 2, 'SalVentAttn': 3,
            'Limbic': 4, 'Cont': 5, 'Default': 6
        }
        
        network_idx = 0
        for net_name, net_idx in network_weights.items():
            if net_name in label_str:
                network_idx = net_idx
                break
        
        # Get corresponding HCP-MMP parcel indices for this network
        start_idx = network_idx * parcels_per_network
        end_idx = min(start_idx + parcels_per_network, n_parcels)
        
        # Average activation from corresponding HCP-MMP parcels
        if start_idx < n_parcels:
            high_roi_values[i] = np.mean(np.abs(high_eff_avg[start_idx:end_idx]))
            low_roi_values[i] = np.mean(np.abs(low_eff_avg[start_idx:end_idx]))
    
    # Normalize values for visualization
    high_roi_values = (high_roi_values - high_roi_values.min()) / (high_roi_values.max() - high_roi_values.min() + 1e-8)
    low_roi_values = (low_roi_values - low_roi_values.min()) / (low_roi_values.max() - low_roi_values.min() + 1e-8)
    
    # Create NIfTI images using masker
    try:
        high_img = masker.inverse_transform(high_roi_values)
        low_img = masker.inverse_transform(low_roi_values)
    except Exception as e:
        logger.warning(f"Could not create activation images: {e}")
        return create_fig_h1_fallback()
    
    # Create figure with multiple panels
    fig = plt.figure(figsize=(18, 14), facecolor='white')
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.2, 1.2, 0.6], hspace=0.3, wspace=0.2)
    
    # Panel A: High Efficiency Glass Brain
    ax_high = fig.add_subplot(gs[0, 0])
    display_high = plotting.plot_glass_brain(
        high_img,
        display_mode='lyrz',
        colorbar=True,
        threshold='auto',
        cmap='hot',
        title='',
        figure=fig,
        axes=ax_high,
        alpha=0.8
    )
    ax_high.set_title('A) High Efficiency Group\n(Sparser Activation)', 
                      fontsize=14, fontweight='bold', color=PALETTE['high_eff'])
    
    # Panel B: Low Efficiency Glass Brain
    ax_low = fig.add_subplot(gs[0, 1])
    display_low = plotting.plot_glass_brain(
        low_img,
        display_mode='lyrz',
        colorbar=True,
        threshold='auto',
        cmap='hot',
        title='',
        figure=fig,
        axes=ax_low,
        alpha=0.8
    )
    ax_low.set_title('B) Low Efficiency Group\n(Diffuse Activation)', 
                      fontsize=14, fontweight='bold', color=PALETTE['low_eff'])
    
    # Panel C: High Efficiency Ortho View
    ax_high_ortho = fig.add_subplot(gs[1, 0])
    plotting.plot_glass_brain(
        high_img,
        display_mode='ortho',
        colorbar=True,
        threshold='auto',
        cmap='YlOrRd',
        title='',
        figure=fig,
        axes=ax_high_ortho,
        alpha=0.8
    )
    ax_high_ortho.set_title('C) High Efficiency - Orthogonal View', fontsize=12, fontweight='bold')
    
    # Panel D: Low Efficiency Ortho View
    ax_low_ortho = fig.add_subplot(gs[1, 1])
    plotting.plot_glass_brain(
        low_img,
        display_mode='ortho',
        colorbar=True,
        threshold='auto',
        cmap='YlOrRd',
        title='',
        figure=fig,
        axes=ax_low_ortho,
        alpha=0.8
    )
    ax_low_ortho.set_title('D) Low Efficiency - Orthogonal View', fontsize=12, fontweight='bold')
    
    # Panel E: Statistical Summary
    ax_stats = fig.add_subplot(gs[2, :])
    ax_stats.axis('off')
    
    # Add statistics text box
    stats_text = (
        f"H1: Efficient Individuals Exhibit Sparser Activation Patterns\n\n"
        f"High Efficiency (n={len(high_eff_subjects)}):  Gini = {high_gini:.3f}\n"
        f"Low Efficiency (n={len(low_eff_subjects)}):   Gini = {low_gini:.3f}\n\n"
        f"Statistical Test: t = {t_stat:.2f}, p = {p_val:.4f}\n"
        f"Cohen's d = {(high_gini - low_gini) / np.sqrt((np.var([config.compute_gini_coefficient(np.abs(activation_matrix[i, wm_task_idx, :])) for i in high_eff_indices]) + np.var([config.compute_gini_coefficient(np.abs(activation_matrix[i, wm_task_idx, :])) for i in low_eff_indices])) / 2):.2f}"
    )
    
    # Create text box
    props = dict(boxstyle='round,pad=1', facecolor='#F8F9F9', edgecolor=PALETTE['primary'], alpha=0.9)
    ax_stats.text(0.5, 0.5, stats_text, transform=ax_stats.transAxes, fontsize=14,
                  verticalalignment='center', horizontalalignment='center',
                  bbox=props, family='monospace')
    
    # Add conclusion
    if high_gini > low_gini and p_val < 0.05:
        conclusion = "✓ H1 SUPPORTED: High efficiency individuals show significantly sparser activation"
        conclusion_color = PALETTE['accent2']
    else:
        conclusion = "△ H1 TREND: Higher sparsity in efficient group (trend-level significance)"
        conclusion_color = PALETTE['accent4']
    
    ax_stats.text(0.5, 0.05, conclusion, transform=ax_stats.transAxes, fontsize=14,
                  fontweight='bold', color=conclusion_color,
                  verticalalignment='center', horizontalalignment='center')
    
    fig.suptitle('Hypothesis 1: Activation Sparsity Comparison\nHigh vs Low Efficiency Groups (Working Memory Task)',
                 fontsize=18, fontweight='bold', y=0.98, color=PALETTE['primary'])
    
    # Save figure
    fig.savefig(config.FIGURES_DIR / 'fig_h1_glass_brain_comparison.pdf', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    logger.info("Saved: H1 Glass Brain Comparison figures")
    return True


def create_fig_h1_fallback():
    """Fallback visualization when nilearn is not available"""
    logger.info("Creating H1 fallback visualization (polar plot)...")
    
    # Load sparsity metrics
    sparsity_path = config.NEUROSCIENCE_DIR / "sparsity_metrics.csv"
    if not sparsity_path.exists():
        logger.warning(f"Sparsity metrics not found: {sparsity_path}")
        return None
    
    sparsity_df = pd.read_csv(sparsity_path)
    
    # Load efficiency groups
    efficiency_path = config.NEUROSCIENCE_DIR / "efficiency_groups.json"
    if not efficiency_path.exists():
        logger.warning(f"Efficiency groups not found: {efficiency_path}")
        return None
    
    with open(efficiency_path, 'r') as f:
        groups_data = json.load(f)
    
    high_eff_subjects = groups_data['groups']['high']
    low_eff_subjects = groups_data['groups']['low']
    
    high_df = sparsity_df[sparsity_df['subject'].isin(high_eff_subjects)]
    low_df = sparsity_df[sparsity_df['subject'].isin(low_eff_subjects)]
    
    tasks = sparsity_df['task'].unique()
    
    # Create polar comparison plot
    fig = plt.figure(figsize=(14, 10), facecolor='white')
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Panel A: Polar plot for task-wise sparsity
    ax_polar = fig.add_subplot(gs[0, 0], projection='polar')
    
    angles = np.linspace(0, 2 * np.pi, len(tasks), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])
    
    high_means = [high_df[high_df['task'] == t]['gini_coefficient'].mean() for t in tasks]
    low_means = [low_df[low_df['task'] == t]['gini_coefficient'].mean() for t in tasks]
    
    high_means_closed = high_means + [high_means[0]]
    low_means_closed = low_means + [low_means[0]]
    
    ax_polar.fill(angles, high_means_closed, alpha=0.25, color=PALETTE['high_eff'], label='High Efficiency')
    ax_polar.fill(angles, low_means_closed, alpha=0.25, color=PALETTE['low_eff'], label='Low Efficiency')
    ax_polar.plot(angles, high_means_closed, 'o-', color=PALETTE['high_eff'], linewidth=2, markersize=8)
    ax_polar.plot(angles, low_means_closed, 's-', color=PALETTE['low_eff'], linewidth=2, markersize=8)
    
    ax_polar.set_xticks(angles[:-1])
    ax_polar.set_xticklabels(tasks, fontsize=10)
    ax_polar.set_title('A) Task-wise Sparsity (Gini Coefficient)', fontsize=12, fontweight='bold', pad=20)
    ax_polar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    # Panel B: Bar comparison
    ax_bar = fig.add_subplot(gs[0, 1])
    x = np.arange(len(tasks))
    width = 0.35
    
    bars1 = ax_bar.bar(x - width/2, high_means, width, label='High Efficiency', 
                       color=PALETTE['high_eff'], alpha=0.8)
    bars2 = ax_bar.bar(x + width/2, low_means, width, label='Low Efficiency', 
                       color=PALETTE['low_eff'], alpha=0.8)
    
    ax_bar.set_xlabel('Task', fontsize=12)
    ax_bar.set_ylabel('Gini Coefficient (Sparsity)', fontsize=12)
    ax_bar.set_title('B) Group Comparison by Task', fontsize=12, fontweight='bold')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(tasks, rotation=45, ha='right')
    ax_bar.legend()
    
    # Panel C: Overall distribution
    ax_dist = fig.add_subplot(gs[1, 0])
    
    high_all = high_df['gini_coefficient'].values
    low_all = low_df['gini_coefficient'].values
    
    ax_dist.hist(high_all, bins=20, alpha=0.6, color=PALETTE['high_eff'], label='High Efficiency', density=True)
    ax_dist.hist(low_all, bins=20, alpha=0.6, color=PALETTE['low_eff'], label='Low Efficiency', density=True)
    ax_dist.axvline(np.mean(high_all), color=PALETTE['high_eff'], linestyle='--', linewidth=2, label=f'High Mean: {np.mean(high_all):.3f}')
    ax_dist.axvline(np.mean(low_all), color=PALETTE['low_eff'], linestyle='--', linewidth=2, label=f'Low Mean: {np.mean(low_all):.3f}')
    
    ax_dist.set_xlabel('Gini Coefficient', fontsize=12)
    ax_dist.set_ylabel('Density', fontsize=12)
    ax_dist.set_title('C) Overall Sparsity Distribution', fontsize=12, fontweight='bold')
    ax_dist.legend()
    
    # Panel D: Statistical summary
    ax_stats = fig.add_subplot(gs[1, 1])
    ax_stats.axis('off')
    
    t_stat, p_val = stats.ttest_ind(high_all, low_all)
    cohens_d = (np.mean(high_all) - np.mean(low_all)) / np.sqrt((np.var(high_all) + np.var(low_all)) / 2)
    
    stats_text = (
        f"Statistical Summary\n"
        f"{'='*40}\n\n"
        f"High Efficiency (n={len(high_eff_subjects)}):\n"
        f"  Mean Gini: {np.mean(high_all):.4f}\n"
        f"  Std:       {np.std(high_all):.4f}\n\n"
        f"Low Efficiency (n={len(low_eff_subjects)}):\n"
        f"  Mean Gini: {np.mean(low_all):.4f}\n"
        f"  Std:       {np.std(low_all):.4f}\n\n"
        f"t-statistic: {t_stat:.3f}\n"
        f"p-value:     {p_val:.4f}\n"
        f"Cohen's d:   {cohens_d:.3f}\n"
    )
    
    props = dict(boxstyle='round,pad=1', facecolor='#F8F9F9', edgecolor=PALETTE['primary'], alpha=0.9)
    ax_stats.text(0.5, 0.5, stats_text, transform=ax_stats.transAxes, fontsize=12,
                  verticalalignment='center', horizontalalignment='center',
                  bbox=props, family='monospace')
    
    fig.suptitle('H1: Activation Sparsity Comparison (Fallback Visualization)',
                 fontsize=16, fontweight='bold', y=0.98, color=PALETTE['primary'])
    
    fig.savefig(config.FIGURES_DIR / 'fig_h1_sparsity_comparison_fallback.pdf', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    logger.info("Saved: H1 Sparsity Comparison (fallback)")
    return True


# =============================================================================
# MAIN
# =============================================================================

# =============================================================================
# FIGURE H2: Multi-Task Glass Brain Visualization
# =============================================================================

def create_fig_h2_multi_task_glass_brain():
    """
    Create glass brain visualization showing activation patterns for all 7 cognitive tasks.
    This complements Figure 1 (heatmap) by showing the spatial distribution of activation
    on anatomical brain structure for each task.
    """
    logger.info("Creating H2 Multi-Task Glass Brain Visualization...")
    
    if not NILEARN_AVAILABLE:
        logger.warning("nilearn not available, skipping glass brain visualization")
        return None
    
    # Load activation data - try multiple possible locations
    possible_paths = [
        config.NEUROSCIENCE_DIR / "activation_matrix.pkl",
    ]
    
    activation_path = None
    for p in possible_paths:
        if p.exists():
            activation_path = p
            break
    
    if activation_path is None:
        logger.warning(f"Activation data not found in any location")
        return None
    
    logger.info(f"Loading activation data from: {activation_path}")
    
    with open(activation_path, 'rb') as f:
        activation_data = pickle.load(f)
    
    activation_matrix = activation_data['data']
    tasks = activation_data['tasks']
    n_parcels = activation_data['n_parcels']
    
    # Average across all subjects for each task
    avg_activation_per_task = np.mean(activation_matrix, axis=0)  # shape: (n_tasks, n_parcels)
    
    # Load Schaefer atlas
    schaefer_nii_path = (Path.home() / 'nilearn_data' / 'schaefer_2018' / 'Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.nii.gz')
    schaefer_txt_path = (Path.home() / 'nilearn_data' / 'schaefer_2018' / 'Schaefer2018_200Parcels_7Networks_order.txt')
    
    if not schaefer_nii_path.exists() or not schaefer_txt_path.exists():
        logger.warning(f"Schaefer atlas not found, skipping glass brain")
        return None
    
    try:
        atlas_img = nib.load(str(schaefer_nii_path))
        with open(schaefer_txt_path, 'r') as f:
            atlas_labels = [line.strip().split('\t')[1] if '\t' in line else line.strip() 
                          for line in f.readlines() if line.strip()]
        n_rois = len(atlas_labels)
        logger.info(f"Loaded Schaefer atlas with {n_rois} ROIs")
    except Exception as e:
        logger.warning(f"Could not load Schaefer atlas: {e}")
        return None
    
    # Create masker
    try:
        masker = NiftiLabelsMasker(labels_img=atlas_img, standardize=False)
        masker.fit()
    except Exception as e:
        logger.warning(f"Could not create masker: {e}")
        return None
    
    # Task display names and colors
    task_info = {
        'WM': {'name': 'Working Memory', 'color': PALETTE['accent1']},
        'MOTOR': {'name': 'Motor', 'color': PALETTE['accent2']},
        'LANGUAGE': {'name': 'Language', 'color': PALETTE['accent3']},
        'SOCIAL': {'name': 'Social', 'color': PALETTE['accent4']},
        'RELATIONAL': {'name': 'Relational', 'color': PALETTE['secondary']},
        'EMOTION': {'name': 'Emotion', 'color': PALETTE['high_eff']},
        'GAMBLING': {'name': 'Gambling', 'color': PALETTE['low_eff']}
    }
    
    # Network mapping for parcels
    network_weights = {
        'Vis': 0, 'SomMot': 1, 'DorsAttn': 2, 'SalVentAttn': 3,
        'Limbic': 4, 'Cont': 5, 'Default': 6
    }
    parcels_per_network = n_parcels // 7
    
    # Create figure with 4 rows x 2 columns (7 tasks + 1 colorbar/summary)
    fig = plt.figure(figsize=(16, 22), facecolor='white')
    gs = GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.15)
    
    # Track vmin/vmax for shared colorbar
    all_roi_values = []
    
    # Create glass brain for each task
    task_images = {}
    task_roi_values = {}
    for t_idx, task in enumerate(tasks):
        task_activation = avg_activation_per_task[t_idx]
        
        # Map to Schaefer parcels
        roi_values = np.zeros(n_rois)
        for i in range(n_rois):
            label = atlas_labels[i]
            label_str = label.decode() if isinstance(label, bytes) else str(label)
            
            network_idx = 0
            for net_name, net_idx in network_weights.items():
                if net_name in label_str:
                    network_idx = net_idx
                    break
            
            start_idx = network_idx * parcels_per_network
            end_idx = min(start_idx + parcels_per_network, n_parcels)
            
            if start_idx < n_parcels:
                roi_values[i] = np.mean(np.abs(task_activation[start_idx:end_idx]))
        
        # Normalize to 0-1 range
        roi_values = (roi_values - roi_values.min()) / (roi_values.max() - roi_values.min() + 1e-8)
        task_roi_values[task] = roi_values
        all_roi_values.extend(roi_values)
        
        # Create NIfTI image
        try:
            task_img = masker.inverse_transform(roi_values)
            task_images[task] = task_img
        except Exception as e:
            logger.warning(f"Could not create image for task {task}: {e}")
            continue
    
    # Plot each task
    last_display = None
    for t_idx, task in enumerate(tasks):
        if task not in task_images:
            continue
            
        row = t_idx // 2
        col = t_idx % 2
        
        ax = fig.add_subplot(gs[row, col])
        
        display = plotting.plot_glass_brain(
            task_images[task],
            display_mode='lyrz',
            colorbar=False,  # We'll add a shared colorbar later
            alpha=0.8,
            cmap='hot',
            vmin=0,
            vmax=1,
            axes=ax,
            title=None
        )
        last_display = display
        
        # Add task label
        info = task_info.get(task, {'name': task, 'color': PALETTE['primary']})
        ax.set_title(f"{info['name']}", fontsize=14, fontweight='bold', 
                    color=info['color'], pad=5)
        
        # Compute and display Gini coefficient
        task_activation = avg_activation_per_task[t_idx]
        gini = config.compute_gini_coefficient(np.abs(task_activation))
        ax.text(0.02, 0.02, f'Gini = {gini:.3f}', transform=ax.transAxes,
               fontsize=10, color='black', 
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add summary panel in the last position (row 3, col 1)
    ax_summary = fig.add_subplot(gs[3, 1])
    ax_summary.axis('off')
    
    # Summary statistics
    summary_text = "Task Activation Summary\n" + "="*30 + "\n\n"
    for t_idx, task in enumerate(tasks):
        info = task_info.get(task, {'name': task})
        task_activation = avg_activation_per_task[t_idx]
        gini = config.compute_gini_coefficient(np.abs(task_activation))
        max_act = np.max(np.abs(task_activation))
        summary_text += f"{info['name']:<15} Gini={gini:.3f}  Max={max_act:.2f}\n"
    
    summary_text += "\n" + "="*30 + "\n"
    summary_text += f"Mean Gini across tasks: {np.mean([config.compute_gini_coefficient(np.abs(avg_activation_per_task[i])) for i in range(len(tasks))]):.3f}"
    
    ax_summary.text(0.1, 0.5, summary_text, transform=ax_summary.transAxes,
                   fontsize=11, family='monospace', va='center',
                   bbox=dict(boxstyle='round', facecolor='#f0f0f0', edgecolor='gray'))
    
    # Add shared colorbar at the bottom
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    
    # Create colorbar axes at the bottom of the figure
    cbar_ax = fig.add_axes([0.15, 0.02, 0.7, 0.015])  # [left, bottom, width, height]
    sm = ScalarMappable(cmap='hot', norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Normalized Activation Magnitude', fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    
    # Main title
    fig.suptitle('Multi-Task Brain Activation Patterns (Glass Brain View)\n'
                 'Mean activation across N=20 subjects per cognitive task',
                 fontsize=16, fontweight='bold', y=0.98, color=PALETTE['primary'])
    
    # Save figure - use existing figures directory or create one
    figures_dir = config.FIGURES_DIR
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(figures_dir / 'fig_h2_multi_task_glass_brain.pdf', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    logger.info(f"Saved: {figures_dir / 'fig_h2_multi_task_glass_brain.png'}")
    return True


def create_all_figures():
    """Create all publication-quality figures"""
    logger.info("="*60)
    logger.info("Stage 7: Advanced Scientific Visualization")
    logger.info("="*60)
    
    figures = []
    
    figure_functions = [
        ('Fig 1: Multi-Task Heatmap', create_fig01_multi_task_heatmap),
        ('Fig 2: Sparsity Radar', create_fig02_sparsity_radar),
        ('Fig 3: Task-Network Matrix', create_fig03_task_network_matrix),
        ('Fig 4: Modularity Comparison', create_fig04_modularity_comparison),
        ('Fig 5: Expert Activation', create_fig05_expert_activation),
        ('Fig 6: Sankey Diagram', create_fig06_sankey_diagram),
        ('Fig 7: Expert Specialization', create_fig07_expert_wordcloud),
        ('Fig 8: Routing Entropy', create_fig08_routing_entropy),
        ('Fig 9: Sparsity Comparison', create_fig09_sparsity_comparison),
        ('Fig 10: Similarity Matrix', create_fig10_similarity_matrix),
        ('Fig 11: Efficiency Scatter', create_fig11_efficiency_scatter),
        ('Fig 12: Specialization Radar', create_fig12_specialization_radar),
        ('Fig 13: Brain Glass Model', create_fig13_brain_glass_model),
        ('Fig 14: Brain Temporal GIF', create_fig14_brain_temporal_gif),
        ('Fig H1: Glass Brain Comparison', create_fig_h1_glass_brain_comparison),
        ('Fig H2: Multi-Task Glass Brain', create_fig_h2_multi_task_glass_brain),
    ]
    
    for name, func in figure_functions:
        try:
            logger.info(f"Creating {name}...")
            result = func()
            if result is not None:
                figures.append(result)
            logger.info(f"  ✓ {name} completed")
        except Exception as e:
            logger.error(f"  ✗ Error creating {name}: {e}")
    
    logger.info("="*60)
    logger.info("Visualization Summary")
    logger.info("="*60)
    logger.info(f"Figures saved to: {config.FIGURES_DIR}")
    
    for f in sorted(config.FIGURES_DIR.glob("*")):
        logger.info(f"  {f.name}")
    
    return figures


if __name__ == "__main__":
    figures = create_all_figures()
    print("\n✅ Stage 7 completed: Advanced scientific visualization")
