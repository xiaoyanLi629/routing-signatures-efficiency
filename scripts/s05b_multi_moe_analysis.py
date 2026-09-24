#!/usr/bin/env python3
"""
=============================================================================
Stage 5b: Multi-Model MoE Analysis with Real Models and Datasets
=============================================================================

Extended analysis comparing multiple MoE architectures using REAL model inference:
- Google Switch Transformer (8 experts, Top-1)
- Qwen MoE (60 experts, Top-4)  
- DeepSeek MoE (64 experts, Top-6)

This script uses:
- Real HuggingFace datasets: RACE, SST-2, GSM8K, SNLI
- Real model inference to extract routing patterns
- GPU acceleration for efficient processing

Tests H5: MoE models show sparse expert activation at aggregate level
Tests H6: MoE experts specialize for different input types
Tests H7: Brain-model structural similarity

Outputs:
    - multi_moe_comparison.json: Cross-model comparison results
    - multi_moe_summary.csv: Sparsity metrics for all models
    - multi_moe_figures/: Comparative visualizations
"""

import sys
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict
import pickle
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns

import torch
import gc

from configs import config

# Ensure directories exist
config.ensure_run_directories()
logger = config.setup_logging('multi_moe_analysis')

# =============================================================================
# HuggingFace Mirror for China
# =============================================================================

HF_MIRROR = "https://hf-mirror.com"

def setup_hf_mirror():
    """Setup HuggingFace mirror for faster access in China"""
    os.environ['HF_ENDPOINT'] = HF_MIRROR
    logger.info(f"Using HuggingFace mirror: {HF_MIRROR}")

# =============================================================================
# MODEL CONFIGURATIONS
# =============================================================================

MOE_MODELS = [
    {
        'name': 'google/switch-base-8',
        'short_name': 'Switch-8',
        'display_name': 'Google Switch\n(8 experts, Top-1)',
        'n_experts': 8,
        'routing': 'top-1',
        'top_k': 1,
        'size': '220M',
        'color': '#4285F4',  # Google Blue
        'organization': 'Google',
        'model_type': 'switch',
    },
    {
        'name': 'Qwen/Qwen1.5-MoE-A2.7B',
        'short_name': 'Qwen-MoE',
        'display_name': 'Qwen MoE\n(60 experts, Top-4)',
        'n_experts': 60,
        'routing': 'top-4',
        'top_k': 4,
        'size': '2.7B active',
        'color': '#FF6A00',  # Alibaba Orange
        'organization': 'Alibaba',
        'model_type': 'qwen_moe',
    },
    {
        'name': 'deepseek-ai/deepseek-moe-16b-base',
        'short_name': 'DeepSeek-16B',
        'display_name': 'DeepSeek MoE\n(64 experts, Top-6)',
        'n_experts': 64,
        'routing': 'top-6',
        'top_k': 6,
        'size': '16B',
        'color': '#00D4AA',  # DeepSeek Teal
        'organization': 'DeepSeek',
        'model_type': 'deepseek_moe',
    },
]

# Input categories (same as original)
INPUT_CATEGORIES = config.MOE_CONFIG['input_categories']

# =============================================================================
# DATASET LOADING FROM HUGGINGFACE
# =============================================================================

def load_samples_from_huggingface(n_samples_per_category=100):
    """
    Load text samples from standard HuggingFace datasets.
    Maps datasets to cognitive categories paralleling fMRI tasks.
    """
    setup_hf_mirror()
    
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("HuggingFace datasets not installed. Run: pip install datasets")
        return create_fallback_samples()
    
    samples = {cat: [] for cat in ['scientific', 'social', 'emotional', 
                                    'procedural', 'narrative', 'mathematical', 'risk_decision']}
    
    logger.info("Loading samples from HuggingFace datasets...")
    
    # 1. RACE dataset for scientific and narrative
    try:
        logger.info("  Loading RACE dataset...")
        race = load_dataset("ehovy/race", "all", split="train[:5000]", trust_remote_code=True)
        
        science_keywords = ['science', 'experiment', 'research', 'study', 'data', 'theory', 
                           'hypothesis', 'evidence', 'analysis', 'result', 'conclusion',
                           'temperature', 'energy', 'chemical', 'physics', 'biology']
        narrative_keywords = ['story', 'once', 'life', 'journey', 'adventure', 'character',
                             'lived', 'happened', 'told', 'remember', 'dream', 'felt', 'day']
        procedural_keywords = ['step', 'first', 'then', 'next', 'finally', 'instructions', 
                              'how to', 'procedure', 'method', 'follow', 'begin']
        
        for item in race:
            text = item['article'][:500]
            text_lower = text.lower()
            
            if len(samples['scientific']) < n_samples_per_category:
                if any(kw in text_lower for kw in science_keywords):
                    samples['scientific'].append(text)
            
            if len(samples['narrative']) < n_samples_per_category:
                if any(kw in text_lower for kw in narrative_keywords):
                    samples['narrative'].append(text)
            
            if len(samples['procedural']) < n_samples_per_category:
                if any(kw in text_lower for kw in procedural_keywords):
                    samples['procedural'].append(text)
            
            if all(len(samples[k]) >= n_samples_per_category for k in ['scientific', 'narrative', 'procedural']):
                break
                
        logger.info(f"    Scientific: {len(samples['scientific'])} samples")
        logger.info(f"    Narrative: {len(samples['narrative'])} samples")
        logger.info(f"    Procedural: {len(samples['procedural'])} samples")
    except Exception as e:
        logger.warning(f"  Could not load RACE: {e}")
    
    # 2. SST-2 for emotional content
    try:
        logger.info("  Loading SST-2 dataset...")
        sst2 = load_dataset("stanfordnlp/sst2", split="train[:2000]", trust_remote_code=True)
        
        for item in sst2:
            if len(samples['emotional']) >= n_samples_per_category:
                break
            text = item['sentence']
            if len(text) > 20:
                samples['emotional'].append(text)
        
        logger.info(f"    Emotional: {len(samples['emotional'])} samples")
    except Exception as e:
        logger.warning(f"  Could not load SST-2: {e}")
    
    # 3. SNLI for social reasoning
    try:
        logger.info("  Loading SNLI dataset...")
        snli = load_dataset("stanfordnlp/snli", split="train[:5000]", trust_remote_code=True)
        
        social_keywords = ['people', 'person', 'friend', 'family', 'team', 'group',
                          'said', 'told', 'asked', 'helped', 'talked', 'meeting', 'man', 'woman']
        risk_keywords = ['decide', 'choice', 'risk', 'option', 'chance', 'bet',
                        'invest', 'gamble', 'uncertain', 'probability', 'outcome', 'money']
        
        for item in snli:
            if item['label'] == -1:
                continue
            premise = item['premise']
            premise_lower = premise.lower()
            
            if len(samples['social']) < n_samples_per_category:
                if any(kw in premise_lower for kw in social_keywords):
                    samples['social'].append(premise)
            
            if len(samples['risk_decision']) < n_samples_per_category:
                if any(kw in premise_lower for kw in risk_keywords):
                    samples['risk_decision'].append(premise)
            
            if len(samples['social']) >= n_samples_per_category and \
               len(samples['risk_decision']) >= n_samples_per_category:
                break
        
        logger.info(f"    Social: {len(samples['social'])} samples")
        logger.info(f"    Risk/Decision: {len(samples['risk_decision'])} samples")
    except Exception as e:
        logger.warning(f"  Could not load SNLI: {e}")
    
    # 4. GSM8K for mathematical reasoning
    try:
        logger.info("  Loading GSM8K dataset...")
        gsm8k = load_dataset("openai/gsm8k", "main", split="train", trust_remote_code=True)
        
        for item in gsm8k:
            if len(samples['mathematical']) >= n_samples_per_category:
                break
            question = item['question']
            samples['mathematical'].append(question)
        
        logger.info(f"    Mathematical: {len(samples['mathematical'])} samples")
    except Exception as e:
        logger.warning(f"  Could not load GSM8K: {e}")
    
    # Fill gaps with fallback samples
    fallback = create_fallback_samples()
    for category in samples:
        if len(samples[category]) < 10:
            logger.warning(f"  {category} has few samples ({len(samples[category])}), using fallback")
            samples[category].extend(fallback.get(category, []))
        samples[category] = samples[category][:n_samples_per_category]
    
    total = sum(len(v) for v in samples.values())
    logger.info(f"Total samples loaded: {total}")
    
    return samples


def create_fallback_samples():
    """Fallback samples if HuggingFace datasets are unavailable."""
    samples = {
        'scientific': [
            "The hypothesis suggests that quantum entanglement enables information transfer.",
            "Photosynthesis converts carbon dioxide and water into glucose through biochemical processes.",
            "Neural plasticity allows the brain to reorganize by forming new synaptic connections.",
            "Climate models predict increasing global temperatures due to greenhouse gas concentrations.",
            "The double helix structure of DNA was discovered through X-ray crystallography.",
        ] * 10,
        'social': [
            "I noticed you seemed upset at the party yesterday. Is everything okay?",
            "The team dynamics improved after we addressed communication issues.",
            "He felt betrayed when his colleague took credit for their project.",
            "Building trust requires consistent honesty and reliability over time.",
            "The community came together to support the family during difficult times.",
        ] * 10,
        'emotional': [
            "The overwhelming joy she felt when reuniting with her family brought tears.",
            "Grief consumed him completely after losing his beloved companion.",
            "The crippling anxiety before the interview made her hands tremble.",
            "Pure elation filled the stadium as the team scored the winning goal.",
            "Anger surged through him when he discovered the deliberate deception.",
        ] * 10,
        'procedural': [
            "First, preheat the oven to 350 degrees Fahrenheit.",
            "To change a tire: loosen the lug nuts, jack up the car, remove the flat.",
            "Install the software by running the setup wizard and following prompts.",
            "Begin by cleaning the surface thoroughly, then apply primer evenly.",
            "Connect the device using USB, wait for driver installation, launch app.",
        ] * 10,
        'narrative': [
            "Once upon a time, in a kingdom far away, lived a young princess.",
            "The detective examined the crime scene with meticulous care.",
            "As the sun set over the mountains, the travelers made camp.",
            "The old sailor entertained children with tales of his voyages.",
            "In the year 2150, humanity had colonized Mars.",
        ] * 10,
        'mathematical': [
            "Calculate the derivative of f(x) = 3x² + 2x - 5.",
            "If a train travels 120 miles in 2 hours, what is its velocity?",
            "Solve the equation 2x + 5 = 15 for x.",
            "What is the probability of rolling 7 with two dice?",
            "Find the area of a circle with radius 5 using A = πr².",
        ] * 10,
        'risk_decision': [
            "Should I invest in volatile stocks or choose safer bonds?",
            "The gambler faced a decision: double down or walk away.",
            "Weighing the risks of surgery against living with chronic pain.",
            "The startup founder had to accept a lower offer or continue independently.",
            "Insurance decisions involve balancing premium costs against protection.",
        ] * 10,
    }
    return samples


# =============================================================================
# MODEL LOADING AND ROUTING EXTRACTION
# =============================================================================

def load_model_and_tokenizer(model_config):
    """Load a specific MoE model and tokenizer."""
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
    
    model_name = model_config['name']
    model_type = model_config['model_type']
    
    logger.info(f"Loading model: {model_name}")
    
    # Set up cache directory
    cache_dir = PROJECT_DIR / "models" / "huggingface_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        
        # Set pad token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        if model_type == 'switch':
            # Switch Transformer is Seq2Seq
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
        else:
            # Qwen and DeepSeek are CausalLM
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
        
        model.eval()
        logger.info(f"Successfully loaded: {model_name}")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        return None, None


def extract_routing_patterns(model, tokenizer, model_config, texts, category):
    """
    Extract expert routing patterns from a MoE model.
    Uses hooks to capture router decisions during forward pass.
    """
    n_experts = model_config['n_experts']
    model_type = model_config['model_type']
    top_k = model_config['top_k']
    
    # Storage for routing decisions
    routing_decisions = []
    router_outputs = []
    
    def router_hook(module, input, output):
        """Hook to capture router output"""
        # Different models have different output formats
        if isinstance(output, tuple):
            for o in output:
                if isinstance(o, torch.Tensor):
                    router_outputs.append(o.detach().cpu())
        elif isinstance(output, torch.Tensor):
            router_outputs.append(output.detach().cpu())
    
    # Register hooks on router/gate modules
    hooks = []
    for name, module in model.named_modules():
        name_lower = name.lower()
        # Look for router/gate modules in different architectures
        if any(keyword in name_lower for keyword in ['router', 'gate', 'expert_gate']):
            hooks.append(module.register_forward_hook(router_hook))
            logger.debug(f"  Hook registered on: {name}")
    
    expert_counts = defaultdict(int)
    total_tokens = 0
    
    # Process texts in batches
    batch_size = 4 if model_type != 'switch' else 8
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        router_outputs.clear()
        
        try:
            # Tokenize
            inputs = tokenizer(
                batch_texts, 
                return_tensors="pt", 
                max_length=128,
                truncation=True, 
                padding=True
            )
            
            # Move to model's device
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                if model_type == 'switch':
                    # For Seq2Seq, we need decoder_input_ids
                    outputs = model(**inputs, decoder_input_ids=inputs['input_ids'][:, :1])
                else:
                    outputs = model(**inputs)
            
            # Process captured router outputs — REAL DATA ONLY.
            # If the forward hook did not capture router/gate output for this
            # batch, we skip it rather than falling back to a hash-based
            # expert assignment. Previous hash/uniform fallbacks fabricated
            # routing decisions and are not acceptable for publication.
            if router_outputs:
                for router_prob in router_outputs:
                    if router_prob.dim() >= 2:
                        probs = router_prob.view(-1, router_prob.shape[-1]).numpy()
                        for token_probs in probs:
                            if len(token_probs) > 0:
                                top_indices = np.argsort(token_probs)[-top_k:]
                                for idx in top_indices:
                                    expert_id = idx % n_experts
                                    expert_counts[expert_id] += 1
                                total_tokens += 1
            else:
                logger.warning(
                    f"  No router output captured for batch {i}-{i+batch_size} "
                    f"({model_name}); skipping batch. Fix hook registration "
                    f"or model before re-running."
                )

        except Exception as e:
            logger.error(
                f"  Error processing batch {i}-{i+batch_size} for {model_name}: "
                f"{e}. Skipping batch; no synthetic fallback."
            )
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Normalize counts
    total = sum(expert_counts.values())
    expert_frequency = {
        i: expert_counts.get(i, 0) / max(total, 1)
        for i in range(n_experts)
    }
    
    return {
        'expert_frequency': expert_frequency,
        'expert_counts': dict(expert_counts),
        'total_tokens': total_tokens,
        'category': category,
    }


def analyze_single_model_real(model_config, samples):
    """
    Analyze routing patterns for a single MoE model using real inference.
    """
    model_name = model_config['short_name']
    n_experts = model_config['n_experts']
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Analyzing {model_name} ({n_experts} experts)")
    logger.info(f"{'='*60}")
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(model_config)

    if model is None:
        raise RuntimeError(
            f"Could not load MoE model '{model_name}'. Real inference is "
            f"required; synthetic-routing fallback is disabled for research "
            f"integrity. Verify model is downloaded and accessible."
        )
    
    # Storage
    all_expert_counts = defaultdict(int)
    category_expert_mapping = defaultdict(lambda: defaultdict(int))
    total_tokens = 0
    
    # Process each category
    for cat_info in tqdm(INPUT_CATEGORIES, desc=f"Processing {model_name}"):
        category = cat_info['name']
        category_samples = samples.get(category, [])[:50]  # Limit samples for speed
        
        if len(category_samples) == 0:
            continue
        
        result = extract_routing_patterns(
            model, tokenizer, model_config, 
            category_samples, category
        )
        
        # Aggregate results
        for expert_id, count in result['expert_counts'].items():
            all_expert_counts[expert_id] += count
            category_expert_mapping[category][expert_id] += count
        
        total_tokens += result['total_tokens']
    
    # Clean up model to free memory
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    # Compute metrics
    total_count = sum(all_expert_counts.values())
    expert_frequency = {
        i: all_expert_counts.get(i, 0) / max(total_count, 1)
        for i in range(n_experts)
    }
    
    freq_array = np.array(list(expert_frequency.values()))
    
    # Gini coefficient
    gini = config.compute_gini_coefficient(freq_array)
    
    # Entropy-based sparsity
    freq_nonzero = freq_array[freq_array > 0]
    if len(freq_nonzero) > 0:
        entropy = -np.sum(freq_nonzero * np.log2(freq_nonzero + 1e-10))
    else:
        entropy = 0
    max_entropy = np.log2(n_experts)
    entropy_sparsity = 1 - (entropy / max_entropy) if max_entropy > 0 else 0
    
    # Effective number of experts
    effective_experts = 2 ** entropy if entropy > 0 else 1
    
    # Concentration ratio (top 10%)
    top_k_experts = max(1, n_experts // 10)
    sorted_freq = np.sort(freq_array)[::-1]
    concentration_ratio = np.sum(sorted_freq[:top_k_experts])
    
    # Per-category sparsity
    category_sparsity = {}
    for cat_info in INPUT_CATEGORIES:
        cat = cat_info['name']
        cat_counts = category_expert_mapping.get(cat, {})
        cat_freq = np.array([cat_counts.get(i, 0) for i in range(n_experts)])
        cat_total = cat_freq.sum()
        if cat_total > 0:
            cat_freq = cat_freq / cat_total
        category_sparsity[cat] = {
            'gini': float(config.compute_gini_coefficient(cat_freq)),
            'distribution': cat_freq.tolist(),
        }
    
    # Expert specialization
    expert_specialization = {}
    for expert_id in range(n_experts):
        category_weights = {}
        for cat_info in INPUT_CATEGORIES:
            cat = cat_info['name']
            category_weights[cat] = category_expert_mapping[cat].get(expert_id, 0)
        
        total = sum(category_weights.values())
        if total > 0:
            dominant = max(category_weights, key=category_weights.get)
            selectivity = category_weights[dominant] / total
            distribution = {k: v/total for k, v in category_weights.items()}
        else:
            dominant = 'none'
            selectivity = 1.0 / len(INPUT_CATEGORIES)
            distribution = {cat['name']: 1.0/len(INPUT_CATEGORIES) for cat in INPUT_CATEGORIES}
        
        expert_specialization[expert_id] = {
            'dominant_category': dominant,
            'selectivity': float(selectivity),
            'distribution': distribution,
        }
    
    # Summary statistics
    mean_selectivity = np.mean([e['selectivity'] for e in expert_specialization.values()])
    specialized_experts = sum(1 for e in expert_specialization.values() if e['selectivity'] > 0.5)
    
    logger.info(f"  Gini coefficient: {gini:.4f}")
    logger.info(f"  Effective experts: {effective_experts:.1f} / {n_experts}")
    logger.info(f"  Mean selectivity: {mean_selectivity:.4f}")
    
    return {
        'model': model_config,
        'n_experts': n_experts,
        'top_k': model_config['top_k'],
        'expert_frequency': expert_frequency,
        'sparsity_metrics': {
            'gini': float(gini),
            'entropy': float(entropy),
            'entropy_sparsity': float(entropy_sparsity),
            'effective_experts': float(effective_experts),
            'concentration_ratio': float(concentration_ratio),
            'uniform_baseline': 1.0 / n_experts,
        },
        'category_sparsity': category_sparsity,
        'expert_specialization': expert_specialization,
        'specialization_summary': {
            'mean_selectivity': float(mean_selectivity),
            'specialized_experts': specialized_experts,
            'specialization_rate': specialized_experts / n_experts,
        },
        'total_tokens_processed': total_tokens,
    }


# --- SIMULATED FALLBACK REMOVED for research integrity ---
# =============================================================================
# CROSS-MODEL COMPARISON
# =============================================================================

def compare_models(results_list):
    """Compare routing patterns across multiple MoE models."""
    logger.info("Performing cross-model comparison...")
    
    comparison = {
        'models': [],
        'sparsity_comparison': {},
        'specialization_comparison': {},
    }
    
    for result in results_list:
        model = result['model']
        metrics = result['sparsity_metrics']
        spec_summary = result['specialization_summary']
        
        comparison['models'].append({
            'name': model['short_name'],
            'n_experts': model['n_experts'],
            'top_k': model['top_k'],
            'gini': metrics['gini'],
            'entropy_sparsity': metrics['entropy_sparsity'],
            'effective_experts': metrics['effective_experts'],
            'concentration_ratio': metrics['concentration_ratio'],
            'mean_selectivity': spec_summary['mean_selectivity'],
            'specialization_rate': spec_summary['specialization_rate'],
        })
    
    gini_values = [r['sparsity_metrics']['gini'] for r in results_list]
    selectivity_values = [r['specialization_summary']['mean_selectivity'] for r in results_list]
    
    comparison['summary'] = {
        'gini_range': [float(min(gini_values)), float(max(gini_values))],
        'gini_mean': float(np.mean(gini_values)),
        'selectivity_range': [float(min(selectivity_values)), float(max(selectivity_values))],
        'selectivity_mean': float(np.mean(selectivity_values)),
        'most_sparse_model': results_list[np.argmax(gini_values)]['model']['short_name'],
        'most_specialized_model': results_list[np.argmax(selectivity_values)]['model']['short_name'],
    }
    
    # Compare with brain
    brain_gini = 0.458
    comparison['brain_comparison'] = {
        'brain_gini': brain_gini,
        'model_gini_mean': float(np.mean(gini_values)),
        'brain_vs_moe_ratio': float(brain_gini / np.mean(gini_values)) if np.mean(gini_values) > 0 else float('inf'),
        'interpretation': f"Brain is {brain_gini / np.mean(gini_values):.1f}x more sparse than average MoE model"
    }
    
    return comparison


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_multi_model_visualizations(results_list, comparison, output_dir):
    """Create comprehensive visualizations comparing multiple MoE models."""
    logger.info("Creating multi-model visualizations...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    colors = [r['model']['color'] for r in results_list]
    model_names = [r['model']['short_name'] for r in results_list]
    
    # =========================================================================
    # Figure 1: Sparsity Comparison (2 panels)
    # =========================================================================
    fig1, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Gini coefficient comparison
    ax = axes[0]
    gini_values = [r['sparsity_metrics']['gini'] for r in results_list]
    bars = ax.bar(model_names, gini_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.axhline(y=0.458, color='red', linestyle='--', linewidth=2, label='Brain (0.458)')
    ax.set_ylabel('Gini Coefficient', fontsize=12)
    ax.set_title('A) Aggregate Sparsity (Gini)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.set_ylim(0, 0.6)
    
    for bar, val in zip(bars, gini_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Effective experts
    ax = axes[1]
    eff_experts = [r['sparsity_metrics']['effective_experts'] for r in results_list]
    n_experts = [r['n_experts'] for r in results_list]
    
    x = np.arange(len(model_names))
    width = 0.35
    bars1 = ax.bar(x - width/2, n_experts, width, label='Total Experts', color='lightgray', edgecolor='black')
    bars2 = ax.bar(x + width/2, eff_experts, width, label='Effective Experts', color=colors, alpha=0.8, edgecolor='black')
    
    ax.set_ylabel('Number of Experts', fontsize=12)
    ax.set_title('B) Effective vs Total Experts', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    
    fig1.suptitle('Multi-Model MoE Sparsity Comparison', fontsize=16, fontweight='bold', y=1.02)
    fig1.tight_layout()
    fig1.savefig(output_dir / 'fig_multi_moe_sparsity_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig1)
    logger.info("Saved: fig_multi_moe_sparsity_comparison.png/svg")
    
    # =========================================================================
    # Figure 2: Expert Usage Distribution
    # =========================================================================
    n_models = len(results_list)
    fig2, axes = plt.subplots(1, n_models, figsize=(5*n_models, 5))
    if n_models == 1:
        axes = [axes]
    
    for idx, (result, ax) in enumerate(zip(results_list, axes)):
        model = result['model']
        freq = result['expert_frequency']
        n_exp = result['n_experts']
        
        sorted_items = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        expert_ids = [f"E{k}" for k, v in sorted_items]
        frequencies = [v for k, v in sorted_items]
        
        if n_exp > 20:
            expert_ids = expert_ids[:20]
            frequencies = frequencies[:20]
            ax.set_xlabel('Expert ID (Top 20)', fontsize=10)
        else:
            ax.set_xlabel('Expert ID', fontsize=10)
        
        bars = ax.bar(range(len(frequencies)), frequencies, color=model['color'], alpha=0.8)
        ax.axhline(y=1/n_exp, color='red', linestyle='--', linewidth=2, label=f'Uniform ({1/n_exp:.3f})')
        
        ax.set_ylabel('Activation Frequency', fontsize=10)
        ax.set_title(f"{model['short_name']}\n({n_exp} experts, top-{model['top_k']})", 
                     fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(expert_ids)))
        ax.set_xticklabels(expert_ids, rotation=45, ha='right', fontsize=8)
        ax.legend(loc='upper right', fontsize=8)
    
    fig2.suptitle('Expert Usage Distribution Across Models', fontsize=16, fontweight='bold', y=1.02)
    fig2.tight_layout()
    fig2.savefig(output_dir / 'fig_multi_moe_expert_distribution.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig2)
    logger.info("Saved: fig_multi_moe_expert_distribution.png/svg")
    
    # =========================================================================
    # Figure 3: Category-Expert Heatmaps
    # =========================================================================
    categories = [c['name'] for c in INPUT_CATEGORIES]
    
    fig3, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    if n_models == 1:
        axes = [axes]
    
    for idx, (result, ax) in enumerate(zip(results_list, axes)):
        model = result['model']
        n_exp = result['n_experts']
        n_show = min(n_exp, 20)
        
        matrix = np.zeros((len(categories), n_show))
        for c_idx, cat in enumerate(categories):
            cat_sparsity = result['category_sparsity'].get(cat, {})
            dist = cat_sparsity.get('distribution', [0] * n_exp)
            matrix[c_idx, :] = dist[:n_show]
        
        matrix = matrix / (matrix.sum(axis=1, keepdims=True) + 1e-10)
        
        sns.heatmap(matrix, ax=ax, cmap='YlOrRd', 
                    xticklabels=[f'E{i}' for i in range(matrix.shape[1])],
                    yticklabels=categories, cbar_kws={'shrink': 0.8})
        ax.set_title(f"{model['short_name']}", fontsize=12, fontweight='bold')
        ax.set_xlabel('Expert ID' + (' (first 20)' if n_exp > 20 else ''), fontsize=10)
        ax.set_ylabel('Input Category', fontsize=10)
    
    fig3.suptitle('Category-Expert Routing Patterns', fontsize=16, fontweight='bold', y=1.02)
    fig3.tight_layout()
    fig3.savefig(output_dir / 'fig_multi_moe_category_routing.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig3)
    logger.info("Saved: fig_multi_moe_category_routing.png/svg")
    
    # =========================================================================
    # Figure 4: Brain vs MoE Comparison (Single Panel - Gini Comparison)
    # =========================================================================
    fig4, ax = plt.subplots(figsize=(8, 6))
    
    # Gini comparison with brain
    all_gini = gini_values + [0.458]
    all_names = model_names + ['Human Brain']
    all_colors = colors + ['#E74C3C']
    
    bars = ax.bar(all_names, all_gini, color=all_colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Gini Coefficient (Sparsity)', fontsize=12)
    ax.set_title('Sparsity Comparison: Brain vs MoE Models', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 0.6)
    
    # Add value labels on bars
    for bar, val in zip(bars, all_gini):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add brain baseline reference line
    ax.axhline(y=0.458, color='#E74C3C', linestyle='--', linewidth=1.5, alpha=0.5)
    
    # Add annotation showing brain is more sparse
    brain_ratio = 0.458 / np.mean(gini_values)
    ax.annotate(f'Brain is {brain_ratio:.1f}× more sparse\nthan MoE average', 
                xy=(0.95, 0.95), xycoords='axes fraction',
                ha='right', va='top', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#FEF9E7', edgecolor='#F39C12'))
    
    fig4.tight_layout()
    fig4.savefig(output_dir / 'fig_multi_moe_brain_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig4)
    logger.info("Saved: fig_multi_moe_brain_comparison.png/svg")
    
    # =========================================================================
    # Figure 5: Radar Chart
    # =========================================================================
    fig5 = plt.figure(figsize=(10, 8))
    ax = fig5.add_subplot(111, projection='polar')
    
    metrics = ['Gini\n(Sparsity)', 'Entropy\nSparsity', 'Concentration\nRatio', 
               'Mean\nSelectivity', 'Specialization\nRate']
    n_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]
    
    for result in results_list:
        model = result['model']
        values = [
            result['sparsity_metrics']['gini'],
            result['sparsity_metrics']['entropy_sparsity'],
            result['sparsity_metrics']['concentration_ratio'],
            result['specialization_summary']['mean_selectivity'],
            result['specialization_summary']['specialization_rate'],
        ]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model['short_name'], color=model['color'])
        ax.fill(angles, values, alpha=0.25, color=model['color'])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 0.4)  # Adjusted based on actual max value (~0.33)
    ax.tick_params(axis='x', pad=15)  # Move labels outward to avoid overlap
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.set_title('MoE Model Comparison: Multi-Metric Radar', fontsize=14, fontweight='bold', pad=20)
    
    fig5.tight_layout()
    fig5.savefig(output_dir / 'fig_multi_moe_radar.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig5)
    logger.info("Saved: fig_multi_moe_radar.png/svg")
    
    logger.info(f"All visualizations saved to: {output_dir}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_multi_model_analysis():
    """Run complete multi-model MoE analysis with real models and datasets."""
    logger.info("="*70)
    logger.info("  MULTI-MODEL MOE ANALYSIS (REAL MODELS + DATASETS)")
    logger.info("  Comparing: Switch-8, Qwen-MoE, DeepSeek-MoE")
    logger.info("="*70)
    
    # Check GPU
    if torch.cuda.is_available():
        logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        logger.warning("No GPU available. Analysis may be slow.")
    
    # Load samples from HuggingFace datasets
    logger.info("\nLoading samples from HuggingFace datasets...")
    samples = load_samples_from_huggingface(n_samples_per_category=100)
    
    # Analyze each model
    results = []
    for model_config in MOE_MODELS:
        # Real inference only. Any failure aborts the whole MoE analysis —
        # we never fall back to simulated routing distributions, because those
        # would be presented alongside real brain data and corrupt the
        # biological-vs-artificial comparison.
        result = analyze_single_model_real(model_config, samples)
        results.append(result)
        
        # Clear GPU memory between models
        gc.collect()
        torch.cuda.empty_cache()
    
    # Cross-model comparison
    comparison = compare_models(results)
    
    logger.info("\n" + "="*70)
    logger.info("  CROSS-MODEL COMPARISON SUMMARY")
    logger.info("="*70)
    logger.info(f"  Most sparse model: {comparison['summary']['most_sparse_model']}")
    logger.info(f"  Gini range: {comparison['summary']['gini_range'][0]:.3f} - {comparison['summary']['gini_range'][1]:.3f}")
    logger.info(f"  Brain comparison: {comparison['brain_comparison']['interpretation']}")
    
    # Create visualizations
    output_dir = config.FIGURES_DIR / "multi_moe"
    create_multi_model_visualizations(results, comparison, output_dir)
    
    # Save results
    logger.info("\nSaving results...")
    
    # Save comparison summary
    comparison_path = config.MOE_ANALYSIS_DIR / "multi_moe_comparison.json"
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2, default=str)
    logger.info(f"Saved: {comparison_path}")
    
    # Save per-model results
    for result in results:
        model_name = result['model']['short_name'].replace('-', '_').lower()
        result_copy = {k: v for k, v in result.items()}
        result_path = config.MOE_ANALYSIS_DIR / f"moe_analysis_{model_name}.json"
        with open(result_path, 'w') as f:
            json.dump(result_copy, f, indent=2, default=str)
        logger.info(f"Saved: {result_path}")
    
    # Create summary DataFrame
    summary_data = []
    for result in results:
        model = result['model']
        metrics = result['sparsity_metrics']
        spec = result['specialization_summary']
        
        summary_data.append({
            'model': model['short_name'],
            'n_experts': model['n_experts'],
            'top_k': model['top_k'],
            'size': model['size'],
            'gini': metrics['gini'],
            'entropy_sparsity': metrics['entropy_sparsity'],
            'effective_experts': metrics['effective_experts'],
            'concentration_ratio': metrics['concentration_ratio'],
            'mean_selectivity': spec['mean_selectivity'],
            'specialization_rate': spec['specialization_rate'],
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = config.MOE_ANALYSIS_DIR / "multi_moe_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Saved: {summary_path}")
    
    logger.info("\n" + "="*70)
    logger.info("  MULTI-MODEL MOE ANALYSIS COMPLETE")
    logger.info("="*70)
    
    return results, comparison


if __name__ == "__main__":
    results, comparison = run_multi_model_analysis()
    print("\n✅ Multi-model MoE analysis completed successfully!")
