#!/usr/bin/env python3
"""
=============================================================================
Analysis Pipeline — IEEE BIBM 2026
"Routing-Level Signatures of Cognitive Efficiency:
 A Multi-Task fMRI Null with a Pairwise-Correlation Caveat"
=============================================================================

Three contributions, and the pipeline stages that produce each:

  (1) FDR-controlled joint test of four candidate routing signatures in 100 HCP
      subjects across 7 task-fMRI paradigms  -> stages 1-5, 8
  (2) The pairwise-correlation inference pitfall (16 orders of magnitude)
      -> stage 4 (s03b, independence-safe H3 re-test)
  (3) Cross-domain comparison against three MoE foundation models
      (Table II: brain ~2.5x more aggregate-sparse)  -> stages 6-7

Contribution (3) requires the MoE model weights in models/huggingface_cache
(a symlink to ../../moe_models, shared read-only with the TST journal paper).
Run `python download_models.py` if they are missing.

Usage:
    python run_full_pipeline.py              # Run all stages
    python run_full_pipeline.py --stage 1 2  # Run specific stages
    python run_full_pipeline.py --skip-moe   # Brain-side only (stages 6-7 skipped)
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))

from configs import config

# =============================================================================
# PIPELINE STAGES
# =============================================================================

STAGES = {
    1: {
        'name': 'Multi-Task Extraction',
        'script': 's01_multi_task_extraction.py',
        'description': 'Extract activation patterns from 7 HCP tasks',
        'requires_moe': False,
    },
    2: {
        'name': 'Sparsity Analysis (H1)',
        'script': 's02_sparsity_analysis.py',
        'description': 'Analyze activation sparsity across tasks',
        'requires_moe': False,
    },
    3: {
        'name': 'Routing Patterns (H2/H3)',
        'script': 's03_routing_patterns.py',
        'description': 'Analyze task-network routing patterns',
        'requires_moe': False,
    },
    4: {
        'name': 'H3 Robust Re-test',
        'script': 's03b_h3_robust.py',
        'description': 'Independence-safe subject-level LOO test of inter-subject routing consistency',
        'requires_moe': False,
    },
    5: {
        'name': 'Functional Modularity (H4)',
        'script': 's04_functional_modularity.py',
        'description': 'Louvain community detection and modularity',
        'requires_moe': False,
    },
    6: {
        'name': 'MoE Analysis (Multi-Model)',
        'script': 's05b_multi_moe_analysis.py',
        'description': 'Expert routing in Switch/Qwen/DeepSeek (Table II)',
        'requires_moe': True,
    },
    7: {
        'name': 'Cross-Domain Comparison',
        'script': 's06_cross_domain_comparison.py',
        'description': 'Compare brain and MoE routing sparsity (contribution 3)',
        'requires_moe': True,
    },
    8: {
        'name': 'Hypothesis FDR Correction',
        'script': 's06b_hypothesis_fdr.py',
        'description': 'Global BH-FDR across H1/H3/H4 primary hypothesis tests',
        'requires_moe': False,
    },
    9: {
        'name': 'Advanced Visualization',
        'script': 's07_advanced_visualization.py',
        'description': 'Generate publication-quality figures',
        'requires_moe': False,
    },
}

# =============================================================================
# PIPELINE RUNNER
# =============================================================================

def run_stage(stage_num, skip_moe=False):
    """Run a single pipeline stage"""
    stage = STAGES[stage_num]
    
    # Check if we should skip MoE stages
    if skip_moe and stage['requires_moe']:
        print(f"  ⏭️  Skipping Stage {stage_num} (MoE required)")
        return True
    
    script_path = PROJECT_DIR / "scripts" / stage['script']
    
    if not script_path.exists():
        print(f"  ⚠️  Script not found: {script_path}")
        return False
    
    print(f"\n{'='*60}")
    print(f"Stage {stage_num}: {stage['name']}")
    print(f"Description: {stage['description']}")
    print(f"{'='*60}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            cwd=str(PROJECT_DIR)
        )
        print(f"\n  ✅ Stage {stage_num} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n  ❌ Stage {stage_num} failed with error code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n  ❌ Stage {stage_num} failed: {str(e)}")
        return False


def run_pipeline(stages=None, skip_moe=False):
    """Run the full pipeline or specified stages"""
    
    import os
    # Respect an externally-set PROJECT2_RUN_TIMESTAMP (e.g. from an orchestrator
    # that is driving the pipeline stage-by-stage). Only generate a fresh
    # timestamp when none is supplied — otherwise all stages land in their
    # own run_* directory and break inter-stage data flow.
    config._RUN_TIMESTAMP = None
    config._DIRECTORIES_INITIALIZED = False

    results_dir = config.initialize_run_directories()
    run_timestamp = config.get_run_timestamp()

    # Always export for any child process spawned below.
    os.environ['PROJECT2_RUN_TIMESTAMP'] = run_timestamp
    
    # Setup logging
    logger = config.setup_logging('pipeline')
    
    start_time = datetime.now()
    print("\n" + "="*70)
    print("  ROUTING-LEVEL SIGNATURES OF COGNITIVE EFFICIENCY")
    print("  IEEE BIBM 2026 submission")
    print("="*70)
    print(f"\nStart time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results directory: {results_dir}")
    
    # Determine which stages to run
    if stages is None:
        stages = list(STAGES.keys())
    
    print(f"Stages to run: {stages}")
    if skip_moe:
        print("Note: MoE-dependent stages will be skipped")
    
    # Run each stage
    results = {}
    for stage_num in stages:
        if stage_num not in STAGES:
            print(f"⚠️  Unknown stage: {stage_num}")
            continue
        
        success = run_stage(stage_num, skip_moe=skip_moe)
        results[stage_num] = success
        
        if not success:
            logger.error(f"Stage {stage_num} failed")
            # Continue with other stages even if one fails
    
    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "="*70)
    print("  PIPELINE SUMMARY")
    print("="*70)
    print(f"\nEnd time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {duration}")
    print(f"\nResults:")
    
    for stage_num, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        print(f"  Stage {stage_num} ({STAGES[stage_num]['name']}): {status}")
    
    # Check if all stages passed
    all_passed = all(results.values())
    if all_passed:
        print(f"\n🎉 All stages completed successfully!")
        logger.info("Pipeline completed successfully")
    else:
        failed = [s for s, r in results.items() if not r]
        print(f"\n⚠️  Some stages failed: {failed}")
        logger.warning(f"Pipeline completed with failures: {failed}")
    
    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description='Run the Sparse Routing Analysis Pipeline'
    )
    parser.add_argument(
        '--stage', '-s',
        type=int,
        nargs='+',
        help='Specific stages to run (e.g., --stage 1 2 3)'
    )
    parser.add_argument(
        '--skip-moe',
        action='store_true',
        help='Skip stages that require MoE model analysis'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available stages'
    )
    
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable Pipeline Stages:")
        print("-" * 60)
        for num, stage in STAGES.items():
            moe_tag = " [MoE]" if stage['requires_moe'] else ""
            print(f"  Stage {num}: {stage['name']}{moe_tag}")
            print(f"            {stage['description']}")
        return
    
    success = run_pipeline(stages=args.stage, skip_moe=args.skip_moe)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

