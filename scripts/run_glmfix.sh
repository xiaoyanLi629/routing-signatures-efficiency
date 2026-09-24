#!/usr/bin/env bash
# Camera-ready: re-run the brain-side pipeline on the corrected GLM (s01b) into
# results/run_20260924_glmfix, then the reviewer analyses and figures.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PROJECT2_RUN_TIMESTAMP=20260924_glmfix OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/root/miniconda3/bin/python
RUN=results/run_${PROJECT2_RUN_TIMESTAMP}
mkdir -p $RUN/neuroscience
$PY -u scripts/s01b_glm_fixed.py
cp results/run_20260419_182908/neuroscience/behavioral_validation.csv $RUN/neuroscience/
for s in s02_sparsity_analysis s03_routing_patterns s03b_h3_robust s04_functional_modularity s06b_hypothesis_fdr; do
  echo "=== $s"; $PY -u scripts/$s.py
done
CR_RUN=run_${PROJECT2_RUN_TIMESTAMP} $PY -u scripts/s08_camera_ready_analyses.py
CR_RUN=run_${PROJECT2_RUN_TIMESTAMP} $PY -u scripts/s09_camera_ready_figures.py
CR_RUN=run_20260419_182908 $PY -u scripts/s09_camera_ready_figures.py
echo ALL_DONE
