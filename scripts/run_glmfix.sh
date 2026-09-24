#!/usr/bin/env bash
# Brain-side pipeline on the corrected first-level GLM (s01b), followed by the
# reviewer analyses (s08) and camera-ready figures (s09).
#
#   HCP_DATA_ROOT=/path/to/hcp  PYTHON=python  scripts/run_glmfix.sh
#
# Outputs: results/run_${PROJECT2_RUN_TIMESTAMP:-20260924_glmfix}/
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PROJECT2_RUN_TIMESTAMP=${PROJECT2_RUN_TIMESTAMP:-20260924_glmfix}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PYTHON:-python}
RUN=results/run_${PROJECT2_RUN_TIMESTAMP}
mkdir -p "$RUN/neuroscience"

$PY -u scripts/s01b_glm_fixed.py

# Behavioural efficiency score from HCP WM_Stats.csv (independent of fMRI).
if [ ! -f "$RUN/neuroscience/behavioral_validation.csv" ]; then
  $PY -u tools/build_behavioral_validation.py \
      --data-root "${HCP_DATA_ROOT:-data}" --output "$RUN/neuroscience/behavioral_validation.csv"
fi

for s in s02_sparsity_analysis s03_routing_patterns s03b_h3_robust s04_functional_modularity s06b_hypothesis_fdr; do
  echo "=== $s"; $PY -u scripts/$s.py
done
CR_RUN=run_${PROJECT2_RUN_TIMESTAMP} $PY -u scripts/s08_camera_ready_analyses.py
CR_RUN=run_${PROJECT2_RUN_TIMESTAMP} $PY -u scripts/s09_camera_ready_figures.py
echo ALL_DONE
