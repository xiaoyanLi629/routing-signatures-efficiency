#!/bin/bash
# =============================================================================
# Auto-rerun orchestrator
# =============================================================================
# Waits for HCP download to finish ("Done!" in hcp_download.log), then runs:
#   stages 1-5  brain analysis (s01, s02, s03, s03b, s04) with expanded sample
#   stage 6      SKIPPED - s05b MoE results are data-invariant, copied from OLD_RUN
#   stage 7      cross-domain brain-vs-MoE comparison (paper contribution 3)
#   stage 8      global BH-FDR across H1/H3/H4
#   stage 9      visualization
#   manuscript number update pass
#
# Intended to be launched detached with nohup.
# Logs progress to logs/auto_rerun.log
# =============================================================================

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

DOWNLOAD_LOG="$PROJECT_DIR/logs/hcp_download.log"
RERUN_LOG="$PROJECT_DIR/logs/auto_rerun.log"
OLD_RUN="$PROJECT_DIR/results/run_20251222_225959"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [AUTO_RERUN] $*" | tee -a "$RERUN_LOG"
}

log "=== Auto-rerun orchestrator started (PID $$) ==="
log "Watching: $DOWNLOAD_LOG"

# --- Wait for download to complete ----------------------------------------
log "Polling every 60s for download completion..."
while true; do
    # Line pattern from download_hcp.py final print: "Done! Downloaded=..."
    # The Python logger emits a leading \n, so the line starts with "Done!"
    # at column 0 rather than after the timestamp prefix.
    if grep -qE "^Done! Downloaded=" "$DOWNLOAD_LOG" 2>/dev/null; then
        log "Download marker found in log."
        break
    fi
    # Fallback: if no python download process exists, check whether log has
    # been idle for >10 min — abort if so
    if ! pgrep -f "python tools/download_hcp.py" > /dev/null; then
        log "Download process not running."
        # Last log line age (seconds since modification)
        last_mod=$(stat -c %Y "$DOWNLOAD_LOG" 2>/dev/null || echo 0)
        now=$(date +%s)
        age=$((now - last_mod))
        if [ "$age" -gt 600 ]; then
            log "Log idle >10 min and no process — aborting."
            exit 1
        fi
        log "  (log was modified ${age}s ago; will re-check)"
    fi
    sleep 60
done

log "=== Download complete — starting brain analysis pipeline ==="

# Fresh pipeline timestamp
unset PROJECT2_RUN_TIMESTAMP
NEW_TS=$(date +%Y%m%d_%H%M%S)
export PROJECT2_RUN_TIMESTAMP=$NEW_TS
NEW_RUN="$PROJECT_DIR/results/run_$NEW_TS"
log "New run directory: $NEW_RUN"

# --- Stages 1-5: brain analysis (s01, s02, s03, s03b, s04) -----------------
for stage in 1 2 3 4 5; do
    log "Running stage $stage..."
    if python3 run_full_pipeline.py --stage $stage 2>&1 | tee -a "$RERUN_LOG"; then
        log "Stage $stage OK"
    else
        log "Stage $stage FAILED — aborting pipeline"
        exit 1
    fi
done

# --- Copy MoE results from old run (data-invariant) -----------------------
log "Copying s05b MoE results from $OLD_RUN to $NEW_RUN..."
mkdir -p "$NEW_RUN/moe_analysis"
cp -r "$OLD_RUN/moe_analysis/"* "$NEW_RUN/moe_analysis/" 2>&1 | tee -a "$RERUN_LOG"

# --- Stage 7: cross-domain (stage 6 / s05b skipped - results copied above) --
log "Running stage 7 (cross-domain brain-vs-MoE)..."
python3 run_full_pipeline.py --stage 7 2>&1 | tee -a "$RERUN_LOG"

# --- Stage 8: global BH-FDR across H1/H3/H4 -------------------------------
log "Running stage 8 (global FDR correction)..."
python3 run_full_pipeline.py --stage 8 2>&1 | tee -a "$RERUN_LOG"

# --- Stage 9: advanced visualization --------------------------------------
log "Running stage 9 (visualization)..."
python3 run_full_pipeline.py --stage 9 2>&1 | tee -a "$RERUN_LOG"

# --- Manuscript number update ---------------------------------------------
log "Running manuscript number update..."
python3 tools/update_manuscript_numbers.py \
    --run-dir "$NEW_RUN" \
    --tex "$PROJECT_DIR/papers/conference_BIBM2026/BIBM2026_paper.tex" \
    2>&1 | tee -a "$RERUN_LOG"

# --- Recompile tex --------------------------------------------------------
log "Recompiling LaTeX..."
cd "$PROJECT_DIR/papers/conference_BIBM2026"
pdflatex -interaction=nonstopmode BIBM2026_paper.tex > /dev/null 2>&1 || true
bibtex BIBM2026_paper > /dev/null 2>&1 || true
pdflatex -interaction=nonstopmode BIBM2026_paper.tex > /dev/null 2>&1 || true
pdflatex -interaction=nonstopmode BIBM2026_paper.tex > /dev/null 2>&1 || true
PAGE_COUNT=$(pdfinfo BIBM2026_paper.pdf | grep Pages | awk '{print $2}')
log "PDF page count: $PAGE_COUNT"

log "=== Auto-rerun complete ==="
