#!/usr/bin/env bash
# Analysis only; one process on one node, uses cuda:0 even on an 8-GPU allocation.
# Reuses checkpoint/sample/statistics defaults; adds bounded raw feature snapshots.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export ANALYSIS_NAME="${ANALYSIS_NAME:-down-token-heatmaps-$(date +%Y%m%d-%H%M%S)}"
export HEATMAP_SEQUENCES="${HEATMAP_SEQUENCES:-2}"
export HEATMAP_TOKENS="${HEATMAP_TOKENS:-64}"
export HEATMAP_CHANNELS="${HEATMAP_CHANNELS:-128}"
exec bash "${SCRIPT_DIR}/run_down_checkpoint_analysis.sh"
