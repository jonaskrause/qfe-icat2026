#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python}
SCRIPT=${SCRIPT:-qfe_icat_2026.py}
OUT_ROOT=${OUT_ROOT:-icat_2026_results}

mkdir -p "$OUT_ROOT" grid_logs

# ICAT comparison grid: 3 sanities/depths can be represented by qubits/layers/seeds.
# Adjust ranges below if you need a smaller test.
QUBITS=(3 4 5 6 7 8 9 10)
LAYERS=(2 3 4)
SEEDS=(0 1 2 3 4)
POPULATION=${POPULATION:-60}
GENERATIONS=${GENERATIONS:-120}
MODES=${MODES:-FIDELITY_ONLY,QFE_SCORE,FLOW_ONLY,GRADIENT_ONLY,QFE_UNITARY_FLOW,RANDOM}

for Q in "${QUBITS[@]}"; do
  for L in "${LAYERS[@]}"; do
    for S in "${SEEDS[@]}"; do
      RUN_DIR="$OUT_ROOT/q${Q}_l${L}_s${S}"
      LOG="grid_logs/q${Q}_l${L}_s${S}.log"
      echo "============================================================"
      echo "Running Q=${Q} L=${L} seed=${S} -> ${RUN_DIR}"
      echo "============================================================"
      "$PYTHON" "$SCRIPT" \
        --n-qubits "$Q" \
        --layers "$L" \
        --population "$POPULATION" \
        --generations "$GENERATIONS" \
        --seed "$S" \
        --modes "$MODES" \
        --fast-gradient \
        --max-gradient-parameters 24 \
        --save-every 20 \
        --output-dir "$RUN_DIR" 2>&1 | tee "$LOG"
    done
  done
done
