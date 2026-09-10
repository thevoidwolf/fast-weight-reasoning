#!/usr/bin/env bash
# dkey16.sh -- final experiment (3a): reduce d_key from 32 to 16.
#
# Hypothesis: the partial-composition attractor requires enough per-position
# state capacity to fit K=1/K=2 marginals independently. Halving d_key should
# reduce or eliminate that capacity, forcing the model to either compose or
# fail cleanly (removing the middle attractor).
#
# Seeds 0-9 x both curricula. 20 runs. Compare universal-robust rate to the
# d_key=32 baselines in outputs/table1__* and outputs/table1b__*.

set -euo pipefail
cd "$(dirname "$0")/.."
VENV="${VENV:-.venv}"
PY="${VENV}/bin/python"; [ -x "$PY" ] || PY=python

DKEY=16

for seed in 0 1 2 3 4 5 6 7 8 9; do
  for mode in mixed joint; do
    if [[ "$mode" == "mixed" ]]; then STEPS=4000; else STEPS=5000; fi
    TAG="dkey16__${mode}_s${seed}"
    OUT="outputs/${TAG}.json"
    [[ -f "$OUT" ]] && { echo "SKIP: $OUT already exists"; continue; }
    echo "=== d_key=${DKEY} ${mode} s${seed} (${STEPS} steps) ==="
    "$PY" train.py --variant rev --K1 4 --K2 4 --seed "$seed" --steps "$STEPS" \
        --mode "$mode" --d-key "$DKEY" --tag "$TAG"
  done
done

echo "=== dkey16.sh complete ==="
