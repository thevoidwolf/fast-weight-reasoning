#!/usr/bin/env bash
# single_mode.sh -- test intervention (1a): drop K=1 from training entirely.
#
# Trains --mode single (K=K1_max only) on seeds 0-9 to give direct comparisons
# vs the existing joint baseline (table1b__rev_joint_K1max4_K2_4_s{0..9}).
#
# Hypothesis: partial-composition attractor exists because current curricula
# explicitly train K=1 marginals. Removing K=1 removes the attractor's foothold.
#
# 5000 steps matches joint step count for a step-for-step comparison. Note:
# single-mode uses ONE task per step (joint uses two per step averaged), so
# wallclock will be roughly half joint's per-step time.

set -euo pipefail
cd "$(dirname "$0")/.."

VENV="${VENV:-.venv}"
PY="${VENV}/bin/python"; [ -x "$PY" ] || PY=python
[[ -x "$PY" ]] || { echo "no venv python at $PY"; exit 1; }

STEPS=5000

for seed in 0 1 2 3 4 5 6 7 8 9; do
  TAG="single__K1max4_K2_4_s${seed}"
  OUT="outputs/${TAG}.json"
  if [[ -f "$OUT" ]]; then
    echo "SKIP: $OUT already exists"; continue
  fi
  echo "=== single-mode s${seed} (K=4 only, ${STEPS} steps) ==="
  "$PY" train.py --variant rev --K1 4 --K2 4 --seed "$seed" --steps "$STEPS" \
      --mode single --tag "$TAG"
done

echo "=== single_mode.sh complete ==="
