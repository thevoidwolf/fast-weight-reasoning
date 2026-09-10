#!/usr/bin/env bash
# h4_validate_fresh.sh -- independent validation of the H4 curriculum-dispatch rule.
#
# Rule (from n=10 training-run analysis):
#   Mixed viable iff k1 >= 0.40 at step 1600
#   Joint viable iff k2 >= 0.28 at step 1200
#
# This script trains seeds 10-19 (fresh, not in the original sample) under
# BOTH curricula to full budget. The rule can then be scored against
# ground-truth outcomes without re-running.
#
# Compute: rev/mixed ~2.5 min/seed at 4000 steps, rev/joint ~5.5 min/seed
# at 5000 steps. 10 seeds x 2 curricula = 20 runs = ~80 min sequential.
# Runs use --tag h4_fresh__* so they don't collide with existing files.

set -euo pipefail
cd "$(dirname "$0")/.."

VENV="${VENV:-.venv}"
PY="${VENV}/bin/python"; [ -x "$PY" ] || PY=python
[[ -x "$PY" ]] || { echo "no venv python at $PY"; exit 1; }

for seed in 10 11 12 13 14 15 16 17 18 19; do
  MIX_OUT="outputs/h4_fresh__mixed_s${seed}.json"
  if [[ -f "$MIX_OUT" ]]; then
    echo "SKIP: $MIX_OUT already exists"
  else
    echo "=== fresh mixed s${seed} (4000 steps) ==="
    "$PY" train.py --variant rev --K1 4 --K2 4 --seed "$seed" --steps 4000 \
        --mode mixed --tag "h4_fresh__mixed_s${seed}"
  fi

  JOI_OUT="outputs/h4_fresh__joint_s${seed}.json"
  if [[ -f "$JOI_OUT" ]]; then
    echo "SKIP: $JOI_OUT already exists"
  else
    echo "=== fresh joint s${seed} (5000 steps) ==="
    "$PY" train.py --variant rev --K1 4 --K2 4 --seed "$seed" --steps 5000 \
        --mode joint --tag "h4_fresh__joint_s${seed}"
  fi
done

echo "=== h4_validate_fresh.sh complete ==="
