#!/usr/bin/env bash
# aux_anneal.sh -- final experiment (2): anneal aux_lambda over steps.
#
# Same 10 seed×curriculum combos as the constant-λ aux_lambda01 experiment,
# but with λ decayed exponentially:
#   eff_lambda(step) = 0.15 * exp(-step / 1500)
# giving λ ≈ 0.15 at step 0, 0.055 at step 1500, 0.02 at step 3000, 0.007 at
# step 5000. High early to escape chance-plateau; decayed by mid-training to
# release the model toward composition (not pinned into partial).

set -euo pipefail
cd "$(dirname "$0")/.."
VENV="${VENV:-.venv}"
PY="${VENV}/bin/python"; [ -x "$PY" ] || PY=python

LAMBDA0=0.15
TAU=1500

run() {
  local mode=$1 seed=$2 steps=$3
  local tag="aux_anneal__${mode}_s${seed}"
  local out="outputs/${tag}.json"
  [[ -f "$out" ]] && { echo "SKIP: $out already exists"; return; }
  echo "=== anneal-aux (λ0=$LAMBDA0 τ=$TAU) ${mode} s${seed} (${steps} steps) ==="
  "$PY" train.py --variant rev --K1 4 --K2 4 --seed "$seed" --steps "$steps" \
      --mode "$mode" --aux-lambda "$LAMBDA0" --aux-decay-tau "$TAU" --tag "$tag"
}

# Same panel as aux_lambda01
run mixed 2 4000     # universal-fail
run mixed 5 4000     # universal-fail
run mixed 3 4000     # joint-only rescuable
run mixed 4 4000     # joint-only rescuable
run mixed 6 4000     # joint-only rescuable
run joint 8 5000     # mixed-only rescuable
run joint 9 5000     # mixed-only rescuable
run mixed 0 4000     # universal-pass control
run joint 0 5000     # universal-pass control
run joint 1 5000     # universal-pass control

echo "=== aux_anneal.sh complete ==="
