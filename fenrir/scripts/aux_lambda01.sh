#!/usr/bin/env bash
# aux_lambda01.sh -- test intervention (C): per-layer aux loss with lambda=0.1.
#
# Focused seed x curriculum design spanning failure classes:
#   - s2 mixed, s5 mixed    : universal-fail (H1 confirmed unrescuable)
#   - s3 mixed, s4 mixed, s6 mixed : joint-only-rescuable (mixed fails baseline)
#   - s8 joint, s9 joint    : mixed-only-rescuable (joint fails baseline)
#   - s0 mixed, s0 joint    : universal-pass CONTROL (should still pass)
#   - s1 joint              : universal-pass CONTROL
#
# Total: 10 runs. mixed ~2.5 min, joint ~5.5 min each. Sequential ~40 min.
# Comparison: same seeds without aux (already trained under both curricula) are
#   outputs/table1__rev_K1max4_K2_4_s{X}.json (mixed baseline)
#   outputs/table1b__rev_joint_K1max4_K2_4_s{X}.json (joint baseline)

set -euo pipefail
cd "$(dirname "$0")/.."

VENV="${VENV:-.venv}"
PY="${VENV}/bin/python"; [ -x "$PY" ] || PY=python
[[ -x "$PY" ]] || { echo "no venv python at $PY"; exit 1; }

LAMBDA=0.1

run() {
  local mode=$1 seed=$2 steps=$3
  local tag="aux01__${mode}_s${seed}"
  local out="outputs/${tag}.json"
  if [[ -f "$out" ]]; then
    echo "SKIP: $out already exists"; return
  fi
  echo "=== aux(lambda=$LAMBDA) ${mode} s${seed} (${steps} steps) ==="
  "$PY" train.py --variant rev --K1 4 --K2 4 --seed "$seed" --steps "$steps" \
      --mode "$mode" --aux-lambda "$LAMBDA" --tag "$tag"
}

# Failure-class seeds
run mixed 2 4000     # universal-fail
run mixed 5 4000     # universal-fail
run mixed 3 4000     # joint-only rescuable
run mixed 4 4000     # joint-only rescuable
run mixed 6 4000     # joint-only rescuable
run joint 8 5000     # mixed-only rescuable
run joint 9 5000     # mixed-only rescuable

# Controls (should still pass)
run mixed 0 4000     # universal-pass control
run joint 0 5000     # universal-pass control
run joint 1 5000     # universal-pass control

echo "=== aux_lambda01.sh complete ==="
