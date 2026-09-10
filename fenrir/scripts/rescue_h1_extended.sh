#!/usr/bin/env bash
# rescue_h1_extended.sh -- test H1 (extended-training rescue) on failing seeds.
#
# Two arms per seed:
#   floor    : LR pinned at 3e-5 (cosine floor) for 4000 more steps
#   restart  : fresh cosine warm-restart (200 warmup, base 3e-4 -> 3e-5)
#
# Failing seeds chosen to span the plateau spectrum:
#   rev/joint  s2  (loss plateau at ~0.88, textbook "stuck")
#   rev/joint  s5  (loss drifting 1.26 -> 0.83 over last 800 steps, "still learning-ish")
#   rev/joint  s8  (loss drifting 1.20 -> 0.78, similar to s5)
#   rev/joint  s9  (loss plateau ~0.90, similar to s2)
#   rev/mixed  s2  (loss plateau ~1.78 -- mixed-mode failure)
#   fwd/joint  s0  (loss ~0.90 with k1=0.999 -- learned marginals, not composition)
#
# Timing target: rev/joint ~10 min per arm on a 5090. 6 seeds x 2 arms = 12 runs.
# Run sequential; total wallclock ~2 hours. Fine to run in background.

set -euo pipefail
cd "$(dirname "$0")/.."

VENV="${VENV:-.venv}"
PY="${VENV}/bin/python"; [ -x "$PY" ] || PY=python
[[ -x "$PY" ]] || { echo "no venv python at $PY"; exit 1; }

EXTRA_STEPS=4000

SEEDS=(
  "table1b__rev_joint_K1max4_K2_4_s2"
  "table1b__rev_joint_K1max4_K2_4_s5"
  "table1b__rev_joint_K1max4_K2_4_s8"
  "table1b__rev_joint_K1max4_K2_4_s9"
  "table1__rev_K1max4_K2_4_s2"
  "table1b__fwd_joint_K1max4_K2_4_s0"
)

for tag in "${SEEDS[@]}"; do
  CKPT="outputs/${tag}.ckpt"
  if [[ ! -f "$CKPT" ]]; then
    echo "SKIP: $CKPT missing"; continue
  fi
  for arm in floor restart; do
    OUT="outputs/rescue_h1__${tag}__${arm}.json"
    if [[ -f "$OUT" ]]; then
      echo "SKIP: $OUT already exists"; continue
    fi
    echo "=== $tag / arm=$arm ==="
    "$PY" resume_extend.py --ckpt "$CKPT" --arm "$arm" --extra-steps "$EXTRA_STEPS"
  done
done

echo "=== rescue_h1_extended.sh complete ==="
