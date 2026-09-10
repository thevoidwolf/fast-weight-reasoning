#!/usr/bin/env bash
# Inference-time eager-knockout eval on every trained checkpoint that
# has use_eager=True (Table 1 mixed + Table 1b joint). Complements
# ablation_no_eager.sh (training-time knockout). This one uses the
# same trained weights but flips use_eager at inference.
#
# For each checkpoint, writes outputs/eval_eager_ko__<tag>.json holding
# both "acc_eager_on" (baseline) and "acc_eager_off" (knockout) rows.
set -euo pipefail
cd "$(dirname "$0")/.."

VARIANTS=(rev fwd)
SEEDS=(0 1 2 3 4 5 6 7 8 9)
K1=4
K2=4

eval_one() {
  local tag="$1"
  local ckpt="outputs/${tag}.ckpt"
  if [ ! -f "${ckpt}" ]; then
    echo "[skip] ${ckpt} missing"
    return 0
  fi
  echo "[eval_ko] ${tag}"
  python eval_eager_knockout.py --ckpt "${ckpt}" --K1 "${K1}" --K2 "${K2}" \
    --tag "eval_ko__${tag}"
}

for variant in "${VARIANTS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    eval_one "table1__${variant}_K1max${K1}_K2_4_s${seed}"
    eval_one "table1b__${variant}_joint_K1max${K1}_K2_4_s${seed}"
  done
done

echo "[eval_ko] all done"
