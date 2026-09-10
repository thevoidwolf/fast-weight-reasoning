#!/usr/bin/env bash
# Table 3: zero-shot K1 depth generalisation.
#
# Each Table 1 checkpoint (trained on K1 in 1..4) is evaluated at
# K1 in {5, 6, 7, 8} without further training. Tests whether the
# mechanism extends beyond the trained depth.
#
# Depends on outputs/ from table1_ladder.sh.
#
# Run from the code/ directory:
#   bash scripts/table3_depth.sh
set -euo pipefail
cd "$(dirname "$0")/.."

VARIANTS=(rev fwd)
SEEDS=(0 1 2 3 4)
K1_TRAIN=4
K2=4

for variant in "${VARIANTS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    tag="table1__${variant}_K1max${K1_TRAIN}_K2_${K2}_s${seed}"
    ckpt="outputs/${tag}.ckpt"
    if [ ! -f "${ckpt}" ]; then
      echo "[skip] ${ckpt} missing"
      continue
    fi
    # K2 must be >= max K1 for the injective sampler; use K2=8 for depths 5..8
    echo "[table3] ${tag}"
    python eval.py --ckpt "${ckpt}" --K1 5 6 7 8 --K2 8 \
      --tag "table3__${tag}__depth"
  done
done

echo "[table3] all done"
