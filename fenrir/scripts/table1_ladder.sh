#!/usr/bin/env bash
# Table 1: K1 ladder head-to-head.
#
# 2 variants (fwd, rev) x 4 K1 settings (1..4) x 5 seeds = 40 checkpoints
# under matched training protocol (mixed K1 cycling, 4000 steps).
#
# Writes:
#   outputs/table1__<variant>_K1max<K1>_K2_4_s<seed>.{ckpt,json}
#
# Run from the code/ directory:
#   bash scripts/table1_ladder.sh
#
# Total wallclock estimate on RTX 5090: about 5 hours.
set -euo pipefail
cd "$(dirname "$0")/.."

VARIANTS=(rev fwd)
K1_LIST=(4)          # each K1 trains through 1..K1 in mixed mode
SEEDS=(0 1 2 3 4)
STEPS=4000

for variant in "${VARIANTS[@]}"; do
  for K1 in "${K1_LIST[@]}"; do
    for seed in "${SEEDS[@]}"; do
      tag="table1__${variant}_K1max${K1}_K2_4_s${seed}"
      if [ -f "outputs/${tag}.ckpt" ]; then
        echo "[skip] ${tag} (checkpoint exists)"
        continue
      fi
      echo "[table1] variant=${variant} K1=${K1} seed=${seed}"
      python train.py \
        --variant "${variant}" \
        --K1 "${K1}" --K2 4 \
        --seed "${seed}" --steps "${STEPS}" \
        --tag "${tag}"
    done
  done
done

echo "[table1] all done"
