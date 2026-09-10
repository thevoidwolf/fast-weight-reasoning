#!/usr/bin/env bash
# Table 1b: same K1 ladder head-to-head as Table 1, but with joint-mode
# training instead of mixed-mode.
#
# Joint mode: at every step, sample one batch each from K1=1 and K1=K1_max
# and average the two cross-entropy losses. This is the training curriculum
# rig 051 called "R5"; PART2 line 1672 reported R5 (fwd variant, seed s1)
# reaching 0.985 at 5000 steps, and later reseeds at 15000 steps reaching
# 0.998 for s1 and s2.
#
# Same 2 variants x 5 seeds = 10 checkpoints; 5000 steps each.
#
# Writes:
#   outputs/table1b__<variant>_joint_K1max<K1>_K2_4_s<seed>.{ckpt,json}
set -euo pipefail
cd "$(dirname "$0")/.."

VARIANTS=(rev fwd)
K1_LIST=(4)
SEEDS=(0 1 2 3 4)
STEPS=5000

for variant in "${VARIANTS[@]}"; do
  for K1 in "${K1_LIST[@]}"; do
    for seed in "${SEEDS[@]}"; do
      tag="table1b__${variant}_joint_K1max${K1}_K2_4_s${seed}"
      if [ -f "outputs/${tag}.ckpt" ]; then
        echo "[skip] ${tag} (checkpoint exists)"
        continue
      fi
      echo "[table1b] variant=${variant} K1=${K1} seed=${seed} mode=joint"
      python train.py \
        --variant "${variant}" \
        --K1 "${K1}" --K2 4 \
        --seed "${seed}" --steps "${STEPS}" \
        --mode joint \
        --tag "${tag}"
    done
  done
done

echo "[table1b] all done"
