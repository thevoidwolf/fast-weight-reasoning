#!/usr/bin/env bash
# Extends Table 1 (mixed) and Table 1b (joint) to n=10 seeds by adding
# rev seeds 5-9 to both curricula. The fwd variant fails consistently
# under both curricula in the first n=5; not extending fwd for now.
#
# Runs sequentially. Chunked kernel makes rev fast (~340s/seed on RTX 5090).
# Safe to run concurrent with fwd Table 1b training (GPU has huge headroom).
#
# Writes:
#   outputs/table1__rev_K1max4_K2_4_s{5..9}.{ckpt,json}
#   outputs/table1b__rev_joint_K1max4_K2_4_s{5..9}.{ckpt,json}
set -euo pipefail
cd "$(dirname "$0")/.."

EXTRA_SEEDS=(5 6 7 8 9)
K1=4
K2=4

# Table 1b (joint) rev seeds 5..9 first (highest-value data for the bimodal claim)
for seed in "${EXTRA_SEEDS[@]}"; do
  tag="table1b__rev_joint_K1max${K1}_K2_4_s${seed}"
  if [ -f "outputs/${tag}.ckpt" ]; then
    echo "[skip] ${tag} (checkpoint exists)"
    continue
  fi
  echo "[extra] variant=rev K1=${K1} seed=${seed} mode=joint"
  python train.py --variant rev --K1 "${K1}" --K2 "${K2}" \
    --seed "${seed}" --steps 5000 --mode joint --tag "${tag}"
done

# Table 1 (mixed) rev seeds 5..9 second, complements the Table 1 bimodal picture
for seed in "${EXTRA_SEEDS[@]}"; do
  tag="table1__rev_K1max${K1}_K2_4_s${seed}"
  if [ -f "outputs/${tag}.ckpt" ]; then
    echo "[skip] ${tag} (checkpoint exists)"
    continue
  fi
  echo "[extra] variant=rev K1=${K1} seed=${seed} mode=mixed"
  python train.py --variant rev --K1 "${K1}" --K2 "${K2}" \
    --seed "${seed}" --steps 4000 --mode mixed --tag "${tag}"
done

echo "[extra] all done"
