#!/usr/bin/env bash
# Table 2: full mechanism sweep across all Table 1 + Table 1b checkpoints.
#
# For every checkpoint from table1_ladder.sh (mixed mode) and table1b_joint.sh
# (joint mode), run:
#   - F4 mid-scan probe at L=0 and L=1, baseline + eager-knockout arms
#   - F3 logit-lens
#   - F5 Q/K alignment at L=0, L=1, L=2
#
# Depends on outputs/ from table1_ladder.sh and table1b_joint.sh.
#
# Run from the code/ directory:
#   bash scripts/table2_mechanism.sh
set -euo pipefail
cd "$(dirname "$0")/.."

VARIANTS=(rev fwd)
# Seeds 0-9 to cover n=10 extension when checkpoints exist (skip via -f guard
# in probe_checkpoint below when they don't); base n=5 always present.
SEEDS=(0 1 2 3 4 5 6 7 8 9)
K1=4
K2=4

probe_checkpoint() {
  local tag="$1"
  local ckpt="outputs/${tag}.ckpt"
  if [ ! -f "${ckpt}" ]; then
    echo "[skip] ${ckpt} missing"
    return 0
  fi
  echo "[table2] ${tag}"

  # F4 at each layer, baseline + KO
  for L in 0 1; do
    python probe_f4.py --ckpt "${ckpt}" --K1 "${K1}" --K2 "${K2}" \
      --layer "${L}" --tag "table2__${tag}__f4_L${L}"
    python probe_f4.py --ckpt "${ckpt}" --K1 "${K1}" --K2 "${K2}" \
      --layer "${L}" --eager-knockout --tag "table2__${tag}__f4_L${L}_KO"
  done

  # F3 logit lens
  python probe_f3.py --ckpt "${ckpt}" --K1 "${K1}" --K2 "${K2}" \
    --tag "table2__${tag}__f3"

  # F5 Q/K alignment (all three layers)
  for L in 0 1 2; do
    python probe_f5.py --ckpt "${ckpt}" --K1 "${K1}" --K2 "${K2}" \
      --layer "${L}" --tag "table2__${tag}__f5_L${L}"
  done
}

# --- Table 1 (mixed) checkpoints ---
for variant in "${VARIANTS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    probe_checkpoint "table1__${variant}_K1max${K1}_K2_${K2}_s${seed}"
  done
done

# --- Table 1b (joint) checkpoints ---
for variant in "${VARIANTS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    probe_checkpoint "table1b__${variant}_joint_K1max${K1}_K2_4_s${seed}"
  done
done

echo "[table2] all done"
