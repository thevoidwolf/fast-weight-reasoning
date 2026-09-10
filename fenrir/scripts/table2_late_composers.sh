#!/usr/bin/env bash
# F4 at L=0 and L=2 on the "late composer" passing seeds -- those that
# passed the K1 ladder but showed no F4 buildup at L=1 (Table 2 focused).
# Question: do they compose at L=0 (early) or L=2 (late)? The F3 decode
# for these seeds already localises at L=2, so we expect F4 buildup at L=2.
#
# 3 seeds x 2 layers x 2 (base + KO) = 12 F4 probes.
set -euo pipefail
cd "$(dirname "$0")/.."

K1=4
K2=4

# Seeds that passed but showed no L=1 F4 buildup in the focused table 2
LATE_SEEDS=(
  table1b__rev_joint_K1max4_K2_4_s0
  table1b__rev_joint_K1max4_K2_4_s3
  table1b__rev_joint_K1max4_K2_4_s4
)

for tag in "${LATE_SEEDS[@]}"; do
  ckpt="outputs/${tag}.ckpt"
  [ -f "${ckpt}" ] || { echo "[skip] ${ckpt} missing"; continue; }
  for L in 0 2; do
    out_base="outputs/table2_late__${tag}__f4_L${L}.json"
    out_ko="outputs/table2_late__${tag}__f4_L${L}_KO.json"
    if [ ! -f "${out_base}" ]; then
      echo "[f4-late] ${tag} L${L} base"
      python probe_f4.py --ckpt "${ckpt}" --K1 "${K1}" --K2 "${K2}" \
        --layer "${L}" --tag "table2_late__${tag}__f4_L${L}"
    fi
    if [ ! -f "${out_ko}" ]; then
      echo "[f4-late] ${tag} L${L} KO"
      python probe_f4.py --ckpt "${ckpt}" --K1 "${K1}" --K2 "${K2}" \
        --layer "${L}" --eager-knockout --tag "table2_late__${tag}__f4_L${L}_KO"
    fi
  done
done

echo "[table2-late] all done"
