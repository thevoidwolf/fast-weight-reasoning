#!/usr/bin/env bash
# F4 at L=0 only on the "late composer" passing seeds (s3, s4). s0 already
# probed at L=0 in the earlier late-composers sweep. L=2 dropped because the
# initial L=2 probe on s0 never converged in reasonable time (sklearn LR
# max_iter on failing-signal features) and the paper story is complete
# with L=0 evidence.
set -euo pipefail
cd "$(dirname "$0")/.."

K1=4
K2=4

for seed in 3 4; do
  tag="table1b__rev_joint_K1max${K1}_K2_4_s${seed}"
  ckpt="outputs/${tag}.ckpt"
  [ -f "${ckpt}" ] || { echo "[skip] ${ckpt} missing"; continue; }

  out_base="outputs/table2_late__${tag}__f4_L0.json"
  out_ko="outputs/table2_late__${tag}__f4_L0_KO.json"
  if [ ! -f "${out_base}" ]; then
    echo "[f4-late-L0] ${tag} base"
    python probe_f4.py --ckpt "${ckpt}" --K1 "${K1}" --K2 "${K2}" \
      --layer 0 --tag "table2_late__${tag}__f4_L0"
  fi
  if [ ! -f "${out_ko}" ]; then
    echo "[f4-late-L0] ${tag} KO"
    python probe_f4.py --ckpt "${ckpt}" --K1 "${K1}" --K2 "${K2}" \
      --layer 0 --eager-knockout --tag "table2_late__${tag}__f4_L0_KO"
  fi
done

echo "[table2-late-L0] all done"
