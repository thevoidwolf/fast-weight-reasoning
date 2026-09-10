#!/usr/bin/env bash
# Table 2 focused: F3 logit-lens across every checkpoint (cheap), plus
# F4 mid-scan at L=1 with baseline + eager-KO on a representative
# subset of checkpoints. Drops F5 (Q/K alignment) from the sweep --
# preliminary F5 data on rev s0 already appears in the paper, and F5
# does not carry a causal claim.
#
# Cost estimate:
#   - F3 per checkpoint : ~30 s (cheap, no per-timestep LR fitting)
#   - F4 per checkpoint : ~10 min (per-timestep LR fitting dominates)
#
# ~25 F3 + (~10 seeds x 2 F4) = ~15 min + ~200 min = ~3.5 hours total.
#
# Representative F4 seeds picked to cover the four cells of the KO
# summary table: rev passing, rev failing, fwd passing, fwd failing.
set -euo pipefail
cd "$(dirname "$0")/.."

K1=4
K2=4

# Every trained checkpoint gets F3 (fast, comprehensive)
run_f3() {
  local tag="$1"
  local ckpt="outputs/${tag}.ckpt"
  [ -f "${ckpt}" ] || { echo "[skip-f3] ${ckpt} missing"; return 0; }
  local out="outputs/table2__${tag}__f3.json"
  [ -f "${out}" ] && { echo "[skip-f3] ${out} exists"; return 0; }
  echo "[f3] ${tag}"
  python probe_f3.py --ckpt "${ckpt}" --K1 "${K1}" --K2 "${K2}" \
    --tag "table2__${tag}__f3"
}

# Selected checkpoints get F4 (expensive, per-timestep LR)
run_f4_pair() {
  local tag="$1"
  local ckpt="outputs/${tag}.ckpt"
  [ -f "${ckpt}" ] || { echo "[skip-f4] ${ckpt} missing"; return 0; }
  local out1="outputs/table2__${tag}__f4_L1.json"
  local out2="outputs/table2__${tag}__f4_L1_KO.json"
  if [ ! -f "${out1}" ]; then
    echo "[f4] ${tag} L1"
    python probe_f4.py --ckpt "${ckpt}" --K1 "${K1}" --K2 "${K2}" \
      --layer 1 --tag "table2__${tag}__f4_L1"
  fi
  if [ ! -f "${out2}" ]; then
    echo "[f4] ${tag} L1 KO"
    python probe_f4.py --ckpt "${ckpt}" --K1 "${K1}" --K2 "${K2}" \
      --layer 1 --eager-knockout --tag "table2__${tag}__f4_L1_KO"
  fi
}

# --- F3 across all seeds/variants/curricula ---
for variant in rev fwd; do
  for seed in 0 1 2 3 4 5 6 7 8 9; do
    run_f3 "table1__${variant}_K1max${K1}_K2_4_s${seed}"
    run_f3 "table1b__${variant}_joint_K1max${K1}_K2_4_s${seed}"
  done
done

# --- F4 on representative seeds ---
# rev PASSING (mixed): s0, s7  ;  rev PASSING (joint): s0, s3, s6, s7
# rev FAILING (joint):  s2, s5
# fwd PASSING (joint):  s3
# fwd FAILING (mixed):  s0  ; fwd FAILING (joint):  s0
for tag in \
    table1__rev_K1max4_K2_4_s0    \
    table1__rev_K1max4_K2_4_s7    \
    table1b__rev_joint_K1max4_K2_4_s0 \
    table1b__rev_joint_K1max4_K2_4_s3 \
    table1b__rev_joint_K1max4_K2_4_s6 \
    table1b__rev_joint_K1max4_K2_4_s2 \
    table1b__rev_joint_K1max4_K2_4_s5 \
    table1b__fwd_joint_K1max4_K2_4_s3 \
    table1__fwd_K1max4_K2_4_s0    \
    table1b__fwd_joint_K1max4_K2_4_s0
do
  run_f4_pair "${tag}"
done

echo "[table2-focused] all done"
