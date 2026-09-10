#!/usr/bin/env bash
# Probes on --no-eager ablation checkpoints (rev only; fwd is identical
# without eager -- both variants collapse to k_eff=k). F3 across all 5
# ablation seeds + F4 at L=1 on the partial-passing seed (s0, k4=0.503).
#
# Question: does the ablated mixer form any of the F3 layer-localisation
# signature that eager-mediated passing seeds show? The partial-passing
# s0 (k4=0.50) is the interesting case -- if the residual-based pathway
# uses the same mid-scan mechanism, F4 should show buildup; if it uses
# a different pathway (e.g. residual stream between blocks), F3/F4 will
# not match the eager-on pattern.
set -euo pipefail
cd "$(dirname "$0")/.."

K1=4
K2=4

# F3 on all 5 rev ablation seeds
for seed in 0 1 2 3 4; do
  tag="ablation_no_eager__rev_joint_K1max${K1}_K2_4_s${seed}"
  ckpt="outputs/${tag}.ckpt"
  [ -f "${ckpt}" ] || { echo "[skip-f3] ${ckpt} missing"; continue; }
  out="outputs/ablation__${tag}__f3.json"
  [ -f "${out}" ] && { echo "[skip-f3] ${out} exists"; continue; }
  echo "[f3] ${tag}"
  python probe_f3.py --ckpt "${ckpt}" --K1 "${K1}" --K2 "${K2}" \
    --tag "ablation__${tag}__f3"
done

# F4 at L=1 on the partial-passing seed (s0). Baseline + KO.
# (KO on a no-eager model should be a no-op since eager was already 0
#  during training; run it as a sanity check.)
for L in 0 1 2; do
  seed=0
  tag="ablation_no_eager__rev_joint_K1max${K1}_K2_4_s${seed}"
  ckpt="outputs/${tag}.ckpt"
  out="outputs/ablation__${tag}__f4_L${L}.json"
  [ -f "${out}" ] && { echo "[skip-f4] ${out} exists"; continue; }
  echo "[f4] ${tag} L${L} base"
  python probe_f4.py --ckpt "${ckpt}" --K1 "${K1}" --K2 "${K2}" \
    --layer "${L}" --tag "ablation__${tag}__f4_L${L}"
done

echo "[ablation-probe] all done"
