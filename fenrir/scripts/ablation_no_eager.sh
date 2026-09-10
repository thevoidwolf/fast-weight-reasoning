#!/usr/bin/env bash
# Ablation: no eager term (k_eff = k, no address perturbation).
#
# This is the TRAINING-TIME causal control for the inference-time F4-KO
# result. F4 shows that zeroing the eager term at inference collapses the
# answer-signal buildup on a trained checkpoint. This experiment asks the
# stronger question: if the eager term is removed at training time too,
# does the mechanism form at all?
#
# Runs 5 seeds x 2 variants under joint mode (5000 steps), the curriculum
# where the un-ablated rev variant reaches ~0.96-0.98. If the ablated rev
# variant fails at chance, the eager term is causally necessary. If it
# passes, the address perturbation is decorative and the paper's mechanism
# claim needs to be weakened.
#
# Note: --no-eager forces --no-chunked (the chunked kernel is derived from
# the eager-term recurrence). Wallclock ~500s per rev seed (up from ~340s
# with chunked), ~1400s per fwd seed.
#
# Writes:
#   outputs/ablation_no_eager__{variant}_joint_K1max4_K2_4_s{seed}.{ckpt,json}
set -euo pipefail
cd "$(dirname "$0")/.."

VARIANTS=(rev fwd)
SEEDS=(0 1 2 3 4)
K1=4
K2=4

for variant in "${VARIANTS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    tag="ablation_no_eager__${variant}_joint_K1max${K1}_K2_4_s${seed}"
    if [ -f "outputs/${tag}.ckpt" ]; then
      echo "[skip] ${tag} (checkpoint exists)"
      continue
    fi
    echo "[ablation] variant=${variant} seed=${seed} mode=joint no-eager"
    python train.py --variant "${variant}" --K1 "${K1}" --K2 "${K2}" \
      --seed "${seed}" --steps 5000 --mode joint \
      --no-eager \
      --tag "${tag}"
  done
done

echo "[ablation] all done"
