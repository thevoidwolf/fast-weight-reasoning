#!/usr/bin/env bash
# Reproduce every number in the FENRIR paper.
#
# Runs in order:
#   1. pytest tests/         (audit-as-code, must pass before any training)
#   2. table1_ladder.sh      (40 training runs)
#   3. table2_mechanism.sh   (probes per checkpoint)
#   4. table3_depth.sh       (zero-shot depth gen eval)
#
# All outputs land in outputs/. Every paper number reads from a JSON there.
#
# Run from the code/ directory:
#   bash scripts/reproduce_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "[reproduce_all] step 1/4: pytest"
pytest tests/ -q

echo "[reproduce_all] step 2/4: Table 1 (ladder)"
bash scripts/table1_ladder.sh

echo "[reproduce_all] step 3/4: Table 2 (mechanism probes)"
bash scripts/table2_mechanism.sh

echo "[reproduce_all] step 4/4: Table 3 (depth generalisation)"
bash scripts/table3_depth.sh

echo "[reproduce_all] complete. All numbers in outputs/."
