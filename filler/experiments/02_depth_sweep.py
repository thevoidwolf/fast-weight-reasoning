"""Experiment 2 - the filler recipe across depths (K_hops = 3..6).

The same recipe, unchanged, applied at each depth. It restores target accuracy,
a state-persistent readout (low QTOK-flip), and numerical stability across the
whole range. This is the "one recipe, four depths" claim; experiment 1 is the
hardest single cell (K=6).

    python filler/experiments/02_depth_sweep.py --smoke
    python filler/experiments/02_depth_sweep.py --full          # filler, K=3..6, 3 seeds each (GPU)
    python filler/experiments/02_depth_sweep.py --k-hops 4 --seed 0
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from filler.sweep import run_arm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k-hops", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        run_arm(recipe="filler", k_hops=3, seed=0, steps=20, batch=4,
                d_model=64, eval_n=8, warmup_steps=5, ramp_steps=5, eval_every=20)
        return
    if args.full:
        for k in (3, 4, 5, 6):
            for seed in (0, 1, 2):
                run_arm(recipe="filler", k_hops=k, seed=seed, steps=args.steps)
        return
    if args.k_hops is None:
        ap.error("pass --k-hops, --smoke, or --full")
    run_arm(recipe="filler", k_hops=args.k_hops, seed=args.seed, steps=args.steps)


if __name__ == "__main__":
    main()
