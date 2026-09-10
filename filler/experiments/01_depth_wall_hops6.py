"""Experiment 1 - the 6-hop depth wall (the headline).

At 6 hops the baseline recipe fails: seeds land at chance, or "solve" the target
with a query-conditioned readout, or diverge (residual norm explodes). The only
change in the filler arm is that training splices a ramped random count of filler
tokens between the last fact and the query. That single change restores a passing
seed rate.

    python filler/experiments/01_depth_wall_hops6.py --smoke      # tiny CPU check
    python filler/experiments/01_depth_wall_hops6.py --full       # baseline n=3 + filler n=5 (GPU)
    python filler/experiments/01_depth_wall_hops6.py --recipe filler --seed 0
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from filler.sweep import run_arm

K_HOPS = 6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", choices=("baseline", "filler"), default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        # k_hops=3 keeps the sequence short; tiny model + few steps for CPU.
        for recipe in ("baseline", "filler"):
            run_arm(recipe=recipe, k_hops=3, seed=0, steps=20, batch=4,
                    d_model=64, eval_n=8, warmup_steps=5, ramp_steps=5,
                    eval_every=20)
        return
    if args.full:
        for seed in (0, 1, 2):
            run_arm(recipe="baseline", k_hops=K_HOPS, seed=seed, steps=args.steps)
        for seed in (0, 1, 2, 3, 4):
            run_arm(recipe="filler", k_hops=K_HOPS, seed=seed, steps=args.steps)
        return
    if args.recipe is None:
        ap.error("pass --recipe, --smoke, or --full")
    run_arm(recipe=args.recipe, k_hops=K_HOPS, seed=args.seed,
            steps=args.steps, batch=args.batch)


if __name__ == "__main__":
    main()
