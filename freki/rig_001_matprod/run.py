"""rig_001_matprod — n=3 seeds, MatProd primitive, mixed curriculum, K1=4."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common.harness import RunCfg, train_one
from model import MatProdConfig, build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--K1", type=int, default=4)
    ap.add_argument("--K2", type=int, default=4)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--d-key", type=int, default=32)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--beta-bias-init", type=float, default=-3.0)
    ap.add_argument("--additive-b", action="store_true",
                    help="Hybrid variant: add M += β·k·vᵀ alongside the multiplicative update.")
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--tag-prefix", default="rig001_matprod")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = RunCfg(
        K1=args.K1, K2=args.K2, steps=args.steps, batch=args.batch,
        mode="mixed", eval_every=args.eval_every, device=args.device,
    )
    mp_cfg = MatProdConfig(
        d_model=args.d_model, d_key=args.d_key,
        n_layers=args.n_layers, beta_bias_init=args.beta_bias_init,
        additive_b=args.additive_b,
    )
    out_dir = HERE / "outputs"

    def make_builder(mp_cfg):
        return lambda vocab_size: build_model(vocab_size, mp_cfg)

    per_seed = []
    for seed in args.seeds:
        payload = train_one(
            build_model=make_builder(mp_cfg),
            rig_name="rig001_matprod", seed=seed, cfg=cfg,
            out_dir=out_dir, tag=f"{args.tag_prefix}_s{seed}",
        )
        per_seed.append(payload)

    # Summary
    summary = {
        "rig": "rig001_matprod",
        "seeds": args.seeds,
        "mp_cfg": mp_cfg.__dict__,
        "run_cfg": cfg.__dict__,
        "final_k4_per_seed": [p["final_acc"].get(f"k{args.K1}") for p in per_seed],
        "wallclock_s_per_seed": [p["wallclock_s"] for p in per_seed],
    }
    (out_dir / f"{args.tag_prefix}_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
