"""rig_002_neumann — Neumann-Read primitive, n=3 seeds, K1=4 mixed."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common.harness import RunCfg, train_one
from model import NeumannConfig, build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--K1", type=int, default=4)
    ap.add_argument("--K2", type=int, default=4)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--d-key", type=int, default=32)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--beta-bias-init", type=float, default=-3.0)
    ap.add_argument("--K-read", type=int, default=3)
    ap.add_argument("--norm-between-iters", action="store_true")
    ap.add_argument("--alpha-init", type=float, nargs="+", default=None,
                    help="Per-order initial α; length K_read+1. Default (1,1,0,0,...).")
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--tag-prefix", default="rig002_neumann")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = RunCfg(
        K1=args.K1, K2=args.K2, steps=args.steps, batch=args.batch,
        mode="mixed", eval_every=args.eval_every, device=args.device,
    )
    alpha_init = tuple(args.alpha_init) if args.alpha_init else (1.0, 1.0, 0.0, 0.0)
    n_cfg = NeumannConfig(
        d_model=args.d_model, d_key=args.d_key, n_layers=args.n_layers,
        beta_bias_init=args.beta_bias_init, K_read=args.K_read,
        norm_between_iters=args.norm_between_iters,
        alpha_init=alpha_init,
    )
    out_dir = HERE / "outputs"

    def make_builder(n_cfg):
        return lambda vocab_size: build_model(vocab_size, n_cfg)

    def probe_alphas(model):
        """Dump trained α weights per block — tells us whether Neumann iterations
        were actually used or if training left α_2/α_3 near zero."""
        return {
            f"block_{i}_alpha": blk.mixer.alpha.detach().cpu().tolist()
            for i, blk in enumerate(model.blocks)
        }

    per_seed = []
    for seed in args.seeds:
        payload = train_one(
            build_model=make_builder(n_cfg),
            rig_name="rig002_neumann", seed=seed, cfg=cfg,
            out_dir=out_dir, tag=f"{args.tag_prefix}_s{seed}",
            probe=probe_alphas,
        )
        per_seed.append(payload)

    summary = {
        "rig": "rig002_neumann",
        "seeds": args.seeds,
        "neumann_cfg": n_cfg.__dict__,
        "run_cfg": cfg.__dict__,
        "final_k4_per_seed": [p["final_acc"].get(f"k{args.K1}") for p in per_seed],
        "final_full_per_seed": [p["final_acc"] for p in per_seed],
        "wallclock_s_per_seed": [p["wallclock_s"] for p in per_seed],
    }
    (out_dir / f"{args.tag_prefix}_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
