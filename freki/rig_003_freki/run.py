"""rig_003_freki — FREKI (Fixed Read via Explicit K-chain Iteration), n=5 seeds."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common.harness import RunCfg, train_one
from model import FrekiConfig, build_model


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
    ap.add_argument("--beta-bias-init", type=float, default=-0.5)
    ap.add_argument("--K-chain", type=int, default=2)
    ap.add_argument("--tie-kv-proj", action="store_true",
                    help="Share weights between k_proj and v_proj (alignment fix).")
    ap.add_argument("--compile", action="store_true",
                    help="Wrap model in torch.compile(dynamic=True).")
    ap.add_argument("--compile-mode", default="reduce-overhead",
                    choices=["default", "reduce-overhead", "max-autotune"],
                    help="torch.compile mode.")
    ap.add_argument("--constant-lr", action="store_true",
                    help="Skip cosine decay — lr_floor set to lr.")
    ap.add_argument("--no-autocast", action="store_true",
                    help="Force fp32 (disable bf16 autocast).")
    ap.add_argument("--pad-seq-to", type=int, default=0,
                    help="Left-pad every sequence to this length (fixed-shape training).")
    ap.add_argument("--hops", type=int, default=2, choices=[2, 3],
                    help="Task hop-count: 2-hop (default) or 3-hop chain.")
    ap.add_argument("--K3", type=int, default=4, help="3-hop task's third-bank size.")
    ap.add_argument("--eval-K1-max", type=int, default=0,
                    help="Eval at K1=1..eval_K1_max (extrapolation). 0 = same as training K1.")
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--tag-prefix", default="rig003_freki")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = RunCfg(
        K1=args.K1, K2=args.K2, steps=args.steps, batch=args.batch,
        mode="mixed", eval_every=args.eval_every, device=args.device,
    )
    if args.constant_lr:
        cfg.lr_floor = cfg.lr    # skip cosine decay
    if args.no_autocast:
        cfg.autocast = False
    if args.pad_seq_to > 0:
        cfg.pad_seq_to = args.pad_seq_to
    cfg.hops = args.hops
    cfg.K3 = args.K3
    cfg.eval_K1_max = args.eval_K1_max
    freki_cfg = FrekiConfig(
        d_model=args.d_model, d_key=args.d_key, n_layers=args.n_layers,
        beta_bias_init=args.beta_bias_init, K_chain=args.K_chain,
        tie_kv_proj=args.tie_kv_proj,
    )
    out_dir = HERE / "outputs"

    import torch as _torch
    def make_builder(freki_cfg):
        def build(vocab_size):
            m = build_model(vocab_size, freki_cfg)
            if args.compile:
                m = _torch.compile(m, dynamic=True, mode=args.compile_mode)
            return m
        return build

    per_seed = []
    for seed in args.seeds:
        payload = train_one(
            build_model=make_builder(freki_cfg),
            rig_name="rig003_freki", seed=seed, cfg=cfg,
            out_dir=out_dir, tag=f"{args.tag_prefix}_s{seed}",
        )
        per_seed.append(payload)

    summary = {
        "rig": "rig003_freki",
        "seeds": args.seeds,
        "freki_cfg": freki_cfg.__dict__,
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
