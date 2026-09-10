"""rig_000_baseline_fenrir — sanity: does the existing FENRIR-rev primitive
break chance-plateau at OUR small config (2 layers, d=128, 1500 steps)?

Establishes the floor before we can interpret MatProd / Neumann results.
Imports FenrirStack directly from the fenrir/ chapter so we are testing
the exact same primitive that was characterized at 3 layers d=256 4000 steps.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]                                # repo root (fast-weight-reasoning/)
sys.path.insert(0, str(REPO_ROOT / "fenrir"))   # for FenrirStack + chunked
sys.path.insert(0, str(HERE.parent))                            # for common/

from common.harness import RunCfg, train_one
from model import FenrirStack, MixerConfig                      # from the fenrir/ chapter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="rev", choices=["rev", "fwd"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--K1", type=int, default=4)
    ap.add_argument("--K2", type=int, default=4)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--d-key", type=int, default=32)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--tag-prefix", default="rig000_fenrir_rev")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = RunCfg(
        K1=args.K1, K2=args.K2, steps=args.steps, batch=args.batch,
        mode="mixed", eval_every=args.eval_every, device=args.device,
    )
    mixer_cfg = MixerConfig(d_model=args.d_model, d_key=args.d_key,
                            d_value=args.d_key)
    out_dir = HERE / "outputs"

    def make_builder(variant, n_layers):
        return lambda vocab_size: FenrirStack(
            vocab_size=vocab_size, cfg=mixer_cfg, variant=variant,
            n_layers=n_layers, chunked=False, use_eager=True,
        )

    per_seed = []
    for seed in args.seeds:
        payload = train_one(
            build_model=make_builder(args.variant, args.n_layers),
            rig_name=f"rig000_fenrir_{args.variant}", seed=seed, cfg=cfg,
            out_dir=out_dir, tag=f"{args.tag_prefix}_s{seed}",
        )
        per_seed.append(payload)

    summary = {
        "rig": f"rig000_fenrir_{args.variant}",
        "seeds": args.seeds,
        "mixer_cfg": mixer_cfg.__dict__,
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
