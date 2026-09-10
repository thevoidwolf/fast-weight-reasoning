"""D1 — Zero-shot depth-transfer probe (Fable's #1 recommended diagnostic).

Plan:
  1. Train FREKI K_chain=2 seed 0 on the 2-hop task, 12K steps.
     (Known 100% passer for FREKI-2 at n=10.)
  2. Extract trained state_dict via the harness `probe` callback.
  3. Build a fresh FREKI K_chain=3 model of matching shape.
  4. Copy matching weights; leave chain_norms[2] at its init (ones).
  5. Evaluate the K_chain=3 model, ZERO-SHOT (no additional training),
     on the 3-hop task ladder (k1..k4).
  6. Report.

Interpretation:
  - k above chance (>0.35) => solution partially exists in FREKI-2's
    weights; depth wall is from-scratch optimization, not architectural.
  - marginal (0.28-0.35) => weak transfer; try D1b (warm-start finetune).
  - at chance (<0.28) => solution doesn't transfer; either the learned g
    is value-token-specialized (bank-2 pair-second are v-tokens in 2-hop
    but entity-tokens in 3-hop banks 1-2), or the primitive can't
    extend in-place.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import torch

from common.harness import RunCfg, train_one, eval_ladder
from common.tasks import TaskCfg
from model import FrekiConfig, build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-steps", type=int, default=12000)
    ap.add_argument("--eval-K1-max", type=int, default=4)
    ap.add_argument("--eval-batches", type=int, default=4)
    ap.add_argument("--eval-batch-size", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="d1_transfer")
    args = ap.parse_args()

    out_dir = HERE / "outputs"
    out_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print(f"D1 depth-transfer probe, seed {args.seed}")
    print("=" * 60)

    # Train FREKI-2 seed 0, extract state_dict via probe callback.
    freki2_cfg = FrekiConfig(K_chain=2)
    cfg_train = RunCfg(
        K1=4, K2=4, steps=args.train_steps, batch=64,
        mode="mixed", eval_every=500, device=args.device, hops=2,
    )

    def build2(vocab_size):
        return build_model(vocab_size, freki2_cfg)

    # Capture trained state via closure — probe return must be JSON-safe.
    capture = {}
    def capture_state(model):
        capture["state_dict"] = {k: v.detach().cpu().clone()
                                 for k, v in model.state_dict().items()}
        capture["vocab_size"] = int(model.embed.num_embeddings)
        return {"captured_params": len(capture["state_dict"])}

    print(f"\n[phase 1] Training FREKI K_chain=2 on 2-hop, "
          f"{args.train_steps} steps...")
    payload = train_one(
        build_model=build2, rig_name="d1_freki2_base",
        seed=args.seed, cfg=cfg_train, out_dir=out_dir,
        tag=f"{args.tag}_freki2_base_s{args.seed}",
        probe=capture_state,
    )

    freki2_final_k4 = payload["final_acc"].get("k4")
    print(f"[phase 1] done. Final 2-hop k4={freki2_final_k4:.3f}")
    if freki2_final_k4 < 0.85:
        print("Base FREKI-2 didn't pass. D1 not interpretable. Aborting.")
        return

    trained_sd = capture["state_dict"]
    vocab_size = capture["vocab_size"]

    # Rebuild FREKI-2 with captured weights, do a clean 2-hop eval to confirm.
    freki2 = build2(vocab_size).to(args.device)
    freki2.load_state_dict({k: v.to(args.device) for k, v in trained_sd.items()})
    freki2._hrs_autocast = cfg_train.autocast

    task_cfg = TaskCfg(seed=args.seed)
    with torch.no_grad():
        clean2 = eval_ladder(
            freki2, task_cfg, K1_max=4, K2=4,
            batch=args.eval_batch_size, n_batches=args.eval_batches,
            device=args.device, eval_seed=args.seed + 10_000,
            hops=2, K3=4,
        )
    print(f"[sanity] FREKI-2 (loaded) on 2-hop: {clean2}")

    # Build FREKI K_chain=3, copy compatible weights.
    print("\n[phase 2] Building FREKI K_chain=3 and copying weights...")
    freki3_cfg = FrekiConfig(K_chain=3)
    freki3 = build_model(vocab_size, freki3_cfg).to(args.device)
    freki3._hrs_autocast = cfg_train.autocast

    src_sd = {k: v.to(args.device) for k, v in trained_sd.items()}
    dst_sd = freki3.state_dict()

    copied, skipped = [], []
    new_sd = {}
    for name, param in dst_sd.items():
        if name in src_sd and src_sd[name].shape == param.shape:
            new_sd[name] = src_sd[name].clone()
            copied.append(name)
        else:
            new_sd[name] = param.clone()   # keep FREKI-3's fresh init
            skipped.append(name)
    freki3.load_state_dict(new_sd)

    print(f"[phase 2] copied {len(copied)} params from FREKI-2 → FREKI-3")
    print(f"[phase 2] kept fresh init for {len(skipped)} params (new to FREKI-3):")
    for n in skipped:
        print(f"           {n}  shape={tuple(dst_sd[n].shape)}")

    # Sanity: FREKI-3 with 2-hop weights on 2-hop task — extra chain step
    # applied to already-completed 2-hop retrieval. Expect degradation.
    with torch.no_grad():
        freki3_on_2hop = eval_ladder(
            freki3, task_cfg, K1_max=4, K2=4,
            batch=args.eval_batch_size, n_batches=args.eval_batches,
            device=args.device, eval_seed=args.seed + 10_000,
            hops=2, K3=4,
        )
    print(f"\n[sanity] FREKI-3 (weights from FREKI-2) on 2-hop: {freki3_on_2hop}")

    # Zero-shot 3-hop eval.
    print("\n[phase 3] ZERO-SHOT 3-hop eval (no training)...")
    with torch.no_grad():
        transfer_acc = eval_ladder(
            freki3, task_cfg, K1_max=args.eval_K1_max, K2=4,
            batch=args.eval_batch_size, n_batches=args.eval_batches,
            device=args.device, eval_seed=args.seed + 10_000,
            hops=3, K3=4,
        )

    print("\n" + "=" * 60)
    print("D1 RESULT")
    print("=" * 60)
    print(f"FREKI-2 on 2-hop:                      k4={clean2['k4']:.3f}")
    print(f"FREKI-3 (transferred weights) on 2-hop: k4={freki3_on_2hop['k4']:.3f}")
    print(f"ZERO-SHOT 3-hop transfer: {transfer_acc}")
    print(f"Chance floor: 0.250 (1/K3)")
    k_vals = [transfer_acc[f'k{i}'] for i in range(1, args.eval_K1_max + 1)]
    max_k = max(k_vals)
    print(f"Max k in ladder: {max_k:.3f}")
    if max_k > 0.35:
        print("→ ABOVE CHANCE. 3-hop solution partially exists in FREKI-2's "
              "learned weights. Depth wall is FROM-SCRATCH OPTIMIZATION, "
              "not architectural. Fable recommendation: skip cleanup bridge, "
              "aux-only fix (#2) may suffice.")
    elif max_k > 0.28:
        print("→ MARGINAL. Weak transfer. Try D1b (warm-start finetune) to see "
              "if the basin is reachable by continuation.")
    else:
        print("→ AT CHANCE. Solution does not transfer zero-shot. Try D1b "
              "next; if that also fails, move to #1 (cleanup bridge + in-path "
              "aux) — the full architectural fix is warranted.")

    result = {
        "seed": args.seed,
        "train_steps": args.train_steps,
        "freki2_on_2hop": clean2,
        "freki3_transferred_on_2hop": freki3_on_2hop,
        "freki3_zero_shot_on_3hop": transfer_acc,
        "params_copied_count": len(copied),
        "params_skipped": skipped,
    }
    result_path = out_dir / f"{args.tag}_s{args.seed}.json"
    result_path.write_text(json.dumps(result, indent=2))
    print(f"\nResults written to {result_path}")


if __name__ == "__main__":
    main()
