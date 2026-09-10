"""rig_007_aux_only — Arm 1: aux-only middle-ground design.

Fable-approved design (per follow-up):
  - Model: FREKI K_chain=3 (unmodified). No cleanup, no forward-path change.
  - Per-hop bridge_head[h] = Linear(d_key → d_model), decoded via tied lm_head.
  - Applied at LAYER 0 ONLY. Rationale: at layer 0 the residual stream carries
    only the input embeddings, so any bridge information in y_h must have come
    through M — the head cannot be a bypass.
  - Aux at ATOK position ONLY. Rationale: at ATOK the conv1d window covers
    (last-value, QTOK, q_ent, ATOK) — no bridge token locally visible.
    Predicting bridge from y_h@ATOK is structurally impossible without
    M-mediated retrieval.
  - Supervise BOTH intermediate hops (bridge1 and bridge2). Every alignment
    now gets first-order gradient — breaks the ε² product from the depth-3
    plateau.
  - Schedule: λ(t) = λ_init · exp(-t / τ), λ_init=0.3, τ=4000 (slower than
    my previous τ=1500 which killed the aux before consolidation).

Both aux losses share `model.lm_head` weight → anchors the classifier's
output space to the vocab embedding, but leaves the mixer's own linear
code (in which y_1 already carries bridge signal cleanly per D5) untouched.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.harness import RunCfg, cosine_lr, eval_ladder, left_pad
from common.tasks import (
    TaskCfg,
    sample_long_3hop_injective_truncated,
)
from model import Rig7Config, build_model


def train_one_arm1(
    build_model_fn, seed: int, cfg: RunCfg, hs_cfg: Rig7Config,
    out_dir: Path, tag: str,
    aux_lambda_init: float, aux_decay_tau: float,
    aux_layer: int,
):
    """Arm 1 training loop with aux-only middle-ground supervision.

    Bridge heads live outside `build_model_fn`'s stack; they're
    instantiated here as separate parameters that share `model.lm_head`
    via reuse-at-forward.
    """
    log = lambda s: print(s, flush=True)

    torch.manual_seed(seed)
    task_cfg = TaskCfg(seed=seed)
    model = build_model_fn(task_cfg.vocab_size).to(cfg.device)
    model._hrs_autocast = cfg.autocast

    d_key = hs_cfg.d_key
    d_model = hs_cfg.d_model
    # Two bridge heads: one for hop-1 (y_1 → bridge1), one for hop-2 (y_2 → bridge2).
    bridge_heads = nn.ModuleList([
        nn.Linear(d_key, d_model, bias=False).to(cfg.device)
        for _ in range(hs_cfg.K_chain - 1)   # h=1..K_chain-1
    ])

    all_params = list(model.parameters()) + list(bridge_heads.parameters())
    n_params = sum(p.numel() for p in all_params if p.requires_grad)

    opt = torch.optim.AdamW(all_params, lr=cfg.lr,
                            weight_decay=cfg.weight_decay, betas=cfg.betas)

    train_tasks = [(K1_r, cfg.K2) for K1_r in range(1, cfg.K1 + 1)]
    train_gens = {tk: torch.Generator(device="cpu").manual_seed(seed + 1000 + i)
                  for i, tk in enumerate(train_tasks)}

    # Hooks on layer `aux_layer`'s chain_norms[h] to capture y_h post-normalization.
    captured = {}   # h → [B, L, d_key]
    hooks = []
    for h, cn in enumerate(model.blocks[aux_layer].mixer.chain_norms):
        def make_hook(h=h):
            def _hook(_mod, _inp, out):
                captured[h] = out
            return _hook
        hooks.append(cn.register_forward_hook(make_hook()))

    log(f"[{tag}] params={n_params:,} tasks={train_tasks} steps={cfg.steps} "
        f"aux(layer={aux_layer}, λ_init={aux_lambda_init}, τ={aux_decay_tau}, "
        f"style=middle-ground-atok-only)")
    curve = []
    start = time.time()

    try:
        for step in range(cfg.steps):
            lr = cosine_lr(step, cfg.warmup, cfg.steps, cfg.lr, cfg.lr_floor)
            for pg in opt.param_groups:
                pg["lr"] = lr

            tk = train_tasks[step % len(train_tasks)]
            k1r = tk[0]
            k2_use = max(cfg.K2, k1r)
            k3_use = max(cfg.K3, k2_use)
            seq, tgt, bridge1, bridge2 = sample_long_3hop_injective_truncated(
                k1r, k2_use, k3_use, task_cfg, cfg.batch, train_gens[tk],
                device=cfg.device)
            if cfg.pad_seq_to > 0:
                seq = left_pad(seq, cfg.pad_seq_to)

            aux_lambda = aux_lambda_init * math.exp(-step / aux_decay_tau)

            captured.clear()
            with torch.autocast(device_type=cfg.device, dtype=torch.bfloat16,
                                enabled=(cfg.device == "cuda" and cfg.autocast)):
                logits = model(seq)
                main_loss = F.cross_entropy(logits[:, -1], tgt)

                if aux_lambda >= 1e-6:
                    # y_h for h in 0..K_chain-1 captured. Bridge targets are
                    # for hops 0 (=bridge1) and 1 (=bridge2). Wait — in FREKI's
                    # code, chain_norms[h] fires AFTER the h-th M application,
                    # so captured[0] = y_1 (after 1st Mᵀ), captured[1] = y_2,
                    # captured[2] = y_3.
                    # We want y_1 → bridge1 and y_2 → bridge2.
                    aux_targets = [bridge1, bridge2]      # in that order
                    aux_losses = []
                    for h in range(hs_cfg.K_chain - 1):
                        y_h_atok = captured[h][:, -1, :]           # [B, d_key]
                        # bridge_head[h]: d_key → d_model, then decode via tied lm_head
                        proj = bridge_heads[h](y_h_atok)            # [B, d_model]
                        # lm_head is Linear(d_model, vocab), so:
                        aux_logits = model.lm_head(proj)            # [B, vocab]
                        aux_losses.append(F.cross_entropy(aux_logits, aux_targets[h]))
                    aux_mean = sum(aux_losses) / len(aux_losses)
                    loss = main_loss + aux_lambda * aux_mean
                    aux_val = float(aux_mean.item())
                else:
                    loss = main_loss
                    aux_val = float("nan")

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, cfg.grad_clip)
            opt.step()

            if (step + 1) % cfg.eval_every == 0 or (step + 1) == cfg.steps:
                eval_K1_max = cfg.eval_K1_max if cfg.eval_K1_max > 0 else cfg.K1
                accs = eval_ladder(model, task_cfg, eval_K1_max, cfg.K2,
                                   cfg.eval_batch_size, cfg.eval_batches,
                                   cfg.device, seed + 10_000,
                                   pad_seq_to=cfg.pad_seq_to,
                                   hops=cfg.hops, K3=cfg.K3)
                curve.append({
                    "step": step + 1,
                    "loss": float(main_loss.item()),
                    "aux_loss": aux_val,
                    "aux_lambda": aux_lambda,
                    "acc": accs,
                })
                acc_str = " ".join(f"{k}={v:.3f}" for k, v in accs.items())
                log(f"  step {step + 1:>4d} loss {main_loss.item():.3f} "
                    f"aux {aux_val:.3f} λ {aux_lambda:.4f} lr {lr:.1e} {acc_str}")
    finally:
        for h in hooks:
            h.remove()

    wallclock = time.time() - start
    final_acc = curve[-1]["acc"]
    payload = {
        "tag": tag, "rig": "rig007_aux_only", "seed": seed,
        "params": n_params, "cfg": asdict(cfg),
        "aux_style": "middle_ground_atok_only",
        "aux_layer": aux_layer,
        "aux_lambda_init": aux_lambda_init, "aux_decay_tau": aux_decay_tau,
        "vocab_size": task_cfg.vocab_size,
        "train_tasks": [list(t) for t in train_tasks],
        "final_acc": final_acc,
        "wallclock_s": round(wallclock, 3),
        "curve": curve,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{tag}.json").write_text(json.dumps(payload, indent=2))
    log(f"[{tag}] done in {wallclock:.1f}s  final k{cfg.K1}={final_acc.get(f'k{cfg.K1}'):.3f}")
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--K1", type=int, default=4)
    ap.add_argument("--K2", type=int, default=4)
    ap.add_argument("--K3", type=int, default=4)
    ap.add_argument("--K-chain", type=int, default=3)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--d-key", type=int, default=32)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--beta-bias-init", type=float, default=-0.5)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--aux-lambda-init", type=float, default=0.3)
    ap.add_argument("--aux-decay-tau", type=float, default=4000.0)
    ap.add_argument("--aux-layer", type=int, default=0)
    ap.add_argument("--tag-prefix", default="rig007_arm1")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = RunCfg(
        K1=args.K1, K2=args.K2, steps=args.steps, batch=args.batch,
        mode="mixed", eval_every=args.eval_every, device=args.device,
        hops=3, K3=args.K3,
    )
    hs_cfg = Rig7Config(
        d_model=args.d_model, d_key=args.d_key, n_layers=args.n_layers,
        beta_bias_init=args.beta_bias_init, K_chain=args.K_chain,
    )
    out_dir = HERE / "outputs"

    def make_builder(hs_cfg):
        return lambda vocab_size: build_model(vocab_size, hs_cfg)

    per_seed = []
    for seed in args.seeds:
        payload = train_one_arm1(
            build_model_fn=make_builder(hs_cfg), seed=seed,
            cfg=cfg, hs_cfg=hs_cfg, out_dir=out_dir,
            tag=f"{args.tag_prefix}_s{seed}",
            aux_lambda_init=args.aux_lambda_init,
            aux_decay_tau=args.aux_decay_tau,
            aux_layer=args.aux_layer,
        )
        per_seed.append(payload)

    summary = {
        "rig": "rig007_aux_only",
        "seeds": args.seeds,
        "hs_cfg": hs_cfg.__dict__,
        "run_cfg": cfg.__dict__,
        "aux": {
            "style": "middle_ground_atok_only",
            "layer": args.aux_layer,
            "lambda_init": args.aux_lambda_init,
            "decay_tau": args.aux_decay_tau,
        },
        "final_k4_per_seed": [p["final_acc"].get(f"k{args.K1}") for p in per_seed],
        "final_full_per_seed": [p["final_acc"] for p in per_seed],
        "wallclock_s_per_seed": [p["wallclock_s"] for p in per_seed],
    }
    (out_dir / f"{args.tag_prefix}_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
