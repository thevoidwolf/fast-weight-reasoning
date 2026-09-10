"""rig_008_cleanup_aux — Arm 2: cleanup bridge + in-path aux (Fable's #1).

Model: rig_008/model.py CleanupStack with embedding-anchored softmax
snap between chain steps. Model exposes a settable `cleanup_T` that
this training loop anneals from `T_init` (soft) → `T_final` (sharp)
linearly over training.

Aux: reuse the model's own `_last_aux_logits` from the mixer forward
(logits produced by the cleanup bridge — same tensor whose softmax
feeds the next hop, so aux can't bypass). CE against bridge1, bridge2
at ATOK only, from LAYER 0 only (rationale as Arm 1).
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
import torch.nn.functional as F

from common.harness import RunCfg, cosine_lr, eval_ladder, left_pad
from common.tasks import (
    TaskCfg,
    sample_long_3hop_injective_truncated,
)
from model import CleanupConfig, build_model


def train_one_arm2(
    build_model_fn, seed: int, cfg: RunCfg, hs_cfg: CleanupConfig,
    out_dir: Path, tag: str,
    aux_lambda_init: float, aux_decay_tau: float,
    T_init: float, T_final: float, aux_layer: int,
):
    log = lambda s: print(s, flush=True)

    torch.manual_seed(seed)
    task_cfg = TaskCfg(seed=seed)
    model = build_model_fn(task_cfg.vocab_size).to(cfg.device)
    model._hrs_autocast = cfg.autocast
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay, betas=cfg.betas)

    train_tasks = [(K1_r, cfg.K2) for K1_r in range(1, cfg.K1 + 1)]
    train_gens = {tk: torch.Generator(device="cpu").manual_seed(seed + 1000 + i)
                  for i, tk in enumerate(train_tasks)}

    log(f"[{tag}] params={n_params:,} tasks={train_tasks} steps={cfg.steps} "
        f"cleanup(T:{T_init}→{T_final}) aux(layer={aux_layer}, "
        f"λ_init={aux_lambda_init}, τ={aux_decay_tau})")
    curve = []
    start = time.time()

    for step in range(cfg.steps):
        lr = cosine_lr(step, cfg.warmup, cfg.steps, cfg.lr, cfg.lr_floor)
        for pg in opt.param_groups:
            pg["lr"] = lr

        # Anneal T linearly.
        progress = step / max(1, cfg.steps - 1)
        T = T_init + (T_final - T_init) * progress
        model.set_cleanup_T(T)

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

        with torch.autocast(device_type=cfg.device, dtype=torch.bfloat16,
                            enabled=(cfg.device == "cuda" and cfg.autocast)):
            logits = model(seq)
            main_loss = F.cross_entropy(logits[:, -1], tgt)

            if aux_lambda >= 1e-6:
                # Layer `aux_layer`'s cleanup logits captured on the model side.
                aux_stack = model.blocks[aux_layer].mixer._last_aux_logits
                # aux_stack[h] is [B, L, V] for h=0..K_chain-2 (i.e., y_1 and y_2)
                aux_targets = [bridge1, bridge2]
                aux_losses = []
                for h in range(len(aux_stack)):
                    logits_h_atok = aux_stack[h][:, -1, :]     # [B, V]
                    aux_losses.append(F.cross_entropy(logits_h_atok, aux_targets[h]))
                aux_mean = sum(aux_losses) / len(aux_losses)
                loss = main_loss + aux_lambda * aux_mean
                aux_val = float(aux_mean.item())
            else:
                loss = main_loss
                aux_val = float("nan")

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
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
                "cleanup_T": T,
                "acc": accs,
            })
            acc_str = " ".join(f"{k}={v:.3f}" for k, v in accs.items())
            log(f"  step {step + 1:>4d} loss {main_loss.item():.3f} "
                f"aux {aux_val:.3f} λ {aux_lambda:.4f} T {T:.2f} "
                f"lr {lr:.1e} {acc_str}")

    wallclock = time.time() - start
    final_acc = curve[-1]["acc"]
    payload = {
        "tag": tag, "rig": "rig008_cleanup_aux", "seed": seed,
        "params": n_params, "cfg": asdict(cfg),
        "aux_style": "cleanup_plus_in_path_aux",
        "aux_layer": aux_layer,
        "aux_lambda_init": aux_lambda_init, "aux_decay_tau": aux_decay_tau,
        "T_init": T_init, "T_final": T_final,
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
    ap.add_argument("--T-init", type=float, default=5.0)
    ap.add_argument("--T-final", type=float, default=1.0)
    ap.add_argument("--tag-prefix", default="rig008_arm2")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = RunCfg(
        K1=args.K1, K2=args.K2, steps=args.steps, batch=args.batch,
        mode="mixed", eval_every=args.eval_every, device=args.device,
        hops=3, K3=args.K3,
    )
    hs_cfg = CleanupConfig(
        d_model=args.d_model, d_key=args.d_key, n_layers=args.n_layers,
        beta_bias_init=args.beta_bias_init, K_chain=args.K_chain,
    )
    out_dir = HERE / "outputs"

    def make_builder(hs_cfg):
        return lambda vocab_size: build_model(vocab_size, hs_cfg)

    per_seed = []
    for seed in args.seeds:
        payload = train_one_arm2(
            build_model_fn=make_builder(hs_cfg), seed=seed,
            cfg=cfg, hs_cfg=hs_cfg, out_dir=out_dir,
            tag=f"{args.tag_prefix}_s{seed}",
            aux_lambda_init=args.aux_lambda_init,
            aux_decay_tau=args.aux_decay_tau,
            T_init=args.T_init, T_final=args.T_final,
            aux_layer=args.aux_layer,
        )
        per_seed.append(payload)

    summary = {
        "rig": "rig008_cleanup_aux",
        "seeds": args.seeds,
        "hs_cfg": hs_cfg.__dict__,
        "run_cfg": cfg.__dict__,
        "aux": {
            "style": "cleanup_plus_in_path_aux",
            "layer": args.aux_layer,
            "lambda_init": args.aux_lambda_init,
            "decay_tau": args.aux_decay_tau,
            "T_init": args.T_init,
            "T_final": args.T_final,
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
