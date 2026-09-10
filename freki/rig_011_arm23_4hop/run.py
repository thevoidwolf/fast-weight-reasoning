"""rig_011 — Extends rig_010's cleanup+aux+curriculum recipe to K_chain=4 / 4-hop.

Model: same CleanupStack, K_chain=4 (produces 3 aux logits per forward,
one per intermediate hop).
Task: real 4-hop (4 disjoint banks) + identity-homotopy curriculum (all
banks 2, 3, 4 self-paired in identity mode, shared pool with bank-1's b's).
Aux: supervises y_1 → bridge1, y_2 → bridge2, y_3 → bridge3 at ATOK,
via cleanup logits (non-bypassable). Aux always on (v2 fix carries over).
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
sys.path.insert(0, str(HERE.parent / "rig_009_id_homotopy"))

import torch
import torch.nn.functional as F

from common.harness import RunCfg, cosine_lr, eval_ladder, left_pad
from common.tasks import (
    TaskCfg,
    sample_long_4hop_injective_truncated,
)
from tasks_id import sample_long_4hop_identity_mode
from model import CleanupConfig, build_model


def rho_schedule(step: int, warm_id_steps: int, anneal_steps: int) -> float:
    if step < warm_id_steps:
        return 0.0
    ramp = (step - warm_id_steps) / max(1, anneal_steps)
    return min(1.0, ramp)


def train_one_4hop(
    build_model_fn, seed: int, cfg: RunCfg, hs_cfg: CleanupConfig,
    out_dir: Path, tag: str,
    aux_lambda_init: float, aux_decay_tau: float,
    T_init: float, T_final: float, aux_layer: int,
    warm_id_steps: int, anneal_steps: int,
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
    mode_gen = torch.Generator(device="cpu").manual_seed(seed + 42)

    log(f"[{tag}] params={n_params:,} tasks={train_tasks} steps={cfg.steps} "
        f"K_chain={hs_cfg.K_chain} hops=4 cleanup(T:{T_init}→{T_final}) "
        f"aux(layer={aux_layer}, λ_init={aux_lambda_init}, τ={aux_decay_tau}) "
        f"id_homotopy(warm={warm_id_steps}, anneal={anneal_steps})")
    curve = []
    start = time.time()

    for step in range(cfg.steps):
        lr = cosine_lr(step, cfg.warmup, cfg.steps, cfg.lr, cfg.lr_floor)
        for pg in opt.param_groups:
            pg["lr"] = lr

        progress = step / max(1, cfg.steps - 1)
        T = T_init + (T_final - T_init) * progress
        model.set_cleanup_T(T)

        rho = rho_schedule(step, warm_id_steps, anneal_steps)
        use_real = torch.rand(1, generator=mode_gen).item() < rho

        tk = train_tasks[step % len(train_tasks)]
        k1r = tk[0]
        k2_use = max(cfg.K2, k1r)
        k3_use = max(cfg.K3, k2_use)
        k4_use = max(cfg.K4, k3_use)
        if use_real:
            seq, tgt, bridge1, bridge2, bridge3 = sample_long_4hop_injective_truncated(
                k1r, k2_use, k3_use, k4_use, task_cfg, cfg.batch,
                train_gens[tk], device=cfg.device)
        else:
            seq, tgt, bridge1, bridge2, bridge3 = sample_long_4hop_identity_mode(
                k1r, k2_use, k3_use, k4_use, task_cfg, cfg.batch,
                train_gens[tk], device=cfg.device)
        if cfg.pad_seq_to > 0:
            seq = left_pad(seq, cfg.pad_seq_to)

        aux_lambda = aux_lambda_init * math.exp(-step / aux_decay_tau)
        apply_aux = aux_lambda >= 1e-6

        with torch.autocast(device_type=cfg.device, dtype=torch.bfloat16,
                            enabled=(cfg.device == "cuda" and cfg.autocast)):
            logits = model(seq)
            main_loss = F.cross_entropy(logits[:, -1], tgt)

            if apply_aux:
                aux_stack = model.blocks[aux_layer].mixer._last_aux_logits
                # K_chain=4 → aux_stack has 3 entries: h=0 (y_1), h=1 (y_2), h=2 (y_3)
                aux_targets = [bridge1, bridge2, bridge3]
                aux_losses = []
                for h in range(len(aux_stack)):
                    logits_h_atok = aux_stack[h][:, -1, :]
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
                               hops=4, K3=cfg.K3, K4=cfg.K4)
            curve.append({
                "step": step + 1, "loss": float(main_loss.item()),
                "aux_loss": aux_val, "aux_lambda": aux_lambda,
                "cleanup_T": T, "rho": rho,
                "used_real_this_batch": use_real, "acc": accs,
            })
            acc_str = " ".join(f"{k}={v:.3f}" for k, v in accs.items())
            mode_flag = "R" if use_real else "I"
            log(f"  step {step + 1:>4d} [{mode_flag}] loss {main_loss.item():.3f} "
                f"aux {aux_val:.3f} λ {aux_lambda:.4f} T {T:.2f} ρ {rho:.2f} "
                f"lr {lr:.1e} {acc_str}")

    wallclock = time.time() - start
    final_acc = curve[-1]["acc"]
    payload = {
        "tag": tag, "rig": "rig011_arm23_4hop", "seed": seed,
        "params": n_params, "cfg": asdict(cfg),
        "aux": {"style": "cleanup_plus_aux_plus_curriculum_4hop",
                "layer": aux_layer, "lambda_init": aux_lambda_init,
                "decay_tau": aux_decay_tau, "T_init": T_init, "T_final": T_final},
        "curriculum": {"warm_id_steps": warm_id_steps, "anneal_steps": anneal_steps},
        "vocab_size": task_cfg.vocab_size,
        "train_tasks": [list(t) for t in train_tasks],
        "final_acc": final_acc, "wallclock_s": round(wallclock, 3),
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
    ap.add_argument("--K4", type=int, default=4)
    ap.add_argument("--K-chain", type=int, default=4)
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
    ap.add_argument("--warm-id-steps", type=int, default=2000)
    ap.add_argument("--anneal-steps", type=int, default=4000)
    ap.add_argument("--tag-prefix", default="rig011_4hop")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = RunCfg(
        K1=args.K1, K2=args.K2, steps=args.steps, batch=args.batch,
        mode="mixed", eval_every=args.eval_every, device=args.device,
        hops=4, K3=args.K3, K4=args.K4,
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
        payload = train_one_4hop(
            build_model_fn=make_builder(hs_cfg), seed=seed,
            cfg=cfg, hs_cfg=hs_cfg, out_dir=out_dir,
            tag=f"{args.tag_prefix}_s{seed}",
            aux_lambda_init=args.aux_lambda_init,
            aux_decay_tau=args.aux_decay_tau,
            T_init=args.T_init, T_final=args.T_final,
            aux_layer=args.aux_layer,
            warm_id_steps=args.warm_id_steps,
            anneal_steps=args.anneal_steps,
        )
        per_seed.append(payload)

    summary = {
        "rig": "rig011_arm23_4hop",
        "seeds": args.seeds,
        "hs_cfg": hs_cfg.__dict__,
        "run_cfg": cfg.__dict__,
        "aux": {"style": "cleanup_plus_aux_plus_curriculum_4hop",
                "layer": args.aux_layer, "lambda_init": args.aux_lambda_init,
                "decay_tau": args.aux_decay_tau,
                "T_init": args.T_init, "T_final": args.T_final},
        "curriculum": {"warm_id_steps": args.warm_id_steps,
                       "anneal_steps": args.anneal_steps},
        "final_k4_per_seed": [p["final_acc"].get(f"k{args.K1}") for p in per_seed],
        "final_full_per_seed": [p["final_acc"] for p in per_seed],
        "wallclock_s_per_seed": [p["wallclock_s"] for p in per_seed],
    }
    (out_dir / f"{args.tag_prefix}_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
