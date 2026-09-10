"""rig_014 — PART3 §49 task-shape arms.

Same DeltaNet + shared-bus + sym recipe as rig_013 (which cracks the
n=512 vocab wall and K=4 depth); only the task-shape sampler is swapped.

  --arm arm1_no_bank_sep    (bank-boundary SEPs removed; within-bank
                             pair ordering preserved)
  --arm arm2_interleaved    (all pairs shuffled per row; between-pair
                             SEP kept)
  --arm arm3_no_sep         (Arm 2 + no SEPs at all)

Test: does the mechanism form when the structural priors are removed?
C1 = all arms pass; C2 = only Arm 1 without curriculum, others need it;
C3 = Arm 2/3 fail even with the full recipe (structural priors are
load-bearing).
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
sys.path.insert(0, str(HERE.parent / "rig_013_delta_write"))

import torch
import torch.nn.functional as F

from common.harness import RunCfg, cosine_lr, left_pad
from common.tasks import TaskCfg, N_CONTROL
from common.tasks_shape import (
    sample_3hop_shape, sample_3hop_shape_identity,
    sample_3hop_reuse_arm1, sample_3hop_reuse_arm1_identity,
)
from model import DeltaBusConfig as SharedBusConfig, build_model
from eval_shape import eval_ladder_shape


def rho_schedule(step: int, warm_id_steps: int, anneal_steps: int) -> float:
    if step < warm_id_steps:
        return 0.0
    ramp = (step - warm_id_steps) / max(1, anneal_steps)
    return min(1.0, ramp)


def _contrastive_aux_mask(seq: torch.Tensor, vocab_size: int,
                          n_entities: int) -> torch.Tensor:
    B, L = seq.shape
    device = seq.device
    mask = torch.zeros(B, vocab_size, device=device, dtype=torch.bool)
    mask.scatter_(1, seq, True)
    entity_lo, entity_hi = N_CONTROL, N_CONTROL + n_entities
    keep = torch.zeros(vocab_size, device=device, dtype=torch.bool)
    keep[entity_lo:entity_hi] = True
    mask = mask & keep.unsqueeze(0)
    return mask


def train_one_shape(
    build_model_fn, seed: int, cfg: RunCfg, hs_cfg: SharedBusConfig,
    out_dir: Path, tag: str,
    arm: str,
    aux_lambda_init: float, aux_decay_tau: float,
    sym_lambda_init: float,
    T_init: float, T_final: float, aux_layer: int,
    warm_id_steps: int, anneal_steps: int,
    cfg_n_entities: int = 128, cfg_n_values: int = 128,
    contrastive_aux: bool = True,
    task_family: str = "shape",
    n_reuse: int = 2,
):
    log = lambda s: print(s, flush=True)

    torch.manual_seed(seed)
    task_cfg = TaskCfg(seed=seed, n_entities=cfg_n_entities, n_values=cfg_n_values)
    model = build_model_fn(task_cfg.vocab_size).to(cfg.device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay, betas=cfg.betas)

    train_tasks = [(K1_r, cfg.K2) for K1_r in range(1, cfg.K1 + 1)]
    train_gens = {tk: torch.Generator(device="cpu").manual_seed(seed + 1000 + i)
                  for i, tk in enumerate(train_tasks)}
    mode_gen = torch.Generator(device="cpu").manual_seed(seed + 42)

    log(f"[{tag}] arm={arm} params={n_params:,} tasks={train_tasks} "
        f"steps={cfg.steps} cleanup(T:{T_init}→{T_final}) "
        f"aux(layer={aux_layer}, λ_init={aux_lambda_init}, τ={aux_decay_tau}) "
        f"sym(λ={sym_lambda_init}) n_ent={cfg_n_entities} d_key={hs_cfg.d_key} "
        f"contrastive={contrastive_aux}")
    curve = []
    start = time.time()

    for step in range(cfg.steps):
        lr = cosine_lr(step, cfg.warmup, cfg.steps, cfg.lr, cfg.lr_floor)
        for pg in opt.param_groups:
            pg["lr"] = lr

        progress = step / max(1, cfg.steps - 1)
        T = T_init + (T_final - T_init) * progress
        rho = rho_schedule(step, warm_id_steps, anneal_steps)
        model.set_cleanup_T(T)

        use_real = torch.rand(1, generator=mode_gen).item() < rho

        tk = train_tasks[step % len(train_tasks)]
        k1r = tk[0]
        k2_use = max(cfg.K2, k1r)
        k3_use = max(cfg.K3, k2_use)
        if task_family == "shape":
            sampler = sample_3hop_shape if use_real else sample_3hop_shape_identity
            seq, tgt, bridge1, bridge2 = sampler(
                arm, k1r, k2_use, k3_use, task_cfg, cfg.batch, train_gens[tk],
                device=cfg.device)
        elif task_family == "reuse":
            sampler = sample_3hop_reuse_arm1 if use_real else sample_3hop_reuse_arm1_identity
            if use_real:
                seq, tgt, bridge1, bridge2 = sampler(
                    k1r, k2_use, k3_use, task_cfg, cfg.batch, train_gens[tk],
                    device=cfg.device, n_reuse=n_reuse)
            else:
                seq, tgt, bridge1, bridge2 = sampler(
                    k1r, k2_use, k3_use, task_cfg, cfg.batch, train_gens[tk],
                    device=cfg.device)
        else:
            raise ValueError(f"unknown task_family={task_family!r}")
        if cfg.pad_seq_to > 0:
            seq = left_pad(seq, cfg.pad_seq_to)

        aux_lambda = aux_lambda_init * math.exp(-step / aux_decay_tau)
        sym_lambda = sym_lambda_init * math.exp(-step / aux_decay_tau)
        apply_aux = aux_lambda >= 1e-6

        with torch.autocast(device_type=cfg.device, dtype=torch.bfloat16,
                            enabled=(cfg.device == "cuda" and cfg.autocast)):
            logits = model(seq)
            main_loss = F.cross_entropy(logits[:, -1], tgt)

            if apply_aux:
                aux_stack = model.blocks[aux_layer].mixer._last_aux_logits
                sym_stack = model.blocks[aux_layer].mixer._last_sym_logits
                aux_targets = [bridge1, bridge2]

                if contrastive_aux:
                    mask = _contrastive_aux_mask(seq, task_cfg.vocab_size, cfg_n_entities)
                    neg_inf = torch.finfo(aux_stack[0].dtype).min

                aux_losses = []
                sym_losses = []
                for h in range(len(aux_stack)):
                    logits_h_atok = aux_stack[h][:, -1, :]
                    sym_h_atok = sym_stack[h][:, -1, :]
                    if contrastive_aux:
                        logits_h_atok = logits_h_atok.masked_fill(~mask, neg_inf)
                        sym_h_atok = sym_h_atok.masked_fill(~mask, neg_inf)
                    aux_losses.append(F.cross_entropy(logits_h_atok, aux_targets[h]))
                    sym_losses.append(F.cross_entropy(sym_h_atok, aux_targets[h]))
                aux_mean = sum(aux_losses) / len(aux_losses)
                sym_mean = sum(sym_losses) / len(sym_losses)
                loss = main_loss + aux_lambda * aux_mean + sym_lambda * sym_mean
                aux_val = float(aux_mean.item())
                sym_val = float(sym_mean.item())
            else:
                loss = main_loss
                aux_val = float("nan")
                sym_val = float("nan")

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        if (step + 1) % cfg.eval_every == 0 or (step + 1) == cfg.steps:
            eval_K1_max = cfg.eval_K1_max if cfg.eval_K1_max > 0 else cfg.K1
            accs = eval_ladder_shape(model, task_cfg, eval_K1_max, cfg.K2,
                                     cfg.K3, cfg.eval_batch_size, cfg.eval_batches,
                                     cfg.device, seed + 10_000, arm,
                                     pad_seq_to=cfg.pad_seq_to,
                                     task_family=task_family, n_reuse=n_reuse)
            curve.append({
                "step": step + 1,
                "loss": float(main_loss.item()),
                "aux_loss": aux_val, "sym_loss": sym_val,
                "aux_lambda": aux_lambda, "sym_lambda": sym_lambda,
                "cleanup_T": T, "rho": rho, "used_real_this_batch": use_real,
                "acc": accs,
            })
            acc_str = " ".join(f"{k}={v:.3f}" for k, v in accs.items())
            mode_flag = "R" if use_real else "I"
            log(f"  step {step + 1:>5d} [{mode_flag}] loss {main_loss.item():.3f} "
                f"aux {aux_val:.3f} sym {sym_val:.3f} λa {aux_lambda:.4f} λs {sym_lambda:.4f} "
                f"T {T:.2f} ρ {rho:.2f} lr {lr:.1e} {acc_str}")

    wallclock = time.time() - start
    final_acc = curve[-1]["acc"]
    payload = {
        "tag": tag, "rig": "rig014_task_shape", "seed": seed, "arm": arm,
        "params": n_params, "cfg": asdict(cfg),
        "aux": {"style": "shared_bus_cleanup_plus_sym_aux",
                "layer": aux_layer, "lambda_init": aux_lambda_init,
                "sym_lambda_init": sym_lambda_init,
                "decay_tau": aux_decay_tau, "T_init": T_init, "T_final": T_final,
                "contrastive": contrastive_aux},
        "curriculum": {"warm_id_steps": warm_id_steps, "anneal_steps": anneal_steps},
        "vocab_size": task_cfg.vocab_size,
        "train_tasks": [list(t) for t in train_tasks],
        "final_acc": final_acc, "wallclock_s": round(wallclock, 3), "curve": curve,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{tag}.json").write_text(json.dumps(payload, indent=2))
    log(f"[{tag}] done in {wallclock:.1f}s  final k{cfg.K1}={final_acc.get(f'k{cfg.K1}'):.3f}")
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=("arm1_no_bank_sep", "arm2_interleaved", "arm3_no_sep",
                             "reuse_arm1"))
    ap.add_argument("--task-family", choices=("shape", "reuse"), default=None,
                    help="If unset, infers from --arm.")
    ap.add_argument("--n-reuse", type=int, default=2)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--K1", type=int, default=4)
    ap.add_argument("--K2", type=int, default=4)
    ap.add_argument("--K3", type=int, default=8)                  # matched difficulty §49
    ap.add_argument("--K4", type=int, default=4)                  # unused for hops=3
    ap.add_argument("--hops", type=int, default=3)
    ap.add_argument("--K-chain", type=int, default=3)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--d-key", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--beta-bias-init", type=float, default=-0.5)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--aux-lambda-init", type=float, default=0.3)
    ap.add_argument("--sym-lambda-init", type=float, default=0.3)
    ap.add_argument("--aux-decay-tau", type=float, default=4000.0)
    ap.add_argument("--aux-layer", type=int, default=0)
    ap.add_argument("--T-init", type=float, default=5.0)
    ap.add_argument("--T-final", type=float, default=1.0)
    ap.add_argument("--warm-id-steps", type=int, default=2000)
    ap.add_argument("--anneal-steps", type=int, default=4000)
    ap.add_argument("--n-entities", type=int, default=128)
    ap.add_argument("--n-values", type=int, default=128)
    ap.add_argument("--contrastive-aux", action="store_true", default=True)
    ap.add_argument("--no-contrastive-aux", dest="contrastive_aux", action="store_false")
    ap.add_argument("--tag-prefix", default="rig014_shape")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.task_family is None:
        args.task_family = "reuse" if args.arm.startswith("reuse_") else "shape"

    cfg = RunCfg(
        K1=args.K1, K2=args.K2, steps=args.steps, batch=args.batch,
        mode="mixed", eval_every=args.eval_every, device=args.device,
        hops=args.hops, K3=args.K3, K4=args.K4,
    )
    hs_cfg = SharedBusConfig(
        d_model=args.d_model, d_key=args.d_key, n_layers=args.n_layers,
        beta_bias_init=args.beta_bias_init, K_chain=args.K_chain,
    )
    out_dir = HERE / "outputs"

    def make_builder(hs_cfg):
        return lambda vocab_size: build_model(vocab_size, hs_cfg)

    per_seed = []
    for seed in args.seeds:
        payload = train_one_shape(
            build_model_fn=make_builder(hs_cfg), seed=seed,
            cfg=cfg, hs_cfg=hs_cfg, out_dir=out_dir,
            tag=f"{args.tag_prefix}_{args.arm}_s{seed}",
            arm=args.arm,
            aux_lambda_init=args.aux_lambda_init,
            sym_lambda_init=args.sym_lambda_init,
            aux_decay_tau=args.aux_decay_tau,
            T_init=args.T_init, T_final=args.T_final,
            aux_layer=args.aux_layer,
            warm_id_steps=args.warm_id_steps,
            anneal_steps=args.anneal_steps,
            cfg_n_entities=args.n_entities,
            cfg_n_values=args.n_values,
            contrastive_aux=args.contrastive_aux,
            task_family=args.task_family,
            n_reuse=args.n_reuse,
        )
        per_seed.append(payload)

    summary = {
        "rig": "rig014_task_shape", "arm": args.arm, "seeds": args.seeds,
        "hs_cfg": hs_cfg.__dict__, "run_cfg": cfg.__dict__,
        "n_entities": args.n_entities,
        "final_acc_per_seed": [p["final_acc"] for p in per_seed],
    }
    (out_dir / f"{args.tag_prefix}_{args.arm}_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[summary] {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
