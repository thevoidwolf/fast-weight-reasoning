"""rig_005_hopstrat — Hop-Stratified FREKI, K_chain separate M matrices.

Supports two training modes:
  --aux-lambda-init 0    → plain training via common.harness.train_one (v1/v2 default)
  --aux-lambda-init > 0  → local train_one_aux with FENRIR-style deep supervision:
                             for each block b in blocks[:-1]:
                               aux_logits_b = lm_head(final_norm(block_out_b[:, -1]))
                               aux_loss    += CE(aux_logits_b, tgt)  # main tgt, not bridge
                             loss = main + λ(t) · mean(aux_losses)
                             λ(t) = λ_init · exp(-t / τ)
                           Uses forward hooks + shared lm_head — no fresh head,
                           no bypass path. Direct port of the fenrir/ chapter train.py.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common.harness import RunCfg, train_one, cosine_lr, eval_ladder, left_pad
from common.tasks import (
    TaskCfg,
    sample_long_3hop_injective_truncated,
    sample_long_2hop_injective_truncated,
)
from model import HopStratConfig, build_model


def train_one_aux(
    build_model_fn, rig_name: str, seed: int, cfg: RunCfg,
    out_dir: Path, tag: str,
    aux_lambda_init: float, aux_decay_tau: float,
    probe=None,
):
    """FENRIR-style deep-supervision aux training.

    For every block b in `model.blocks[:-1]`, register a forward hook that
    captures its output. During each step, compute aux logits at ATOK via
    `lm_head(final_norm(hooked_output[:, -1]))` (reusing the SHARED lm_head
    — no fresh head, no bypass) and cross-entropy against the main target
    `tgt`. Total loss = main_ce + λ(t) · mean(aux_ce over layers).

    λ(t) = aux_lambda_init · exp(-step / aux_decay_tau).

    Direct port of the fenrir/ chapter train.py's aux design onto
    HopStrat/3-hop. Hops must be 2 or 3 (harness supports both).
    """
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

    # Forward hooks: capture each block's residual output (post-block).
    # Cleared each step.
    intermediate_outs: list[torch.Tensor] = []
    def _make_hook():
        def _h(_mod, _inp, out):
            intermediate_outs.append(out)
        return _h
    hooks = [b.register_forward_hook(_make_hook()) for b in model.blocks]

    log(f"[{tag}] params={n_params:,} tasks={train_tasks} steps={cfg.steps} "
        f"aux(λ_init={aux_lambda_init}, τ={aux_decay_tau}) style=FENRIR-deep")
    curve = []
    start = time.time()

    try:
        for step in range(cfg.steps):
            lr = cosine_lr(step, cfg.warmup, cfg.steps, cfg.lr, cfg.lr_floor)
            for pg in opt.param_groups:
                pg["lr"] = lr

            tk = train_tasks[step % len(train_tasks)]
            k1r = tk[0]
            if cfg.hops == 2:
                seq, tgt, _ = sample_long_2hop_injective_truncated(
                    k1r, cfg.K2, task_cfg, cfg.batch, train_gens[tk],
                    device=cfg.device)
            elif cfg.hops == 3:
                k2_use = max(cfg.K2, k1r)
                k3_use = max(cfg.K3, k2_use)
                seq, tgt, *_ = sample_long_3hop_injective_truncated(
                    k1r, k2_use, k3_use, task_cfg, cfg.batch, train_gens[tk],
                    device=cfg.device)
            else:
                raise ValueError(f"cfg.hops must be 2 or 3, got {cfg.hops}")
            if cfg.pad_seq_to > 0:
                seq = left_pad(seq, cfg.pad_seq_to)

            aux_lambda = aux_lambda_init * math.exp(-step / aux_decay_tau)

            intermediate_outs.clear()
            with torch.autocast(device_type=cfg.device, dtype=torch.bfloat16,
                                enabled=(cfg.device == "cuda" and cfg.autocast)):
                logits = model(seq)
                main_loss = F.cross_entropy(logits[:, -1], tgt)
                # Aux: each block[:-1]'s residual output at ATOK, decoded via
                # the shared main head. Last block's output IS what feeds
                # main_loss — skip it (matches FENRIR's `[:-1]`).
                if aux_lambda >= 1e-6 and len(intermediate_outs) > 1:
                    aux_losses = []
                    for lo in intermediate_outs[:-1]:
                        normed = model.final_norm(lo[:, -1])
                        aux_logits = model.lm_head(normed)
                        aux_losses.append(F.cross_entropy(aux_logits, tgt))
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
        "tag": tag, "rig": rig_name, "seed": seed,
        "params": n_params, "cfg": asdict(cfg),
        "aux_style": "fenrir_deep",
        "aux_lambda_init": aux_lambda_init, "aux_decay_tau": aux_decay_tau,
        "vocab_size": task_cfg.vocab_size,
        "train_tasks": [list(t) for t in train_tasks],
        "final_acc": final_acc,
        "wallclock_s": round(wallclock, 3),
        "curve": curve,
    }
    if probe is not None:
        try:
            payload["probe"] = probe(model)
        except Exception as e:
            payload["probe_error"] = repr(e)
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
    ap.add_argument("--hops", type=int, default=3, choices=[2, 3])
    ap.add_argument("--K-chain", type=int, default=3)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--d-key", type=int, default=32)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--beta-bias-init", type=float, default=-0.5)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--tag-prefix", default="rig005_hopstrat")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--aux-lambda-init", type=float, default=0.0,
                    help="If > 0, use FENRIR-style deep-supervision aux (mean CE "
                         "against final tgt from each layer's residual, via shared "
                         "lm_head) with this initial weight")
    ap.add_argument("--aux-decay-tau", type=float, default=1500.0,
                    help="Decay time constant for aux loss weight schedule")
    args = ap.parse_args()

    cfg = RunCfg(
        K1=args.K1, K2=args.K2, steps=args.steps, batch=args.batch,
        mode="mixed", eval_every=args.eval_every, device=args.device,
        hops=args.hops, K3=args.K3,
    )
    hs_cfg = HopStratConfig(
        d_model=args.d_model, d_key=args.d_key, n_layers=args.n_layers,
        beta_bias_init=args.beta_bias_init, K_chain=args.K_chain,
    )
    out_dir = HERE / "outputs"

    def make_builder(hs_cfg):
        return lambda vocab_size: build_model(vocab_size, hs_cfg)

    def probe_gates(model):
        """Dump gate_proj + per-hop k/v projection stats per layer for diagnostic."""
        out = {}
        for i, blk in enumerate(model.blocks):
            gp = blk.mixer.gate_proj
            out[f"layer_{i}_gate_bias"] = gp.bias.detach().cpu().tolist()
            out[f"layer_{i}_gate_wnorm"] = float(gp.weight.detach().norm().cpu())
            out[f"layer_{i}_k_proj_wnorms"] = [
                float(kp.weight.detach().norm().cpu()) for kp in blk.mixer.k_projs
            ]
            out[f"layer_{i}_v_proj_wnorms"] = [
                float(vp.weight.detach().norm().cpu()) for vp in blk.mixer.v_projs
            ]
        return out

    per_seed = []
    use_aux = args.aux_lambda_init > 0
    for seed in args.seeds:
        if use_aux:
            payload = train_one_aux(
                build_model_fn=make_builder(hs_cfg),
                rig_name="rig005_hopstrat", seed=seed, cfg=cfg,
                out_dir=out_dir, tag=f"{args.tag_prefix}_s{seed}",
                aux_lambda_init=args.aux_lambda_init,
                aux_decay_tau=args.aux_decay_tau,
                probe=probe_gates,
            )
        else:
            payload = train_one(
                build_model=make_builder(hs_cfg),
                rig_name="rig005_hopstrat", seed=seed, cfg=cfg,
                out_dir=out_dir, tag=f"{args.tag_prefix}_s{seed}",
                probe=probe_gates,
            )
        per_seed.append(payload)

    summary = {
        "rig": "rig005_hopstrat",
        "seeds": args.seeds,
        "hs_cfg": hs_cfg.__dict__,
        "run_cfg": cfg.__dict__,
        "aux": {
            "style": "fenrir_deep" if use_aux else "none",
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
