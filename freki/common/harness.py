"""Rapid-iteration training harness for FENRIR primitive candidates.

Design intent: one function `train_one` that takes a *model builder* and a
config, trains it on the 2-hop-with-K1-distractors task, and returns the
per-step trajectory. Deliberately stripped of features not needed for
first-signal readout — no aux loss, no chunked kernel, no compile,
no checkpointing.

Each rig provides a `build_model(vocab_size)` factory. The harness handles
seeding, task sampling, optimization, and eval.

First-signal readout: k1/k2/k4 accuracy trajectory at short intervals.
Attractor collapsed = universal-robust across seeds.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tasks import (
    TaskCfg, PAD,
    sample_long_2hop_injective_truncated,
    sample_long_3hop_injective_truncated,
    sample_long_4hop_injective_truncated,
)


def left_pad(seq: torch.Tensor, target_len: int) -> torch.Tensor:
    """Left-pad [B, L] with PAD token to [B, target_len]. ATOK stays at last position."""
    B, L = seq.shape
    if L >= target_len:
        return seq
    pad = torch.full((B, target_len - L), PAD, dtype=seq.dtype, device=seq.device)
    return torch.cat([pad, seq], dim=1)


@dataclass
class RunCfg:
    K1: int = 4
    K2: int = 4
    steps: int = 1500
    batch: int = 64
    lr: float = 3e-4
    lr_floor: float = 3e-5
    warmup: int = 200
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.95)
    grad_clip: float = 1.0
    mode: str = "mixed"          # "mixed" cycles K1=1..K1_max per step
    eval_every: int = 100
    eval_batches: int = 2
    eval_batch_size: int = 256
    device: str = "cuda"
    autocast: bool = True        # bf16 autocast on CUDA; disable to force fp32
    pad_seq_to: int = 0          # if > 0, left-pad every sequence to this length (fixed-shape training)
    hops: int = 2                # 2-hop (default), 3-hop, or 4-hop chain task
    K3: int = 4                  # only used when hops >= 3
    K4: int = 4                  # only used when hops == 4
    eval_K1_max: int = 0         # if > 0, eval at K1=1..eval_K1_max (extrapolation); else uses K1


def cosine_lr(step: int, warmup: int, total: int, base: float, floor: float) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    if step >= total:
        return floor
    progress = (step - warmup) / max(1, total - warmup)
    return floor + 0.5 * (base - floor) * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def eval_ladder(model: nn.Module, task_cfg: TaskCfg, K1_max: int, K2: int,
                batch: int, n_batches: int, device: str,
                eval_seed: int, pad_seq_to: int = 0,
                hops: int = 2, K3: int = 4, K4: int = 4) -> dict:
    """Evaluate on ladder {1..K1_max} at fixed K2 (and K3 if 3-hop, K4 if 4-hop).
    Returns {'k1': acc, ...}. Supports 2-hop, 3-hop, and 4-hop task variants.
    """
    model.eval()
    accs = {}
    for K1 in range(1, K1_max + 1):
        gen = torch.Generator(device="cpu").manual_seed(eval_seed + K1)
        correct = total = 0
        # Injective task requires K1 <= K2. For extrapolation past training K2,
        # scale K2 up to match K1 so the sampler stays valid.
        k2_use = max(K2, K1)
        for _ in range(n_batches):
            if hops == 2:
                seq, tgt, _ = sample_long_2hop_injective_truncated(
                    K1, k2_use, task_cfg, batch, gen, device=device)
            elif hops == 3:
                # For 3-hop, K1 <= K2 <= K3 required; enforce by clamping upward.
                k3_use = max(K3, k2_use)
                seq, tgt, *_ = sample_long_3hop_injective_truncated(
                    K1, k2_use, k3_use, task_cfg, batch, gen, device=device)
            elif hops == 4:
                k3_use = max(K3, k2_use)
                k4_use = max(K4, k3_use)
                seq, tgt, *_ = sample_long_4hop_injective_truncated(
                    K1, k2_use, k3_use, k4_use, task_cfg, batch, gen, device=device)
            else:
                raise ValueError(f"hops must be 2, 3, or 4, got {hops}")
            if pad_seq_to > 0:
                seq = left_pad(seq, pad_seq_to)
            with torch.autocast(device_type=device, dtype=torch.bfloat16,
                                enabled=(device == "cuda" and getattr(model, "_hrs_autocast", True))):
                pred = model(seq)[:, -1].argmax(-1)
            correct += (pred == tgt).sum().item()
            total += batch
        accs[f"k{K1}"] = correct / total
    model.train()
    return accs


def train_one(build_model: Callable[[int], nn.Module],
              rig_name: str, seed: int, cfg: RunCfg,
              out_dir: Path, tag: str | None = None,
              log: Callable[[str], None] | None = None,
              probe: Callable[[nn.Module], dict] | None = None) -> dict:
    """Train one seed. Returns payload dict; also writes {tag}.json to out_dir.

    build_model: (vocab_size: int) -> nn.Module. Called *after* seeding.
    probe: optional callable applied to the trained model. Its return dict
           is merged into the payload under key "probe".
    """
    log = log or (lambda s: print(s, flush=True))
    tag = tag or f"{rig_name}_s{seed}"

    torch.manual_seed(seed)
    task_cfg = TaskCfg(seed=seed)
    model = build_model(task_cfg.vocab_size).to(cfg.device)
    model._hrs_autocast = cfg.autocast
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay, betas=cfg.betas)

    if cfg.mode == "mixed":
        train_tasks = [(K1_r, cfg.K2) for K1_r in range(1, cfg.K1 + 1)]
    elif cfg.mode == "single":
        train_tasks = [(cfg.K1, cfg.K2)]
    else:
        raise ValueError(f"unknown mode {cfg.mode!r}")

    train_gens = {tk: torch.Generator(device="cpu").manual_seed(seed + 1000 + i)
                  for i, tk in enumerate(train_tasks)}

    log(f"[{tag}] params={n_params:,} tasks={train_tasks} steps={cfg.steps}")
    curve = []
    start = time.time()

    for step in range(cfg.steps):
        lr = cosine_lr(step, cfg.warmup, cfg.steps, cfg.lr, cfg.lr_floor)
        for pg in opt.param_groups:
            pg["lr"] = lr

        tk = train_tasks[step % len(train_tasks)]
        if cfg.hops == 2:
            seq, tgt, _ = sample_long_2hop_injective_truncated(
                tk[0], tk[1], task_cfg, cfg.batch, train_gens[tk],
                device=cfg.device)
        elif cfg.hops == 3:
            k1r = tk[0]
            k2_use = max(cfg.K2, k1r)
            k3_use = max(cfg.K3, k2_use)
            seq, tgt, *_ = sample_long_3hop_injective_truncated(
                k1r, k2_use, k3_use, task_cfg, cfg.batch, train_gens[tk],
                device=cfg.device)
        elif cfg.hops == 4:
            k1r = tk[0]
            k2_use = max(cfg.K2, k1r)
            k3_use = max(cfg.K3, k2_use)
            k4_use = max(cfg.K4, k3_use)
            seq, tgt, *_ = sample_long_4hop_injective_truncated(
                k1r, k2_use, k3_use, k4_use, task_cfg, cfg.batch, train_gens[tk],
                device=cfg.device)
        else:
            raise ValueError(f"cfg.hops must be 2, 3, or 4, got {cfg.hops}")
        if cfg.pad_seq_to > 0:
            seq = left_pad(seq, cfg.pad_seq_to)
        with torch.autocast(device_type=cfg.device, dtype=torch.bfloat16,
                            enabled=(cfg.device == "cuda" and cfg.autocast)):
            logits = model(seq)
            loss = F.cross_entropy(logits[:, -1], tgt)

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
                               hops=cfg.hops, K3=cfg.K3, K4=cfg.K4)
            curve.append({"step": step + 1, "loss": float(loss.item()), "acc": accs})
            acc_str = " ".join(f"{k}={v:.3f}" for k, v in accs.items())
            log(f"  step {step + 1:>4d} loss {loss.item():.3f} lr {lr:.1e} {acc_str}")

    wallclock = time.time() - start
    final_acc = curve[-1]["acc"]
    payload = {
        "tag": tag, "rig": rig_name, "seed": seed,
        "params": n_params, "cfg": asdict(cfg),
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
