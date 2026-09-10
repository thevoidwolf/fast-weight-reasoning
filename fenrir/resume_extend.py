"""Continue training a FENRIR checkpoint under a chosen LR schedule.

Purpose: test H1 (failing-seed rescue by extended training). Loads a
checkpoint written by train.py, continues training for --extra-steps more
steps under one of two LR schedules:

  --arm floor    : LR pinned at 3e-5 (the original cosine floor) throughout
                   continuation. Tests "pure more time" hypothesis.
  --arm restart  : Fresh cosine — 200-step warmup to base 3e-4 then cosine
                   decay to 3e-5 over the extra window. Tests "escape a
                   shallow local min with a kick" hypothesis.

Uses a fresh deterministic batch stream (seed offset lifted by 10*8000) so
the continuation is not a replay of the pretraining batches. Task, model
config, and optimizer type are reconstructed from the ckpt's cfg block;
this script does NOT restore optimizer state -- Adam moments are re-init to
zero for both arms (a warm-restart of optimizer moments alongside the LR
warm-restart is the standard SGDR practice, so this matches convention for
the "restart" arm and is a small departure from "true continuation" for the
"floor" arm; the alternative would require saving optimizer state in the
original checkpoint, which train.py doesn't do).

Instruments continuation with the diagnostics we'd want for early-warning
regardless of rescue outcome:
  - grad_norm (post-clip)
  - weight_l2_vs_ckpt (parameter L2 distance from the loaded ckpt)
  - loss_var_400 (rolling variance of loss over last 400 steps)

Writes outputs/rescue_h1__<orig_tag>__<arm>.json  (extended curve)
       outputs/rescue_h1__<orig_tag>__<arm>.ckpt  (final state)
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import deque
from pathlib import Path

import torch
import torch.nn.functional as F

from model import FenrirStack, MixerConfig
from tasks import TaskCfg, sample_long_2hop_injective_truncated

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

BASE_LR = 3e-4
LR_FLOOR = 3e-5
WARMUP = 200


def lr_at(step: int, arm: str, total: int) -> float:
    """LR for the continuation only (step counts from 0 at continuation start)."""
    if arm == "floor":
        return LR_FLOOR
    # arm == "restart": warmup then cosine
    if step < WARMUP:
        return BASE_LR * (step + 1) / WARMUP
    if step >= total:
        return LR_FLOOR
    progress = (step - WARMUP) / max(1, total - WARMUP)
    return LR_FLOOR + 0.5 * (BASE_LR - LR_FLOOR) * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def eval_ladder(model: FenrirStack, task_cfg: TaskCfg, K1_max: int, K2: int,
                batch: int, n_batches: int, device: str, eval_seed: int) -> dict:
    model.eval()
    accs = {}
    for K1 in range(1, K1_max + 1):
        gen = torch.Generator(device="cpu").manual_seed(eval_seed + K1)
        correct = total = 0
        for _ in range(n_batches):
            seq, tgt, _ = sample_long_2hop_injective_truncated(
                K1, K2, task_cfg, batch, gen, device=device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16,
                                enabled=(device == "cuda")):
                pred = model(seq)[:, -1].argmax(-1)
            correct += (pred == tgt).sum().item()
            total += batch
        accs[f"k{K1}"] = correct / total
    model.train()
    return accs


def build_train_tasks(mode: str, K1: int, K2: int):
    ladder = [(K1_r, K2) for K1_r in range(1, K1 + 1)]
    if mode == "mixed":
        return ladder
    if mode == "joint":
        return [(1, K2), (K1, K2)]
    if mode == "single":
        return [(K1, K2)]
    raise ValueError(f"unknown mode {mode!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--ckpt", type=Path, required=True,
                    help="Checkpoint written by train.py")
    ap.add_argument("--arm", choices=["floor", "restart"], required=True)
    ap.add_argument("--extra-steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--eval-batches", type=int, default=4)
    ap.add_argument("--eval-batch-size", type=int, default=512)
    ap.add_argument("--stream-shift", type=int, default=80000,
                    help="Offset added to per-task generator seeds so the "
                         "continuation sees a fresh deterministic stream, "
                         "not a replay of pretraining batches.")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tag", default=None,
                    help="Output tag override (default: rescue_h1__<orig>__<arm>)")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    cfg = ck["cfg"]
    orig_tag = args.ckpt.stem
    tag = args.tag or f"rescue_h1__{orig_tag}__{args.arm}"

    variant, mode = cfg["variant"], cfg["mode"]
    K1, K2 = cfg["K1_max"], cfg["K2"]
    seed = cfg["seed"]
    use_eager = cfg.get("use_eager", True)

    task_cfg = TaskCfg(seed=seed)
    mixer_cfg = MixerConfig(d_model=cfg["d_model"], d_key=cfg["d_key"],
                            d_value=cfg["d_value"])
    model = FenrirStack(
        vocab_size=cfg["vocab_size"], cfg=mixer_cfg, variant=variant,
        n_layers=cfg["n_layers"],
        chunked=(variant == "rev" and use_eager),
        use_eager=use_eager,
    ).to(args.device)
    model.load_state_dict(ck["state_dict"])
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Snapshot for weight-L2-vs-ckpt diagnostic
    ckpt_snapshot = {k: v.detach().clone() for k, v in model.state_dict().items()}

    opt = torch.optim.AdamW(model.parameters(), lr=LR_FLOOR, weight_decay=0.01,
                            betas=(0.9, 0.95))

    train_tasks = build_train_tasks(mode, K1, K2)
    train_gens = {tk: torch.Generator(device="cpu").manual_seed(
                        seed + 1000 + i + args.stream_shift)
                  for i, tk in enumerate(train_tasks)}

    print(f"[resume] arm={args.arm} ckpt={args.ckpt.name} seed={seed} "
          f"variant={variant} mode={mode} K1={K1} K2={K2} "
          f"extra_steps={args.extra_steps}", flush=True)
    print(f"[resume] train_tasks={train_tasks} stream_shift={args.stream_shift}", flush=True)

    # Baseline eval at continuation start (step 0 of continuation)
    accs0 = eval_ladder(model, task_cfg, K1, K2,
                        args.eval_batch_size, args.eval_batches, args.device,
                        eval_seed=seed + 10_000)
    print(f"[resume] baseline (ckpt eval): {accs0}", flush=True)

    curve = [{"step": 0, "loss": None, "acc": accs0,
              "grad_norm": None, "weight_l2_vs_ckpt": 0.0, "loss_var_400": None,
              "lr": None}]

    loss_window = deque(maxlen=400)   # per-step losses for variance diag
    start = time.time()

    for step in range(args.extra_steps):
        lr = lr_at(step, args.arm, args.extra_steps)
        for pg in opt.param_groups:
            pg["lr"] = lr

        if mode == "joint":
            with torch.autocast(device_type=args.device, dtype=torch.bfloat16,
                                enabled=(args.device == "cuda")):
                loss = 0.0
                for tk in train_tasks:
                    seq, tgt, _ = sample_long_2hop_injective_truncated(
                        tk[0], tk[1], task_cfg, args.batch, train_gens[tk],
                        device=args.device)
                    logits = model(seq)
                    loss = loss + (1.0 / len(train_tasks)) * F.cross_entropy(
                        logits[:, -1], tgt)
        else:
            tk = train_tasks[step % len(train_tasks)]
            seq, tgt, _ = sample_long_2hop_injective_truncated(
                tk[0], tk[1], task_cfg, args.batch, train_gens[tk],
                device=args.device)
            with torch.autocast(device_type=args.device, dtype=torch.bfloat16,
                                enabled=(args.device == "cuda")):
                logits = model(seq)
                loss = F.cross_entropy(logits[:, -1], tgt)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        opt.step()

        loss_val = float(loss.item())
        loss_window.append(loss_val)

        if (step + 1) % args.eval_every == 0 or (step + 1) == args.extra_steps:
            accs = eval_ladder(model, task_cfg, K1, K2,
                               args.eval_batch_size, args.eval_batches,
                               args.device, eval_seed=seed + 10_000)
            # Weight L2 distance from checkpoint (only trainable params)
            with torch.no_grad():
                sqsum = 0.0
                for k, v in model.state_dict().items():
                    if v.dtype.is_floating_point:
                        sqsum += (v - ckpt_snapshot[k]).pow(2).sum().item()
                wl2 = math.sqrt(sqsum)
                if len(loss_window) >= 2:
                    m = sum(loss_window) / len(loss_window)
                    lvar = sum((x - m) ** 2 for x in loss_window) / len(loss_window)
                else:
                    lvar = None
            curve.append({
                "step": step + 1, "loss": loss_val, "acc": accs,
                "grad_norm": grad_norm, "weight_l2_vs_ckpt": wl2,
                "loss_var_400": lvar, "lr": lr,
            })
            print(f"  cont step {step+1:>5d} loss {loss_val:.3f} lr {lr:.2e} "
                  f"gnorm {grad_norm:.3f} wL2 {wl2:.2f} lvar {lvar:.3f} "
                  f"ladder {accs}", flush=True)

    wallclock = time.time() - start
    final_acc = curve[-1]["acc"]
    payload = {
        "tag": tag, "orig_tag": orig_tag, "arm": args.arm,
        "orig_cfg": cfg, "extra_steps": args.extra_steps,
        "batch": args.batch, "stream_shift": args.stream_shift,
        "params": n_params, "wallclock_s": round(wallclock, 3),
        "baseline_acc": accs0, "final_acc": final_acc,
        "final_acc_headline": final_acc.get(f"k{K1}"),
        "curve": curve,
    }
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / f"{tag}.json").write_text(json.dumps(payload, indent=2))
    torch.save({
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "cfg": {**cfg, "extra_steps_from": orig_tag, "arm": args.arm},
    }, OUTPUT_DIR / f"{tag}.ckpt")
    print(f"[resume] wrote {tag}.json + .ckpt  "
          f"baseline k{K1}={accs0.get(f'k{K1}'):.3f} -> "
          f"final k{K1}={final_acc.get(f'k{K1}'):.3f} in {wallclock:.1f}s",
          flush=True)


if __name__ == "__main__":
    main()
