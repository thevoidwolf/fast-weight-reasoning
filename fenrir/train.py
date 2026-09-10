"""Train a single FENRIR checkpoint.

Protocol:
  optimizer  : AdamW, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01
  schedule   : cosine with warmup=200 steps, floor=3e-5
  precision  : bf16 autocast on CUDA
  grad clip  : 1.0 (global norm)
  batch      : 64
  steps      : 4000 (default)
  mode       : mixed (default: cycle through K1=1..K1_max ladder tasks per step)

Note on optimizer hyperparameters. betas=(0.9, 0.95) follows the
large-scale LM convention (GPT-3, Chinchilla, LLaMA) rather than the
original Adam paper default of (0.9, 0.999). The smaller beta_2
shortens the running-variance window from about 1000 steps to about
20 steps, which reduces optimizer instability during long training
runs at scale. At this experiment's size (about 1.5M parameters,
4000 steps) the choice likely has minor effect. It is preserved
across all FENRIR runs for comparability with the rig-of-record and
was not separately ablated in this paper.

Writes:
  outputs/<tag>.ckpt   : model state + config for eval / probes
  outputs/<tag>.json   : training trajectory (loss + per-K1 acc at every 200 steps)

Example:
  python train.py --variant rev --K1 4 --seed 0
  python train.py --variant fwd --K1 4 --seed 0 --tag fenrir_fwd_s0
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from model import FenrirStack, MixerConfig
from tasks import TaskCfg, sample_long_2hop_injective_truncated

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def cosine_lr(step: int, warmup: int, total: int, base: float, floor: float) -> float:
    """Cosine schedule with linear warmup and a floor."""
    if step < warmup:
        return base * (step + 1) / warmup
    if step >= total:
        return floor
    progress = (step - warmup) / max(1, total - warmup)
    return floor + 0.5 * (base - floor) * (1.0 + math.cos(math.pi * progress))


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def eval_ladder(model: FenrirStack, cfg: TaskCfg, K1_max: int, K2: int,
                batch: int, n_batches: int, device: str,
                eval_seed: int) -> dict:
    """Evaluate on the full K1 ladder {1..K1_max} at fixed K2. Returns
    a dict {'k1': acc, 'k2': acc, ...} keyed by ladder rung."""
    model.eval()
    accs = {}
    for K1 in range(1, K1_max + 1):
        gen = torch.Generator(device="cpu").manual_seed(eval_seed + K1)
        correct = total = 0
        for _ in range(n_batches):
            seq, tgt, _ = sample_long_2hop_injective_truncated(
                K1, K2, cfg, batch, gen, device=device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16,
                                enabled=(device == "cuda")):
                pred = model(seq)[:, -1].argmax(-1)
            correct += (pred == tgt).sum().item()
            total += batch
        accs[f"k{K1}"] = correct / total
    model.train()
    return accs


def train_one(variant: str, K1: int, K2: int, seed: int, steps: int,
              d_model: int, d_key: int, n_layers: int, batch: int,
              mode: str, chunked: bool, use_checkpoint: bool,
              eval_every: int, eval_batches: int, eval_batch_size: int,
              tag: str, device: str, compile: bool = False,
              use_eager: bool = True, aux_lambda: float = 0.0,
              aux_decay_tau: float = 0.0) -> dict:
    torch.manual_seed(seed)

    task_cfg = TaskCfg(seed=seed)
    mixer_cfg = MixerConfig(d_model=d_model, d_key=d_key, d_value=d_key)

    model = FenrirStack(
        vocab_size=task_cfg.vocab_size, cfg=mixer_cfg, variant=variant,
        n_layers=n_layers, chunked=chunked, use_checkpoint=use_checkpoint,
        use_eager=use_eager,
    ).to(device)
    n_params = count_params(model)

    if compile:
        # Recompilation-friendly: this task has only a few distinct sequence
        # lengths (one per K1 rung) so dynamic=True keeps the graph reusable.
        model = torch.compile(model, dynamic=True, mode="reduce-overhead")

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01,
                            betas=(0.9, 0.95))

    # Training tasks (per mode)
    ladder = [(K1_r, K2) for K1_r in range(1, K1 + 1)]      # (1, K2) .. (K1, K2)
    if mode == "mixed":
        train_tasks = ladder
    elif mode == "joint":
        train_tasks = [(1, K2), (K1, K2)]                    # {1, K1_max} 0.5/0.5 loss
    elif mode == "single":
        train_tasks = [(K1, K2)]
    else:
        raise ValueError(f"unknown mode {mode!r}")

    # One deterministic generator per training task
    train_gens = {tk: torch.Generator(device="cpu").manual_seed(seed + 1000 + i)
                  for i, tk in enumerate(train_tasks)}

    print(f"[train] variant={variant} mode={mode} seed={seed} "
          f"steps={steps} params={n_params:,} chunked={chunked}", flush=True)
    print(f"[train] train_tasks={train_tasks}", flush=True)

    curve = []
    step_to_95 = None
    start = time.time()

    # Hooks for aux loss: capture residual output of each FenrirBlock at answer
    # position. Only used when aux_lambda > 0. Cleared each step.
    intermediate_outs: list[torch.Tensor] = []
    if aux_lambda > 0:
        def _make_hook():
            def _h(_mod, _inp, out):
                intermediate_outs.append(out)
            return _h
        aux_hooks = [b.register_forward_hook(_make_hook()) for b in model.blocks]
    else:
        aux_hooks = []

    def compute_loss_with_aux(seq, tgt, step_now: int = 0):
        """Forward + main CE + (if aux_lambda>0) per-layer aux CE on intermediate
        residuals via final_norm + lm_head at the answer position.

        If aux_decay_tau > 0, the effective aux weight is
            aux_lambda * exp(-step_now / aux_decay_tau)
        """
        intermediate_outs.clear()
        logits = model(seq)
        main_loss = F.cross_entropy(logits[:, -1], tgt)
        if aux_lambda <= 0 or not intermediate_outs:
            return main_loss
        eff_lambda = aux_lambda
        if aux_decay_tau > 0:
            eff_lambda = aux_lambda * math.exp(-step_now / aux_decay_tau)
        if eff_lambda < 1e-6:
            return main_loss
        aux_losses = []
        for lo in intermediate_outs[:-1]:
            normed = model.final_norm(lo[:, -1])
            aux_logits = model.lm_head(normed)
            aux_losses.append(F.cross_entropy(aux_logits, tgt))
        aux_mean = sum(aux_losses) / len(aux_losses)
        return main_loss + eff_lambda * aux_mean

    for step in range(steps):
        lr = cosine_lr(step, warmup=200, total=steps, base=3e-4, floor=3e-5)
        for pg in opt.param_groups:
            pg["lr"] = lr

        if mode == "joint":
            with torch.autocast(device_type=device, dtype=torch.bfloat16,
                                enabled=(device == "cuda")):
                loss = 0.0
                for tk in train_tasks:
                    seq, tgt, _ = sample_long_2hop_injective_truncated(
                        tk[0], tk[1], task_cfg, batch, train_gens[tk],
                        device=device)
                    loss = loss + (1.0 / len(train_tasks)) * compute_loss_with_aux(seq, tgt, step)
        else:
            tk = train_tasks[step % len(train_tasks)]
            seq, tgt, _ = sample_long_2hop_injective_truncated(
                tk[0], tk[1], task_cfg, batch, train_gens[tk], device=device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16,
                                enabled=(device == "cuda")):
                loss = compute_loss_with_aux(seq, tgt, step)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if (step + 1) % eval_every == 0 or (step + 1) == steps:
            accs = eval_ladder(model, task_cfg, K1_max=K1, K2=K2,
                               batch=eval_batch_size,
                               n_batches=eval_batches, device=device,
                               eval_seed=seed + 10_000)
            curve.append({"step": step + 1, "loss": float(loss.item()), "acc": accs})
            if step_to_95 is None and accs.get(f"k{K1}", 0.0) >= 0.95:
                step_to_95 = step + 1
            acc_str = " ".join(f"{k}={v:.3f}" for k, v in accs.items())
            print(f"  step {step + 1:>5d} loss {loss.item():.3f} lr {lr:.2e} "
                  f"ladder {acc_str}", flush=True)

    wallclock = time.time() - start
    final_acc = curve[-1]["acc"]

    # Clean up aux hooks so they don't leak into any subsequent forward pass
    for h in aux_hooks:
        h.remove()

    payload = {
        "tag": tag, "variant": variant, "mode": mode, "seed": seed,
        "K1_max": K1, "K2": K2, "steps": steps, "batch": batch,
        "d_model": d_model, "d_key": d_key, "d_value": d_key,
        "n_layers": n_layers, "vocab_size": task_cfg.vocab_size,
        "params": n_params, "chunked": chunked, "use_eager": use_eager,
        "aux_lambda": aux_lambda, "aux_decay_tau": aux_decay_tau,
        "precision": "bf16_autocast" if device == "cuda" else "fp32",
        "train_tasks": [list(t) for t in train_tasks],
        "final_acc": final_acc,
        "final_acc_headline": final_acc.get(f"k{K1}"),
        "step_to_95_headline": step_to_95,
        "wallclock_s": round(wallclock, 3),
        "curve": curve,
    }

    OUTPUT_DIR.mkdir(exist_ok=True)
    json_path = OUTPUT_DIR / f"{tag}.json"
    ckpt_path = OUTPUT_DIR / f"{tag}.ckpt"
    json_path.write_text(json.dumps(payload, indent=2))
    torch.save({
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "cfg": {
            "d_model": d_model, "d_key": d_key, "d_value": d_key,
            "n_layers": n_layers, "vocab_size": task_cfg.vocab_size,
            "variant": variant, "seed": seed, "K1_max": K1, "K2": K2,
            "steps": steps, "mode": mode, "use_eager": use_eager,
        },
    }, ckpt_path)
    print(f"[train] wrote {json_path.name} + {ckpt_path.name}  "
          f"final k{K1}={final_acc.get(f'k{K1}'):.3f} in {wallclock:.1f}s",
          flush=True)
    return payload


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--variant", required=True, choices=["fwd", "rev"])
    ap.add_argument("--K1", type=int, default=4,
                    help="Max K1 in the training ladder (default 4)")
    ap.add_argument("--K2", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--d-key", type=int, default=32)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--mode", choices=["mixed", "joint", "single"],
                    default="mixed",
                    help="Training curriculum. 'mixed' cycles through K1=1..K1_max "
                         "per step; 'joint' uses a two-task loss on {1, K1_max}; "
                         "'single' trains on K1_max only.")
    ap.add_argument("--chunked", dest="chunked", action="store_true", default=None,
                    help="Use the chunked kernel for the rev variant "
                         "(default: on for rev, unavailable for fwd)")
    ap.add_argument("--no-chunked", dest="chunked", action="store_false")
    ap.add_argument("--use-checkpoint", action="store_true",
                    help="Recompute intra-chunk intermediates in backward "
                         "(chunked only; saves memory).")
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--eval-batches", type=int, default=4)
    ap.add_argument("--eval-batch-size", type=int, default=512)
    ap.add_argument("--tag", default=None,
                    help="Output filename tag (default fenrir_<variant>_s<seed>)")
    ap.add_argument("--compile", action="store_true",
                    help="Wrap the model in torch.compile (dynamic shapes). "
                         "Empirically not helpful under mixed-mode training: "
                         "cycling through K1=1..K1_max forces per-shape "
                         "recompilation of the mixer's sequential scan and "
                         "warmup exceeds the full training wallclock. Left "
                         "in for use with --mode single, where one shape "
                         "amortises the warmup.")
    ap.add_argument("--no-eager", dest="use_eager", action="store_false",
                    default=True,
                    help="Ablation: skip the k+M*k eager term at both "
                         "training and inference. k_eff = k (standard fast-weight "
                         "write with no address perturbation). Forces --no-chunked "
                         "because the chunked kernel is derived from the eager-term "
                         "recurrence.")
    ap.add_argument("--aux-lambda", type=float, default=0.0,
                    help="Intervention (C): apply per-layer aux CE loss on "
                         "intermediate residual outputs at the answer position "
                         "(final_norm + lm_head). Total = main_CE + lambda*mean(aux). "
                         "0.0 (default) reproduces baseline training bit-for-bit.")
    ap.add_argument("--aux-decay-tau", type=float, default=0.0,
                    help="If > 0, decay aux weight over steps: "
                         "eff_lambda = aux_lambda * exp(-step / tau). "
                         "0.0 (default) keeps aux constant at aux_lambda.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def main():
    args = parse_args()
    tag = args.tag or f"fenrir_{args.variant}_K1max{args.K1}_K2{args.K2}_s{args.seed}"

    chunked = args.chunked
    if chunked is None:
        chunked = (args.variant == "rev")   # default: on for rev, off for fwd
    if not args.use_eager:
        chunked = False                     # ablation forces sequential path

    train_one(
        variant=args.variant, K1=args.K1, K2=args.K2, seed=args.seed,
        steps=args.steps, d_model=args.d_model, d_key=args.d_key,
        n_layers=args.n_layers, batch=args.batch,
        mode=args.mode, chunked=chunked, use_checkpoint=args.use_checkpoint,
        eval_every=args.eval_every, eval_batches=args.eval_batches,
        eval_batch_size=args.eval_batch_size,
        tag=tag, device=args.device, compile=args.compile,
        use_eager=args.use_eager, aux_lambda=args.aux_lambda,
        aux_decay_tau=args.aux_decay_tau,
    )


if __name__ == "__main__":
    main()
