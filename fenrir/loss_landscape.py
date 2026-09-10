"""Loss-landscape probes: interpolation + random-direction sweeps.

Two experiments, both inference-only (no training):

Experiment 1 -- linear interpolation between two checkpoints of the same seed.
  Given ckpt_A and ckpt_B (same seed, same architecture, different training outcome),
  compute:
     W(alpha) = (1 - alpha) * W_A + alpha * W_B      for alpha in [0, 1]
  Evaluate loss + k4 accuracy at each alpha on a fixed batch.
  Curve shape:
     smooth monotone -> same basin, dynamics-only separation
     bump in middle  -> real energy barrier between attractors
     flat then jump  -> non-trivial connectivity, straight-line doesn't reveal path

Experiment 2 -- random-direction probes from a single checkpoint.
  Sample N random weight-space directions with filter normalization (Li et al 2018).
  For each direction d and each magnitude t in a scan, compute
     W(t) = W_0 + t * d
  Measure loss. This maps the local loss landscape around W_0. If any direction
  decreases loss below the plateau, gradient dynamics could escape given the
  right step direction/size (says LR-refractoriness is a numerical artifact).
  If NO direction decreases loss, the local basin is deep -- LR intervention
  is genuinely useless.

Filter normalization: for each convolutional / linear weight parameter of shape
[C_out, C_in, ...], normalize each output-filter row so its norm matches the
corresponding row of W_0. This makes the perturbation scale-invariant across
layers with different magnitudes.
"""
from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F

from model import FenrirStack, MixerConfig
from tasks import TaskCfg, sample_long_2hop_injective_truncated

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def load_model(ckpt_path: Path, device: str) -> tuple[FenrirStack, dict]:
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["cfg"]
    mixer_cfg = MixerConfig(d_model=cfg["d_model"], d_key=cfg["d_key"],
                            d_value=cfg["d_value"])
    model = FenrirStack(
        vocab_size=cfg["vocab_size"], cfg=mixer_cfg, variant=cfg["variant"],
        n_layers=cfg["n_layers"],
        chunked=(cfg["variant"] == "rev" and cfg.get("use_eager", True)),
        use_eager=cfg.get("use_eager", True),
    ).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, cfg


def set_state(model: torch.nn.Module, state: dict) -> None:
    """Copy each tensor in `state` into the model's own state_dict slot.
    Only floating-point params are overwritten; anything else is left alone."""
    with torch.no_grad():
        msd = model.state_dict()
        for k, v in state.items():
            if k in msd and msd[k].dtype.is_floating_point:
                msd[k].copy_(v)


def interp_states(state_a: dict, state_b: dict, alpha: float) -> dict:
    """(1-alpha)*A + alpha*B on floating-point params. Non-float tensors: copy A."""
    out = {}
    for k, va in state_a.items():
        vb = state_b[k]
        if va.dtype.is_floating_point:
            out[k] = (1.0 - alpha) * va + alpha * vb
        else:
            out[k] = va
    return out


@torch.no_grad()
def eval_batch(model, tokens, targets, device):
    """Compute CE loss on last position + per-K accuracy on a fixed batch bundle."""
    with torch.autocast(device_type=device, dtype=torch.bfloat16,
                        enabled=(device == "cuda")):
        logits = model(tokens)
        loss = F.cross_entropy(logits[:, -1], targets)
        pred = logits[:, -1].argmax(-1)
        acc = (pred == targets).float().mean().item()
    return float(loss.item()), acc


def build_fixed_eval_bundle(cfg: dict, batch_size: int, device: str) -> list[dict]:
    """One batch per K rung {1..K1_max} using a deterministic generator."""
    task_cfg = TaskCfg(seed=cfg["seed"])
    K1_max, K2 = cfg["K1_max"], cfg["K2"]
    bundles = []
    for K1 in range(1, K1_max + 1):
        gen = torch.Generator(device="cpu").manual_seed(50_000 + K1)
        seq, tgt, _ = sample_long_2hop_injective_truncated(
            K1, K2, task_cfg, batch_size, gen, device=device)
        bundles.append({"K1": K1, "tokens": seq, "tgt": tgt})
    return bundles


def eval_ladder(model, bundles, device):
    """Return {'k1': acc, 'k2': acc, ...} + mean loss across rungs."""
    accs, losses = {}, []
    for b in bundles:
        loss, acc = eval_batch(model, b["tokens"], b["tgt"], device)
        accs[f"k{b['K1']}"] = acc
        losses.append(loss)
    return accs, sum(losses) / len(losses)


# ------------------ Experiment 1: interpolation --------------------------

def experiment_interp(ckpt_a: Path, ckpt_b: Path, n_alpha: int,
                      batch_size: int, device: str, tag: str) -> dict:
    model, cfg_a = load_model(ckpt_a, device)
    # cfg_b must match cfg_a on architecture-critical fields
    ck_b = torch.load(ckpt_b, map_location=device, weights_only=False)
    cfg_b = ck_b["cfg"]
    for field in ("d_model", "d_key", "d_value", "n_layers", "variant",
                  "vocab_size", "seed", "K1_max", "K2"):
        assert cfg_a[field] == cfg_b[field], f"mismatch {field}: {cfg_a[field]} vs {cfg_b[field]}"
    state_a = {k: v.detach().clone().to(device) for k, v in
               torch.load(ckpt_a, map_location=device, weights_only=False)["state_dict"].items()}
    state_b = {k: v.detach().clone().to(device) for k, v in ck_b["state_dict"].items()}
    bundles = build_fixed_eval_bundle(cfg_a, batch_size, device)

    alphas = [i / (n_alpha - 1) for i in range(n_alpha)]
    curve = []
    for alpha in alphas:
        s = interp_states(state_a, state_b, alpha)
        set_state(model, s)
        accs, mean_loss = eval_ladder(model, bundles, device)
        curve.append({"alpha": alpha, "mean_loss": mean_loss, "acc": accs})
        headline = accs.get(f"k{cfg_a['K1_max']}", None)
        print(f"  alpha={alpha:.3f}  loss={mean_loss:.3f}  "
              f"k{cfg_a['K1_max']}={headline:.3f}  full={accs}", flush=True)

    return {"tag": tag, "kind": "interp", "ckpt_a": ckpt_a.name, "ckpt_b": ckpt_b.name,
            "cfg": {k: cfg_a[k] for k in ["seed", "K1_max", "K2", "variant"]},
            "curve": curve, "batch_size": batch_size}


# ------------------ Experiment 2: random directions ---------------------

def make_filter_normalized_direction(state_dict: dict, gen: torch.Generator) -> dict:
    """Sample a random direction with filter-normalized magnitudes.

    For each floating-point parameter W of shape [d_out, d_in, ...]:
      - Sample d ~ N(0, 1) with the same shape.
      - For each row i in the leading (d_out) dimension:
          scale d[i] so that ||d[i]|| == ||W[i]|| (filter norm match).
      - For 1D tensors (biases, RMSNorm weights): scale so ||d|| == ||W||.

    This is the Li et al (2018) filter-normalized direction.
    """
    direction = {}
    for k, W in state_dict.items():
        if not W.dtype.is_floating_point or W.numel() == 0:
            direction[k] = torch.zeros_like(W)
            continue
        d = torch.empty_like(W)
        d.normal_(generator=gen)
        if W.ndim >= 2:
            # normalize each output-filter row
            d_flat = d.reshape(W.shape[0], -1)
            W_flat = W.reshape(W.shape[0], -1)
            d_norm = d_flat.norm(dim=1, keepdim=True) + 1e-12
            w_norm = W_flat.norm(dim=1, keepdim=True)
            d_flat.mul_(w_norm / d_norm)
        else:
            d_norm = d.norm() + 1e-12
            d.mul_(W.norm() / d_norm)
        direction[k] = d
    return direction


def experiment_random_directions(ckpt: Path, n_directions: int,
                                 magnitudes: list[float], batch_size: int,
                                 device: str, seed: int, tag: str) -> dict:
    model, cfg = load_model(ckpt, device)
    state_0 = {k: v.detach().clone().to(device) for k, v in
               torch.load(ckpt, map_location=device, weights_only=False)["state_dict"].items()}
    bundles = build_fixed_eval_bundle(cfg, batch_size, device)

    # Baseline at t=0
    set_state(model, state_0)
    base_accs, base_loss = eval_ladder(model, bundles, device)
    print(f"  baseline    loss={base_loss:.3f}  acc={base_accs}", flush=True)

    # Sample directions
    gen = torch.Generator(device="cpu").manual_seed(seed)
    # Move generator to same device as state for sampling
    cpu_gen = gen   # torch.empty_like uses same device as W; keep CPU gen but we sample on-device below
    results = []
    for d_idx in range(n_directions):
        # Sample direction using a per-tensor CPU gen so seed is reproducible
        direction = {}
        for k, W in state_0.items():
            if not W.dtype.is_floating_point or W.numel() == 0:
                direction[k] = torch.zeros_like(W); continue
            cpu_d = torch.empty(W.shape, dtype=W.dtype)
            cpu_d.normal_(generator=cpu_gen)
            d = cpu_d.to(W.device)
            if W.ndim >= 2:
                d_flat = d.reshape(W.shape[0], -1)
                W_flat = W.reshape(W.shape[0], -1)
                d_norm = d_flat.norm(dim=1, keepdim=True) + 1e-12
                w_norm = W_flat.norm(dim=1, keepdim=True)
                d_flat.mul_(w_norm / d_norm)
            else:
                d.mul_(W.norm() / (d.norm() + 1e-12))
            direction[k] = d

        for t in magnitudes:
            perturbed = {k: state_0[k] + t * direction[k] if state_0[k].dtype.is_floating_point
                         else state_0[k] for k in state_0}
            set_state(model, perturbed)
            accs, loss = eval_ladder(model, bundles, device)
            results.append({"direction": d_idx, "t": t, "loss": loss, "acc": accs})
            k_hi = accs.get(f"k{cfg['K1_max']}")
            print(f"  dir{d_idx:>2d} t={t:>+6.3f}  loss={loss:.3f}  k{cfg['K1_max']}={k_hi:.3f}",
                  flush=True)

    return {"tag": tag, "kind": "random_directions", "ckpt": ckpt.name,
            "n_directions": n_directions, "magnitudes": magnitudes,
            "baseline_loss": base_loss, "baseline_acc": base_accs,
            "results": results}


# ------------------ CLI ------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_int = sub.add_parser("interp", help="Linear interpolation between two ckpts.")
    p_int.add_argument("--ckpt-a", type=Path, required=True)
    p_int.add_argument("--ckpt-b", type=Path, required=True)
    p_int.add_argument("--n-alpha", type=int, default=25)
    p_int.add_argument("--batch-size", type=int, default=512)
    p_int.add_argument("--tag", required=True)
    p_int.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    p_rd = sub.add_parser("random", help="Random-direction probes from one ckpt.")
    p_rd.add_argument("--ckpt", type=Path, required=True)
    p_rd.add_argument("--n-directions", type=int, default=8)
    p_rd.add_argument("--magnitudes", type=str, default="0.1,0.3,1.0,3.0",
                     help="comma-separated magnitude values")
    p_rd.add_argument("--batch-size", type=int, default=512)
    p_rd.add_argument("--tag", required=True)
    p_rd.add_argument("--seed", type=int, default=42)
    p_rd.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = ap.parse_args()

    if args.cmd == "interp":
        result = experiment_interp(args.ckpt_a, args.ckpt_b, args.n_alpha,
                                    args.batch_size, args.device, args.tag)
    else:
        magnitudes = [float(x) for x in args.magnitudes.split(",")]
        result = experiment_random_directions(args.ckpt, args.n_directions,
                                               magnitudes, args.batch_size,
                                               args.device, args.seed, args.tag)

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / f"landscape__{args.tag}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[landscape] wrote {out_path.name}", flush=True)


if __name__ == "__main__":
    main()
