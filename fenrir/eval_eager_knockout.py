"""Eval a trained FENRIR checkpoint with the eager term switched off at
inference. Complements F4 knockout: F4 removes the eager term from the
probe's reconstruction only; this script removes it from the model's own
forward pass on the actual task.

The strong claim if accuracy collapses: the trained model IS using the
eager term to compute its answer, not just carrying it as an incidental
term the probe happens to include. Combined with the ablation training
(--no-eager from the start), this gives a two-way causal test:

  (A) Trained with eager, eval with eager      : baseline accuracy
  (B) Trained with eager, eval WITHOUT eager   : this script  (inference-time knockout)
  (C) Trained WITHOUT eager, eval WITHOUT eager: ablation_no_eager.sh (training-time knockout)

If (A) high and (B) low: model depends on eager at inference.
If (A) high and (C) low: eager is necessary to FORM the mechanism.
If both hold, the eager term is causally load-bearing in the mechanism
in both directions.

Writes one JSON per checkpoint into outputs/eval_eager_ko__<tag>.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from model import FenrirStack, MixerConfig
from tasks import TaskCfg, sample_long_2hop_injective_truncated

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def load_checkpoint(ckpt_path: Path, device: str, use_eager: bool):
    """Load a trained ckpt but override use_eager on all mixers."""
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["cfg"]
    mixer_cfg = MixerConfig(d_model=cfg["d_model"], d_key=cfg["d_key"],
                            d_value=cfg["d_value"])
    model = FenrirStack(
        vocab_size=cfg["vocab_size"], cfg=mixer_cfg, variant=cfg["variant"],
        n_layers=cfg["n_layers"], chunked=False,
        use_eager=use_eager,   # override at construction
    ).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, cfg


@torch.no_grad()
def eval_ladder(model: FenrirStack, cfg: dict, K1_max: int, K2: int,
                batch: int, n_batches: int, device: str, eval_seed: int) -> dict:
    task_cfg = TaskCfg(seed=cfg["seed"])
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
    return accs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--K1", type=int, required=True)
    ap.add_argument("--K2", type=int, required=True)
    ap.add_argument("--n-batches", type=int, default=8)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=800)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print(f"[eval_ko] {args.ckpt.name} K1={args.K1} K2={args.K2}", flush=True)

    # Baseline: trained with eager, eval with eager
    model_on, cfg = load_checkpoint(args.ckpt, args.device, use_eager=True)
    acc_on = eval_ladder(model_on, cfg, args.K1, args.K2, args.batch,
                         args.n_batches, args.device, args.seed)

    # Knockout: same weights, eval WITHOUT eager
    model_off, _ = load_checkpoint(args.ckpt, args.device, use_eager=False)
    acc_off = eval_ladder(model_off, cfg, args.K1, args.K2, args.batch,
                          args.n_batches, args.device, args.seed)

    chance = 1.0 / args.K2
    print(f"[eval_ko] chance = {chance:.3f}", flush=True)
    print(f"  eager ON : " + " ".join(f"{k}={v:.3f}" for k, v in acc_on.items()),
          flush=True)
    print(f"  eager OFF: " + " ".join(f"{k}={v:.3f}" for k, v in acc_off.items()),
          flush=True)

    payload = {
        "ckpt": args.ckpt.name, "variant": cfg["variant"],
        "K1": args.K1, "K2": args.K2, "chance": round(chance, 4),
        "n_batches": args.n_batches, "batch": args.batch,
        "acc_eager_on": acc_on,
        "acc_eager_off": acc_off,
    }
    OUTPUT_DIR.mkdir(exist_ok=True)
    tag = args.tag or f"eval_eager_ko__{args.ckpt.stem}"
    out = OUTPUT_DIR / f"{tag}.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"[eval_ko] wrote {out.name}", flush=True)


if __name__ == "__main__":
    main()
