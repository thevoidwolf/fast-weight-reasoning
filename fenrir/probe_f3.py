"""F3 logit-lens residual probe: at each layer's output, read the residual
through the model's own LM head and check whether the correct answer is
already decodable.

Localises the layer at which the answer becomes linearly decodable.

Writes one JSON per (checkpoint, K1, K2, tag) into outputs/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from model import FenrirStack, MixerConfig
from tasks import TaskCfg, sample_long_2hop_injective_truncated, positions

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def load_checkpoint(ckpt_path: Path, device: str) -> tuple[FenrirStack, dict]:
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["cfg"]
    mixer_cfg = MixerConfig(d_model=cfg["d_model"], d_key=cfg["d_key"],
                            d_value=cfg["d_value"])
    model = FenrirStack(
        vocab_size=cfg["vocab_size"], cfg=mixer_cfg, variant=cfg["variant"],
        n_layers=cfg["n_layers"], chunked=False,
        use_eager=cfg.get("use_eager", True),
    ).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, cfg


@torch.no_grad()
def residuals_per_layer(model: FenrirStack, seq: torch.Tensor) -> list[torch.Tensor]:
    """Return the residual stream after each block plus the input embedding.
    Result is a list of length n_layers+1; entry k is the residual after
    the first k blocks (entry 0 is the embedding)."""
    x = model.embed(seq)
    residuals = [x.detach()]
    for block in model.blocks:
        x = block(x)
        residuals.append(x.detach())
    return residuals


@torch.no_grad()
def collect(model: FenrirStack, task_cfg: TaskCfg, K1: int, K2: int,
            n_batches: int, batch: int, device: str, seed: int) -> dict:
    """For each of n_batches samples: read residual at ATOK from every
    layer, apply lm_head, and record whether the correct answer is the
    K2-constrained argmax."""
    P = positions(K1, K2)
    hit_per_layer: dict[int, list[bool]] = {}
    for b in range(n_batches):
        gen = torch.Generator(device="cpu").manual_seed(seed + b)
        seq, answer, _ = sample_long_2hop_injective_truncated(
            K1, K2, task_cfg, batch, gen, device=device)
        assert seq.shape[1] - 1 == P["atok"]
        residuals = residuals_per_layer(model, seq)
        val_toks = seq[:, P["val"]]                       # [B, K2]
        target_slot = (val_toks == answer.unsqueeze(1)).float().argmax(dim=1)

        for li, r in enumerate(residuals):
            atok_res = model.final_norm(r[:, P["atok"]])
            logits = model.lm_head(atok_res)              # [B, vocab]
            cand_scores = torch.gather(logits, 1, val_toks)  # [B, K2]
            pred_slot = cand_scores.argmax(dim=1)
            hits = (pred_slot == target_slot).tolist()
            hit_per_layer.setdefault(li, []).extend(hits)

    return {li: float(np.mean(hits)) for li, hits in hit_per_layer.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--K1", type=int, required=True)
    ap.add_argument("--K2", type=int, required=True)
    ap.add_argument("--n-batches", type=int, default=8)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=600)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model, cfg = load_checkpoint(args.ckpt, args.device)
    task_cfg = TaskCfg(seed=cfg["seed"])
    variant = cfg["variant"]

    print(f"[f3] {args.ckpt.name} variant={variant} K1={args.K1} K2={args.K2}", flush=True)
    accs = collect(model, task_cfg, args.K1, args.K2, args.n_batches,
                   args.batch, args.device, args.seed)

    chance = 1.0 / args.K2
    print(f"[f3] chance = {chance:.3f}", flush=True)
    for li in sorted(accs):
        label = "embed" if li == 0 else f"after L{li-1}"
        print(f"  {label:>10}  ATOK-decodable acc = {accs[li]:.4f}", flush=True)

    payload = {
        "ckpt": args.ckpt.name, "variant": variant,
        "K1": args.K1, "K2": args.K2, "chance": round(chance, 4),
        "n_batches": args.n_batches, "batch": args.batch,
        "acc_by_layer": {str(li): accs[li] for li in sorted(accs)},
    }
    OUTPUT_DIR.mkdir(exist_ok=True)
    tag = args.tag or f"probe_f3__{args.ckpt.stem}__K{args.K1}K{args.K2}"
    out = OUTPUT_DIR / f"{tag}.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"[f3] wrote {out.name}", flush=True)


if __name__ == "__main__":
    main()
