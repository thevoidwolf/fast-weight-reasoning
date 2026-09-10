"""Evaluate a FENRIR checkpoint on the K1 ladder.

  python eval.py --ckpt outputs/fenrir_rev_K1max4_K2_4_s0.ckpt
  python eval.py --ckpt <ckpt> --K1 5 6 7 8   # zero-shot depth generalisation

Writes a JSON to outputs/ with per-K1 accuracy and per-run metadata.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from model import FenrirStack, MixerConfig
from tasks import TaskCfg, sample_long_2hop_injective_truncated

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def load_checkpoint(ckpt_path: Path, device: str,
                    chunked_override: bool | None = None,
                    ) -> tuple[FenrirStack, dict]:
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["cfg"]
    mixer_cfg = MixerConfig(d_model=cfg["d_model"], d_key=cfg["d_key"],
                            d_value=cfg["d_value"])
    variant = cfg["variant"]
    chunked = chunked_override if chunked_override is not None \
        else (variant == "rev")
    model = FenrirStack(
        vocab_size=cfg["vocab_size"], cfg=mixer_cfg, variant=variant,
        n_layers=cfg["n_layers"], chunked=chunked,
    ).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, cfg


@torch.no_grad()
def eval_ladder(model: FenrirStack, task_cfg: TaskCfg,
                K1_list: list[int], K2: int, batch: int, n_batches: int,
                device: str, eval_seed: int) -> dict:
    results = {}
    for K1 in K1_list:
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
        acc = correct / total
        chance = 1.0 / K2
        results[f"k{K1}"] = {"acc": acc, "chance": chance, "n": total,
                             "above_chance": acc > chance + 0.05}
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--K1", type=int, nargs="+", default=None,
                    help="K1 ladder rungs to evaluate (default: 1..K1_max in ckpt)")
    ap.add_argument("--K2", type=int, default=None,
                    help="Hop-2 bank size (default: K2 in ckpt)")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--n-batches", type=int, default=4)
    ap.add_argument("--eval-seed", type=int, default=20_000)
    ap.add_argument("--chunked", dest="chunked", action="store_true", default=None)
    ap.add_argument("--no-chunked", dest="chunked", action="store_false")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tag", default=None, help="Output filename tag")
    args = ap.parse_args()

    model, cfg = load_checkpoint(args.ckpt, args.device,
                                 chunked_override=args.chunked)
    K1_list = args.K1 or list(range(1, cfg["K1_max"] + 1))
    K2 = args.K2 or cfg["K2"]
    task_cfg = TaskCfg(seed=cfg["seed"])

    print(f"[eval] {args.ckpt.name}  variant={cfg['variant']} K1={K1_list} K2={K2}",
          flush=True)
    results = eval_ladder(model, task_cfg, K1_list, K2, args.batch,
                          args.n_batches, args.device, args.eval_seed)
    for k, r in results.items():
        print(f"  {k}: {r['acc']:.4f} (chance {r['chance']:.3f}, n={r['n']})",
              flush=True)

    payload = {
        "ckpt": args.ckpt.name,
        "variant": cfg["variant"],
        "trained_K1_max": cfg["K1_max"],
        "trained_K2": cfg["K2"],
        "eval_K1_list": K1_list,
        "eval_K2": K2,
        "batch": args.batch, "n_batches": args.n_batches,
        "eval_seed": args.eval_seed,
        "results": results,
    }
    OUTPUT_DIR.mkdir(exist_ok=True)
    tag = args.tag or f"eval__{args.ckpt.stem}__K2_{K2}"
    out_path = OUTPUT_DIR / f"{tag}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[eval] wrote {out_path.name}", flush=True)


if __name__ == "__main__":
    main()
