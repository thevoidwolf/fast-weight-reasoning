"""F5 Q/K alignment probe: measure the algebraic identifications the
mechanism derivation predicts.

Three cosines, each in a "same" vs "control" setup:

  (a) q_ATOK vs k[k1_target]
      Prediction: query aligns with the queried hop-1 key.
  (b) k[bridge_i] vs k[k2_f(i)]
      Prediction: the hop-1 write's bridge token and the hop-2 read's
      k2 token that carries the answer share a common representation
      (the mechanism relies on this to route through M).
  (c) v[bridge_i] vs k[k1_i]
      Prediction: the hop-1 write's value at the bridge position
      matches the hop-1 key at that position (in a symmetry the model
      may or may not learn).

For each, we report cos(same) and cos(control) as means over n_batches.

Writes one JSON per (checkpoint, K1, K2, tag) into outputs/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model import FenrirStack, MixerConfig, FenrirMixer
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


def cos(a: torch.Tensor, b: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return F.cosine_similarity(a, b, dim=dim, eps=1e-8)


@torch.no_grad()
def collect(model: FenrirStack, task_cfg: TaskCfg, K1: int, K2: int,
            n_batches: int, batch: int, device: str, seed: int,
            probed_layer: int) -> dict:
    P = positions(K1, K2)
    mixer: FenrirMixer = model.blocks[probed_layer].mixer
    mixer.probe_cache = True

    a_same, a_ctrl = [], []
    b_same, b_ctrl = [], []
    c_same, c_ctrl = [], []

    for bi in range(n_batches):
        gen = torch.Generator(device="cpu").manual_seed(seed + bi)
        seq, answer, bridge = sample_long_2hop_injective_truncated(
            K1, K2, task_cfg, batch, gen, device=device)
        _ = model(seq)
        cache = mixer._cache
        q = cache["q"]                                    # [B, L, d]
        k = cache["k"]
        v = cache["v"]

        q_atok = q[:, P["atok"]]                          # [B, d]
        k1_toks = seq[:, P["k1"]]                         # [B, K1]
        # target_idx in the k1 tokens: where the query entity appears
        q_ent = seq[:, P["q_ent"]]                        # [B]
        target_i = (k1_toks == q_ent.unsqueeze(1)).float().argmax(dim=1)  # [B]

        k1_pos = torch.tensor(P["k1"], device=device)     # [K1]
        target_k1_pos = k1_pos[target_i]                  # [B]
        k_at_target_k1 = k[torch.arange(batch, device=device), target_k1_pos]

        # (a) q_ATOK vs k[target_k1]: same
        a_same.append(cos(q_atok, k_at_target_k1).cpu())
        # (a) control: q_ATOK vs k[each other k1] averaged
        for i in range(K1):
            if i == 0:
                a_ctrl_batch = cos(q_atok, k[:, P["k1"][i]])
            else:
                a_ctrl_batch = a_ctrl_batch + cos(q_atok, k[:, P["k1"][i]])
        a_ctrl.append(((a_ctrl_batch - cos(q_atok, k_at_target_k1)) / (K1 - 1)).cpu()
                      if K1 > 1 else cos(q_atok, k_at_target_k1).cpu())

        # (b) k[bridge_i] vs k[k2_f(i)]: same is diagonal, control is off-diagonal
        # bridge token id -> which k2 slot
        k2_toks = seq[:, P["k2"]]                          # [B, K2]
        bridges = seq[:, P["bridge"]]                      # [B, K1]
        # For each row and each i: find j such that k2_toks[b, j] == bridges[b, i]
        b_pos = torch.tensor(P["bridge"], device=device)   # [K1]
        k2_pos = torch.tensor(P["k2"], device=device)      # [K2]
        for i in range(K1):
            j = (k2_toks == bridges[:, i:i+1]).float().argmax(dim=1)   # [B]
            same = cos(k[:, b_pos[i]], k[torch.arange(batch, device=device), k2_pos[j]])
            b_same.append(same.cpu())
            # controls: k[bridge_i] vs k[k2_j'] for j' != f(i)
            ctrl_accum = torch.zeros_like(same)
            n_ctrl = 0
            for jp in range(K2):
                mask = (j != jp)
                if mask.any():
                    ctrl_accum[mask] = ctrl_accum[mask] + \
                        cos(k[mask, b_pos[i]], k[mask, k2_pos[jp]])
                    n_ctrl += 1
            if n_ctrl > 0:
                b_ctrl.append((ctrl_accum / max(1, K2 - 1)).cpu())

        # (c) v[bridge_i] vs k[k1_i]: pairwise same
        for i in range(K1):
            c_same.append(cos(v[:, b_pos[i]], k[:, k1_pos[i]]).cpu())
            # control: v[bridge_i] vs k[k1_j], j != i
            ctrl_accum = torch.zeros(batch, device=device)
            n_ctrl = 0
            for jp in range(K1):
                if jp != i:
                    ctrl_accum = ctrl_accum + cos(v[:, b_pos[i]], k[:, k1_pos[jp]])
                    n_ctrl += 1
            if n_ctrl > 0:
                c_ctrl.append((ctrl_accum / n_ctrl).cpu())

    mixer.probe_cache = False

    def mean(x): return float(torch.cat(x).mean()) if x else float("nan")
    return {
        "a_qatok_vs_ktarget": {"same": mean(a_same), "ctrl": mean(a_ctrl)},
        "b_bridge_vs_k2match": {"same": mean(b_same), "ctrl": mean(b_ctrl)},
        "c_bridgeval_vs_k1":   {"same": mean(c_same), "ctrl": mean(c_ctrl)},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--K1", type=int, required=True)
    ap.add_argument("--K2", type=int, required=True)
    ap.add_argument("--layer", type=int, default=1,
                    help="Which layer's mixer to probe (default L1)")
    ap.add_argument("--n-batches", type=int, default=8)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=700)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model, cfg = load_checkpoint(args.ckpt, args.device)
    task_cfg = TaskCfg(seed=cfg["seed"])
    variant = cfg["variant"]

    print(f"[f5] {args.ckpt.name} variant={variant} K1={args.K1} K2={args.K2} "
          f"L={args.layer}", flush=True)
    result = collect(model, task_cfg, args.K1, args.K2, args.n_batches,
                     args.batch, args.device, args.seed, args.layer)

    for name, r in result.items():
        gap = r["same"] - r["ctrl"]
        print(f"  {name:>28}  same={r['same']:+.3f} ctrl={r['ctrl']:+.3f} "
              f"gap={gap:+.3f}", flush=True)

    payload = {
        "ckpt": args.ckpt.name, "variant": variant,
        "K1": args.K1, "K2": args.K2, "probed_layer": args.layer,
        "n_batches": args.n_batches, "batch": args.batch,
        "alignments": result,
    }
    OUTPUT_DIR.mkdir(exist_ok=True)
    tag = args.tag or f"probe_f5__{args.ckpt.stem}__K{args.K1}K{args.K2}_L{args.layer}"
    out = OUTPUT_DIR / f"{tag}.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"[f5] wrote {out.name}", flush=True)


if __name__ == "__main__":
    main()
