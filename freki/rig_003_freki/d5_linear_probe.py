"""D5 — Linear probe on y_1 → bridge.

Settles Fable's correction of D4b: my full-readout decode of y_1 returned
0.000 to bridge because the readout was only trained to decode y_2
(value-subspace). If the mechanism is ideal in a rotated basis, a fresh
linear classifier on frozen y_1 embeddings should still recover the
bridge token with high accuracy.

Uses cached FREKI-2 state_dict from D4b.
Trains logistic regression (Linear + CE) on y_1@ATOK, target = bridge.
Fable's prediction: >= 95%.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import torch
import torch.nn.functional as F

from common.tasks import TaskCfg, sample_long_2hop_injective_truncated
from model import FrekiConfig, build_model


def collect_y1_at_atok(freki2, task_cfg, n_examples: int, device: str,
                       batch: int = 256):
    """Run FREKI-2 over n_examples of 2-hop, capture y_1@ATOK per layer + bridge."""
    ys = {l: [] for l in range(len(freki2.blocks))}
    bridges = []
    tgts = []

    # Hooks on chain_norms[0] per layer
    caps = {}
    hooks = []
    for l, blk in enumerate(freki2.blocks):
        def make_hook(l=l):
            def _h(_mod, _inp, out):
                caps[l] = out.detach()
            return _h
        hooks.append(blk.mixer.chain_norms[0].register_forward_hook(make_hook()))

    gen = torch.Generator(device="cpu").manual_seed(999)
    n_batches = (n_examples + batch - 1) // batch
    with torch.no_grad():
        for _ in range(n_batches):
            seq, tgt, bridge = sample_long_2hop_injective_truncated(
                K1=4, K2=4, cfg=task_cfg, batch=batch, gen=gen,
                device=device,
            )
            _ = freki2(seq)
            for l in range(len(freki2.blocks)):
                ys[l].append(caps[l][:, -1, :].cpu())   # [B, d_key]
            bridges.append(bridge.cpu())
            tgts.append(tgt.cpu())

    for h in hooks:
        h.remove()

    ys_out = {l: torch.cat(ys[l], dim=0)[:n_examples] for l in range(len(freki2.blocks))}
    bridges = torch.cat(bridges, dim=0)[:n_examples]
    tgts = torch.cat(tgts, dim=0)[:n_examples]
    return ys_out, bridges, tgts


def train_linear_probe(features: torch.Tensor, targets: torch.Tensor,
                       vocab_size: int, device: str, epochs: int = 500,
                       lr: float = 1e-2) -> tuple[float, float]:
    """Multi-class logistic regression. Returns (train_acc, val_acc) on 80/20."""
    n = features.shape[0]
    perm = torch.randperm(n)
    features = features[perm].to(device)
    targets = targets[perm].to(device)
    split = int(0.8 * n)
    Xtr, ytr = features[:split], targets[:split]
    Xva, yva = features[split:], targets[split:]

    W = torch.zeros(features.shape[1], vocab_size, device=device, requires_grad=True)
    b = torch.zeros(vocab_size, device=device, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr, weight_decay=1e-4)

    for _ in range(epochs):
        logits = Xtr @ W + b
        loss = F.cross_entropy(logits, ytr)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    with torch.no_grad():
        tr_pred = (Xtr @ W + b).argmax(-1)
        va_pred = (Xva @ W + b).argmax(-1)
        tr_acc = (tr_pred == ytr).float().mean().item()
        va_acc = (va_pred == yva).float().mean().item()
    return tr_acc, va_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-examples", type=int, default=8192)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ckpt-path", default="outputs/d4b_freki2_s0.pt")
    ap.add_argument("--tag", default="d5_linear_probe")
    args = ap.parse_args()

    out_dir = HERE / "outputs"
    ckpt_path = HERE / args.ckpt_path
    if not ckpt_path.exists():
        print(f"ERROR: no cached FREKI-2 checkpoint at {ckpt_path}. "
              f"Run d4b_logit_lens_fixed.py first.")
        return

    print("=" * 60)
    print("D5 linear probe: y_1@ATOK → bridge")
    print("=" * 60)

    blob = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    freki2 = build_model(blob["vocab_size"], FrekiConfig(K_chain=2)).to(args.device)
    freki2.load_state_dict({k: v.to(args.device) for k, v in blob["state_dict"].items()})
    freki2.eval()
    print(f"Loaded FREKI-2 (final k4 on 2-hop = {blob['final_k4']:.3f})")

    task_cfg = TaskCfg(seed=0)
    print(f"\nCollecting y_1@ATOK across {args.n_examples} examples...")
    ys, bridges, tgts = collect_y1_at_atok(
        freki2, task_cfg, args.n_examples, args.device,
    )

    # Sanity: how many unique bridges and tgts?
    n_uniq_bridge = int(bridges.unique().numel())
    n_uniq_tgt = int(tgts.unique().numel())
    print(f"Unique bridges: {n_uniq_bridge}, unique tgts: {n_uniq_tgt}")
    print(f"Chance = 1/{n_uniq_bridge} = {1.0/n_uniq_bridge:.3f} (uniform over seen bridges)")

    results = {}
    for target_name, target_tensor in [("bridge", bridges), ("tgt", tgts)]:
        print(f"\n--- Probe target: {target_name} ---")
        for l in range(len(freki2.blocks)):
            tr, va = train_linear_probe(
                ys[l], target_tensor, blob["vocab_size"], args.device,
            )
            print(f"  layer {l} y_1: train_acc={tr:.3f}  val_acc={va:.3f}")
            results[f"y1_layer{l}_target_{target_name}"] = {
                "train_acc": tr, "val_acc": va,
            }

    print("\n" + "=" * 60)
    print("D5 VERDICT")
    print("=" * 60)
    l0_bridge = results["y1_layer0_target_bridge"]["val_acc"]
    if l0_bridge > 0.90:
        print(f"Layer 0 y_1 → bridge val_acc = {l0_bridge:.3f}")
        print("→ Linear probe RECOVERS the bridge. Mechanism is ideal in a")
        print("  rotated basis (Fable's prediction). Cleanup constraint is")
        print("  NOT necessary to have a clean intermediate — the intermediate")
        print("  IS clean, just in the model's own linear code.")
    elif l0_bridge > 0.60:
        print(f"Layer 0 y_1 → bridge val_acc = {l0_bridge:.3f}")
        print("→ Partial recovery. Waypoint carries bridge info but noisily.")
    else:
        print(f"Layer 0 y_1 → bridge val_acc = {l0_bridge:.3f}")
        print("→ Bridge NOT linearly decodable. FREKI-2 solves 2-hop via")
        print("  genuinely distributed/non-token intermediate coordination.")
        print("  Cleanup constraint is a real constraint the model chose to skip.")

    result_path = out_dir / f"{args.tag}.json"
    result_path.write_text(json.dumps({
        "n_examples": args.n_examples,
        "n_unique_bridges": n_uniq_bridge,
        "n_unique_tgts": n_uniq_tgt,
        "results": results,
    }, indent=2))
    print(f"\nResults written to {result_path}")


if __name__ == "__main__":
    main()
