"""D4b — Fixed logit-lens probe.

D4 v1's decode skipped the silu(z) gate and the residual stream, which
made every intermediate decode to zero including the final layer's y_2
(known to reach k4=0.994 in the actual model). D4b fixes this by:
  1. Capturing z alongside y_h (via in_proj hook) so the mixer decode
     can apply `readout_proj → * silu(z) → out_proj → final_norm → lm_head`.
  2. Also decoding each block's residual output through
     `final_norm → lm_head` — the standard "logit lens" idiom.
  3. Caching the trained state_dict to disk so re-probes don't retrain.

Also keeps β analysis from D4 (unchanged).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import torch
import torch.nn.functional as F

from common.harness import RunCfg, train_one
from common.tasks import (
    TaskCfg, PAD, FACT, SEP, QTOK, ATOK, N_CONTROL,
    sample_long_2hop_injective_truncated,
)
from model import FrekiConfig, build_model


def train_or_load(args, out_dir: Path):
    """Train FREKI-2 seed 0 fresh (12K, ~7min). Cache state_dict alongside."""
    freki2_cfg = FrekiConfig(K_chain=2)
    cfg_train = RunCfg(
        K1=4, K2=4, steps=args.train_steps, batch=64,
        mode="mixed", eval_every=500, device=args.device, hops=2,
    )
    def build2(vocab_size):
        return build_model(vocab_size, freki2_cfg)

    ckpt_path = out_dir / f"{args.tag}_freki2_s{args.seed}.pt"
    if ckpt_path.exists() and not args.retrain:
        print(f"[phase 1] Loading cached state_dict from {ckpt_path}")
        blob = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        return freki2_cfg, cfg_train, blob["state_dict"], blob["vocab_size"], blob["final_k4"]

    capture = {}
    def snap(model):
        capture["state_dict"] = {k: v.detach().cpu().clone()
                                 for k, v in model.state_dict().items()}
        capture["vocab_size"] = int(model.embed.num_embeddings)
        return {"captured_params": len(capture["state_dict"])}

    print(f"\n[phase 1] Training FREKI-2 on 2-hop, {args.train_steps} steps...")
    payload = train_one(
        build_model=build2, rig_name="d4b_freki2_base",
        seed=args.seed, cfg=cfg_train, out_dir=out_dir,
        tag=f"{args.tag}_freki2_base_s{args.seed}",
        probe=snap,
    )
    final_k4 = payload["final_acc"].get("k4")
    print(f"[phase 1] final 2-hop k4={final_k4:.3f}")
    torch.save({
        "state_dict": capture["state_dict"],
        "vocab_size": capture["vocab_size"],
        "final_k4": final_k4,
    }, ckpt_path)
    print(f"[phase 1] cached to {ckpt_path}")
    return freki2_cfg, cfg_train, capture["state_dict"], capture["vocab_size"], final_k4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-steps", type=int, default=12000)
    ap.add_argument("--probe-batch", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="d4b")
    ap.add_argument("--retrain", action="store_true",
                    help="Retrain even if cached checkpoint exists")
    args = ap.parse_args()

    out_dir = HERE / "outputs"
    out_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print(f"D4b fixed logit-lens probe, seed {args.seed}")
    print("=" * 60)

    freki2_cfg, cfg_train, state_dict, vocab_size, final_k4 = train_or_load(args, out_dir)
    if final_k4 < 0.85:
        print("Base FREKI-2 didn't pass. D4b not interpretable. Aborting.")
        return

    freki2 = build_model(vocab_size, freki2_cfg).to(args.device)
    freki2.load_state_dict({k: v.to(args.device) for k, v in state_dict.items()})
    freki2.eval()

    # ---- Probe batch ----
    task_cfg = TaskCfg(seed=args.seed)
    gen = torch.Generator(device="cpu").manual_seed(args.seed + 20_000)
    seq, tgt, bridge = sample_long_2hop_injective_truncated(
        K1=4, K2=4, cfg=task_cfg, batch=args.probe_batch, gen=gen,
        device=args.device,
    )
    q_ent = seq[:, -2]
    B, L = seq.shape
    print(f"\n[phase 2] probe batch B={B}, L={L}")

    # ---- Hooks ----
    captured_y = {}     # (layer, chain_step) → [B, L, d_key]
    captured_z = {}     # layer → [B, L, d_inner]
    captured_block = {} # layer → [B, L, d_model]  (post-block residual)
    captured_beta_pre = {}  # layer → [B, L, 1]
    hooks = []

    for l, blk in enumerate(freki2.blocks):
        for h, cn in enumerate(blk.mixer.chain_norms):
            def make_ynhook(l=l, h=h):
                def _h(_mod, _inp, out):
                    captured_y[(l, h)] = out.detach()
                return _h
            hooks.append(cn.register_forward_hook(make_ynhook()))

        def make_zhook(l=l):
            def _h(_mod, _inp, out):
                # out is [B, L, 2*d_inner]; z is second half
                d_inner = out.shape[-1] // 2
                captured_z[l] = out[..., d_inner:].detach()
            return _h
        hooks.append(blk.mixer.in_proj.register_forward_hook(make_zhook()))

        def make_blockhook(l=l):
            def _h(_mod, _inp, out):
                captured_block[l] = out.detach()
            return _h
        hooks.append(blk.register_forward_hook(make_blockhook()))

        def make_betahook(l=l):
            def _h(_mod, _inp, out):
                captured_beta_pre[l] = out.detach()
            return _h
        hooks.append(blk.mixer.beta_proj.register_forward_hook(make_betahook()))

    with torch.no_grad():
        actual_logits = freki2(seq)                          # [B, L, V]
    for h in hooks:
        h.remove()

    # Sanity: model's own decode at ATOK matches tgt with high accuracy
    actual_pred = actual_logits[:, -1].argmax(-1)
    actual_acc = (actual_pred == tgt).float().mean().item()
    print(f"[sanity] model's real ATOK argmax accuracy on this batch: "
          f"{actual_acc:.3f} (should be ~0.99 for a passing FREKI-2)")

    # ---- Fixed logit lens: y_h via full readout w/ silu(z) gate ----
    print("\n[phase 3a] LOGIT LENS: mixer intermediates with silu(z) gate\n")

    def lens_mixer(y_atok, z_atok, layer_idx):
        """y_atok: [B, d_key], z_atok: [B, d_inner] → logits [B, V]"""
        blk = freki2.blocks[layer_idx]
        h = blk.mixer.readout_proj(y_atok)     # [B, d_inner]
        h = h * F.silu(z_atok)                 # apply the gate this time
        h = blk.mixer.out_proj(h)              # [B, d_model]
        h = freki2.final_norm(h)               # [B, d_model]
        return freki2.lm_head(h)               # [B, V]

    header = f"  {'layer':<7}{'step':<8}{'acc_tgt':>10}{'acc_bridge':>12}{'acc_q':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    mixer_results = {}
    for l in range(len(freki2.blocks)):
        z_atok = captured_z[l][:, -1, :]
        for h in range(freki2_cfg.K_chain):
            y_atok = captured_y[(l, h)][:, -1, :]
            logits = lens_mixer(y_atok, z_atok, layer_idx=l)
            pred = logits.argmax(-1)
            acc_tgt    = (pred == tgt).float().mean().item()
            acc_bridge = (pred == bridge).float().mean().item()
            acc_q      = (pred == q_ent).float().mean().item()
            mixer_results[f"layer{l}_y{h+1}"] = {
                "acc_tgt": acc_tgt, "acc_bridge": acc_bridge, "acc_q": acc_q,
            }
            print(f"  {l:<7}{'y_'+str(h+1):<8}"
                  f"{acc_tgt:>10.3f}{acc_bridge:>12.3f}{acc_q:>10.3f}")

    # ---- Standard logit lens: block residual outputs via final_norm + lm_head ----
    print("\n[phase 3b] LOGIT LENS: block residual outputs (post-block)\n")
    print(header)
    print("  " + "-" * (len(header) - 2))
    block_results = {}
    for l in range(len(freki2.blocks)):
        residual_atok = captured_block[l][:, -1, :]   # [B, d_model]
        normed = freki2.final_norm(residual_atok)
        logits = freki2.lm_head(normed)
        pred = logits.argmax(-1)
        acc_tgt    = (pred == tgt).float().mean().item()
        acc_bridge = (pred == bridge).float().mean().item()
        acc_q      = (pred == q_ent).float().mean().item()
        block_results[f"layer{l}_block_out"] = {
            "acc_tgt": acc_tgt, "acc_bridge": acc_bridge, "acc_q": acc_q,
        }
        print(f"  {l:<7}{'block':<8}"
              f"{acc_tgt:>10.3f}{acc_bridge:>12.3f}{acc_q:>10.3f}")

    # ---- β distribution ----
    print("\n[phase 4] β distribution by position role\n")
    seq_np = seq[0].cpu().tolist()
    roles = []
    for tok in seq_np:
        if tok < N_CONTROL:
            if tok == FACT: roles.append("FACT")
            elif tok == SEP: roles.append("SEP")
            elif tok == QTOK: roles.append("QTOK")
            elif tok == ATOK: roles.append("ATOK")
            elif tok == PAD: roles.append("PAD")
            else: roles.append("CTL")
        elif tok < N_CONTROL + task_cfg.n_entities:
            roles.append("ENT")
        else:
            roles.append("VAL")

    for l in range(len(freki2.blocks)):
        beta_val = torch.sigmoid(captured_beta_pre[l][0, :, 0].cpu())
        by_role = defaultdict(list)
        for pos, r in enumerate(roles):
            by_role[r].append(float(beta_val[pos]))
        print(f"  layer {l}:")
        for role in ["FACT", "SEP", "QTOK", "ATOK", "PAD", "CTL", "ENT", "VAL"]:
            vals = by_role.get(role, [])
            if vals:
                print(f"    {role:<6} n={len(vals):>3d}  "
                      f"mean_β={sum(vals)/len(vals):.3f}  "
                      f"max_β={max(vals):.3f}  min_β={min(vals):.3f}")

    # ---- Verdict ----
    print("\n" + "=" * 60)
    print("D4b VERDICT")
    print("=" * 60)
    print("\nCross-layer test (block residuals):")
    l0_tgt = block_results["layer0_block_out"]["acc_tgt"]
    l1_tgt = block_results["layer1_block_out"]["acc_tgt"]
    l2_tgt = block_results["layer2_block_out"]["acc_tgt"]
    print(f"  layer 0 → tgt: {l0_tgt:.3f}")
    print(f"  layer 1 → tgt: {l1_tgt:.3f}")
    print(f"  layer 2 → tgt: {l2_tgt:.3f}")
    if l2_tgt > 0.8 and l0_tgt < 0.3:
        print("  → Progressive answer formation: cross-layer composition IS the")
        print("    story. Each layer contributes to building the answer.")
    elif l0_tgt > 0.5:
        print("  → Answer forms early. In-mixer chain is doing real work.")

    print("\nIn-mixer test (mixer intermediates):")
    for l in range(len(freki2.blocks)):
        y1 = mixer_results[f"layer{l}_y1"]
        y2 = mixer_results[f"layer{l}_y2"]
        print(f"  layer {l}: y_1 → bridge={y1['acc_bridge']:.3f}, "
              f"y_2 → tgt={y2['acc_tgt']:.3f}")
        if y1["acc_bridge"] > 0.5:
            print(f"    ⇒ Layer {l}'s y_1 decodes to bridge — in-mixer chain")
            print(f"      genuinely does hop-then-hop here.")

    result = {
        "seed": args.seed,
        "train_steps": args.train_steps,
        "freki2_final_k4": final_k4,
        "actual_atok_acc": actual_acc,
        "mixer_lens": mixer_results,
        "block_lens": block_results,
    }
    result_path = out_dir / f"{args.tag}_s{args.seed}.json"
    result_path.write_text(json.dumps(result, indent=2))
    print(f"\nResults written to {result_path}")


if __name__ == "__main__":
    main()
