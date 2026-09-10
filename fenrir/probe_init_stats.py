"""Init-only statistics for each seed — no training.

Reconstruct FenrirStack(seed=s) as it would look at step 0, then measure
several candidate init statistics on a fixed task batch. Does any init
statistic separate PASS from fail seeds, before any optimizer step?

Statistics per seed:
  - per-layer weight Frobenius norm (mixer.q_proj, k_proj, v_proj, beta_proj,
    readout_proj, out_proj, in_proj, conv1d), summarised as a small vector
  - per-layer init eager magnitude on the fixed batch: mean of |M·k|
    accumulated in-scan at each layer (measured with probe_cache)
  - per-layer output activation norm at each layer (residual output)
  - hidden-state effective-rank at each layer (sum-of-normalised-singular-values)
  - init parameter-gradient L2 per named module on a single loss.backward()
    call from a fixed task batch
  - init loss on the fixed task batch (should be around ln(vocab_size))

Writes outputs/init_stats.json with per-seed stats + a "pass" label pulled
from the corresponding training-run JSON (looking for k4_final >= 0.85).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from model import FenrirStack, MixerConfig
from tasks import TaskCfg, sample_long_2hop_injective_truncated

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def eff_rank(x: torch.Tensor) -> float:
    """Effective rank as exp(entropy of normalised singular values).
    x: [N, d]. Returns a scalar in [1, min(N,d)]."""
    x = x.detach().float()
    if x.numel() == 0 or x.shape[0] < 2:
        return 0.0
    s = torch.linalg.svdvals(x - x.mean(0, keepdim=True))
    s = s[s > 1e-8]
    if s.numel() == 0:
        return 0.0
    p = s / s.sum()
    ent = -(p * (p + 1e-30).log()).sum().item()
    import math
    return math.exp(ent)


def measure_seed(seed: int, variant: str, K1: int, K2: int,
                 d_model: int, d_key: int, n_layers: int,
                 batch: int, device: str) -> dict:
    """All init statistics for one seed. Deterministic under this seed."""
    torch.manual_seed(seed)
    task_cfg = TaskCfg(seed=seed)
    mixer_cfg = MixerConfig(d_model=d_model, d_key=d_key, d_value=d_key)
    model = FenrirStack(vocab_size=task_cfg.vocab_size, cfg=mixer_cfg,
                        variant=variant, n_layers=n_layers,
                        chunked=False,   # need sequential path for probe_cache
                        use_eager=True).to(device)

    # Fixed batch for all measurements (K1_max, K2)
    fixed_gen = torch.Generator(device="cpu").manual_seed(99999)
    seq, tgt, _ = sample_long_2hop_injective_truncated(
        K1, K2, task_cfg, batch, fixed_gen, device=device)

    # --- 1. Per-mixer weight Frobenius norms + input-gradient at token 0 ----
    layer_wnorms = []
    for L, block in enumerate(model.blocks):
        m = block.mixer
        wnorms = {}
        for name in ["in_proj", "q_proj", "k_proj", "v_proj",
                    "beta_proj", "readout_proj", "out_proj"]:
            mod = getattr(m, name)
            wnorms[name] = float(mod.weight.norm().item())
            if hasattr(mod, "bias") and mod.bias is not None:
                wnorms[f"{name}_bias"] = float(mod.bias.norm().item())
        wnorms["conv1d"] = float(m.conv1d.weight.norm().item())
        layer_wnorms.append(wnorms)

    # --- 2. Init eager magnitude per layer, on fixed batch ------------------
    #  Turn probe_cache on so each mixer stores its eager stream
    for block in model.blocks:
        block.mixer.probe_cache = True
    model.eval()
    with torch.no_grad():
        # bf16 to match training precision
        with torch.autocast(device_type=device, dtype=torch.bfloat16,
                            enabled=(device == "cuda")):
            logits = model(seq)
    eager_mag = []
    for L, block in enumerate(model.blocks):
        cache = block.mixer._cache
        # eager cache is [B, L, d]. Mean absolute magnitude per position.
        e = cache["eager"].float()
        eager_mag.append({
            "mean_abs": float(e.abs().mean().item()),
            "mean_abs_last_pos": float(e[:, -1].abs().mean().item()),
            "std": float(e.std().item()),
        })
    # Turn probe_cache off
    for block in model.blocks:
        block.mixer.probe_cache = False
        block.mixer._cache = {}

    # --- 3. Per-layer residual activation norm + effective rank -------------
    #  Recompute forward and grab intermediate residuals via hooks
    resid_norms = []
    resid_effrank = []
    hidden_last_pos = []
    hooks = []
    intermediates = {}
    def make_hook(L):
        def _h(module, inp, out):
            intermediates[L] = out.detach()
        return _h
    for L, block in enumerate(model.blocks):
        hooks.append(block.register_forward_hook(make_hook(L)))
    with torch.no_grad():
        with torch.autocast(device_type=device, dtype=torch.bfloat16,
                            enabled=(device == "cuda")):
            logits = model(seq)
    for h in hooks:
        h.remove()
    for L in range(n_layers):
        r = intermediates[L].float()   # [B, L_seq, d_model]
        resid_norms.append(float(r.norm(dim=-1).mean().item()))
        # Effective rank on flattened [B*L_seq, d_model]
        resid_effrank.append(eff_rank(r.reshape(-1, r.shape[-1])))
        # Hidden state at last position, for later distance metrics
        hidden_last_pos.append(r[:, -1].mean(0).cpu().tolist())

    # --- 4. Init loss + init parameter-gradient L2 per module ---------------
    model.train()
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()
    with torch.autocast(device_type=device, dtype=torch.bfloat16,
                        enabled=(device == "cuda")):
        logits = model(seq)
        init_loss = F.cross_entropy(logits[:, -1], tgt)
    init_loss.backward()
    gradnorms = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            # Group by prefix up to layer
            gradnorms[name] = float(p.grad.norm().item())

    # Aggregate gradnorms per layer for easy reading
    grad_per_layer = []
    for L in range(n_layers):
        pref = f"blocks.{L}.mixer."
        s = {k[len(pref):]: v for k, v in gradnorms.items() if k.startswith(pref)}
        grad_per_layer.append(s)
    grad_embed = gradnorms.get("embed.weight", 0.0)

    # --- 5. Logits entropy at last position on fixed batch (bits per token) --
    with torch.no_grad():
        probs = F.softmax(logits[:, -1].float(), dim=-1)
        ent = -(probs * (probs + 1e-30).log()).sum(-1).mean().item()

    return {
        "seed": seed, "variant": variant,
        "init_loss": float(init_loss.item()),
        "init_last_pos_entropy_nats": ent,
        "layer_wnorms": layer_wnorms,
        "eager_mag": eager_mag,
        "resid_norms": resid_norms,
        "resid_effrank": resid_effrank,
        "grad_per_layer": grad_per_layer,
        "grad_embed": grad_embed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["fwd", "rev"])
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--K1", type=int, default=4)
    ap.add_argument("--K2", type=int, default=4)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--d-key", type=int, default=32)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tag", default="init_stats")
    args = ap.parse_args()

    results = []
    for s in args.seeds:
        print(f"[init_stats] seed={s} variant={args.variant}", flush=True)
        r = measure_seed(s, args.variant, args.K1, args.K2,
                         args.d_model, args.d_key, args.n_layers,
                         args.batch, args.device)
        results.append(r)

    OUTPUT_DIR.mkdir(exist_ok=True)
    out = OUTPUT_DIR / f"{args.tag}__{args.variant}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"[init_stats] wrote {out.name} with {len(results)} seeds")


if __name__ == "__main__":
    main()
