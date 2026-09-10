"""F4 mid-scan probe: when during a layer's causal scan does the answer
signal become available to the answer position?

Method. For a chosen layer, enable probe_cache and forward-pass one batch
of the task. From the cached (q, k, v, beta) we reconstruct the state
M_t at every timestep using the SAME eager operator the mixer used
(via FenrirMixer._eager). Then at every t we compute:

    y_prime_t = q_ATOK^T . M_t

which is what the answer position's query would see if it read at state
M_t instead of at M_L. A V-way logistic-regression probe on y_prime_t
predicts (a) the bridge token and (b) the answer token, K2-constrained.

Because the reconstruction uses the mixer's own _eager method, the probe
cannot diverge from what the mixer actually computes. This is the
audit-safe alternative to external einsum reconstruction.

Writes one JSON per (checkpoint, layer, K1, K2, tag) into outputs/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from model import FenrirStack, MixerConfig, FenrirMixer
from tasks import (TaskCfg, sample_long_2hop_injective_truncated, positions,
                   sequence_length)

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def load_checkpoint(ckpt_path: Path, device: str) -> tuple[FenrirStack, dict]:
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["cfg"]
    mixer_cfg = MixerConfig(d_model=cfg["d_model"], d_key=cfg["d_key"],
                            d_value=cfg["d_value"])
    model = FenrirStack(
        vocab_size=cfg["vocab_size"], cfg=mixer_cfg, variant=cfg["variant"],
        n_layers=cfg["n_layers"], chunked=False,     # sequential for probes
        use_eager=cfg.get("use_eager", True),        # backward-compat default
    ).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, cfg


def reconstruct_M_series(mixer: FenrirMixer, q: torch.Tensor, k: torch.Tensor,
                         v: torch.Tensor, beta: torch.Tensor,
                         atok_pos: int, eager_knockout: bool = False):
    """Given cached (q, k, v, beta) with shapes [B, L, d], externally
    reconstruct M at every timestep using the MIXER'S OWN _eager operator.

    Returns y_prime [B, L, d]:  q_ATOK^T . M_t at each t (after the write at t).
    """
    B, L, d = k.shape
    M = torch.zeros(B, d, d, device=k.device, dtype=k.dtype)
    y_prime = []
    q_atok = q[:, atok_pos]                                    # [B, d]
    for t in range(L):
        eager = mixer._eager(M, k[:, t])                       # [B, d]
        if eager_knockout:
            eager = torch.zeros_like(eager)
        k_eff = k[:, t] + eager
        outer = torch.einsum('bi,bj->bij', k_eff, v[:, t])
        M = M + beta[:, t].view(-1, 1, 1) * outer
        y_prime.append(torch.einsum('bi,bij->bj', q_atok, M))  # [B, d]
    return torch.stack(y_prime, dim=1)                          # [B, L, d]


@torch.no_grad()
def collect(model: FenrirStack, task_cfg: TaskCfg, K1: int, K2: int,
            n_batches: int, batch: int, device: str, seed: int,
            probed_layer: int, eager_knockout: bool) -> dict:
    """Run n_batches of the task through the model, cache the probed
    layer's activations, and reconstruct y_prime at every position for
    every episode."""
    P = positions(K1, K2)
    mixer: FenrirMixer = model.blocks[probed_layer].mixer
    mixer.probe_cache = True

    y_prime_all, bridge_all, answer_all, k2_all, val_all, target_slot_all = \
        [], [], [], [], [], []
    for b in range(n_batches):
        gen = torch.Generator(device="cpu").manual_seed(seed + b)
        seq, answer, bridge = sample_long_2hop_injective_truncated(
            K1, K2, task_cfg, batch, gen, device=device)
        assert seq.shape[1] - 1 == P["atok"], \
            f"layout mismatch: seq_len={seq.shape[1]}, atok={P['atok']}"
        _ = model(seq)                                   # populates mixer._cache

        cache = mixer._cache
        y_prime = reconstruct_M_series(mixer, cache["q"], cache["k"],
                                       cache["v"], cache["beta"],
                                       atok_pos=P["atok"],
                                       eager_knockout=eager_knockout)

        # Slot labels: which of the K2 val positions holds the answer?
        val_toks = seq[:, P["val"]]                       # [B, K2]
        target_slot = (val_toks == answer.unsqueeze(1)).float().argmax(dim=1)

        y_prime_all.append(y_prime.float().cpu())
        bridge_all.append(bridge.cpu())
        answer_all.append(answer.cpu())
        k2_all.append(seq[:, P["k2"]].cpu())
        val_all.append(val_toks.cpu())
        target_slot_all.append(target_slot.cpu())

    mixer.probe_cache = False
    return {
        "y_prime": torch.cat(y_prime_all, dim=0),        # [N, L, d]
        "bridge_tok": torch.cat(bridge_all, dim=0).numpy(),
        "answer_tok": torch.cat(answer_all, dim=0).numpy(),
        "k2_toks": torch.cat(k2_all, dim=0).numpy(),
        "val_toks": torch.cat(val_all, dim=0).numpy(),
        "target_slot": torch.cat(target_slot_all, dim=0).numpy(),
    }


def constrained_lr_accuracy(X_tr, y_tok_tr, X_ev, y_tok_ev,
                            cand_toks_ev, target_slot_ev) -> float:
    """Fit V-way logistic regression on y_tok_tr, then score y_tok_ev with
    argmax restricted to the K2 candidate tokens per episode. Returns
    slot-choice accuracy."""
    if len(np.unique(y_tok_tr)) < 2:
        return float("nan")
    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    clf.fit(X_tr, y_tok_tr)
    scores = clf.decision_function(X_ev)
    class_to_idx = {int(c): i for i, c in enumerate(clf.classes_)}
    N_ev, K2 = cand_toks_ev.shape
    cand_scores = np.full((N_ev, K2), -1e9, dtype=np.float32)
    for j in range(K2):
        for i in range(N_ev):
            idx = class_to_idx.get(int(cand_toks_ev[i, j]))
            if idx is not None:
                cand_scores[i, j] = scores[i, idx]
    pred_slot = cand_scores.argmax(axis=1)
    return float((pred_slot == target_slot_ev).mean())


def probe_per_timestep(train_feats: dict, eval_feats: dict, K1: int, K2: int) -> list[dict]:
    L = train_feats["y_prime"].shape[1]
    out = []
    for t in range(L):
        X_tr = train_feats["y_prime"][:, t].numpy()
        X_ev = eval_feats["y_prime"][:, t].numpy()
        acc_br = constrained_lr_accuracy(
            X_tr, train_feats["bridge_tok"],
            X_ev, eval_feats["bridge_tok"],
            eval_feats["k2_toks"], eval_feats["target_slot"])
        acc_an = constrained_lr_accuracy(
            X_tr, train_feats["answer_tok"],
            X_ev, eval_feats["answer_tok"],
            eval_feats["val_toks"], eval_feats["target_slot"])
        out.append({"t": t, "bridge_acc": round(acc_br, 4),
                    "answer_acc": round(acc_an, 4)})
    return out


def annotate_positions(K1: int, K2: int) -> dict:
    P = positions(K1, K2)
    labels = {P["qtok"]: "QTOK", P["q_ent"]: "q_ent", P["atok"]: "ATOK"}
    for i, pos in enumerate(P["k1"]):     labels[pos] = f"k1[{i}]"
    for i, pos in enumerate(P["bridge"]): labels[pos] = f"bridge[{i}]"
    for i, pos in enumerate(P["sep"]):    labels[pos] = "SEP"
    for j, pos in enumerate(P["k2"]):     labels[pos] = f"k2[{j}]"
    for j, pos in enumerate(P["val"]):    labels[pos] = f"val[{j}]"
    return labels


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--K1", type=int, required=True)
    ap.add_argument("--K2", type=int, required=True)
    ap.add_argument("--layer", type=int, default=1,
                    help="Which layer's mixer to probe (default L1)")
    ap.add_argument("--n-train", type=int, default=8,
                    help="Number of train batches for the LR probe")
    ap.add_argument("--n-eval", type=int, default=4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed-train", type=int, default=400)
    ap.add_argument("--seed-eval", type=int, default=500)
    ap.add_argument("--eager-knockout", action="store_true",
                    help="Zero the eager term during external M reconstruction "
                         "(causal ablation of the address-perturbation).")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model, cfg = load_checkpoint(args.ckpt, args.device)
    task_cfg = TaskCfg(seed=cfg["seed"])
    variant = cfg["variant"]

    print(f"[f4] {args.ckpt.name} variant={variant} K1={args.K1} K2={args.K2} "
          f"L={args.layer} KO={args.eager_knockout}", flush=True)

    print(f"[f4] collecting train ({args.n_train}x{args.batch}) ...", flush=True)
    train_feats = collect(model, task_cfg, args.K1, args.K2,
                          args.n_train, args.batch, args.device,
                          seed=args.seed_train, probed_layer=args.layer,
                          eager_knockout=args.eager_knockout)
    print(f"[f4] collecting eval ({args.n_eval}x{args.batch}) ...", flush=True)
    eval_feats = collect(model, task_cfg, args.K1, args.K2,
                         args.n_eval, args.batch, args.device,
                         seed=args.seed_eval, probed_layer=args.layer,
                         eager_knockout=args.eager_knockout)

    per_t = probe_per_timestep(train_feats, eval_feats, args.K1, args.K2)
    pos_labels = annotate_positions(args.K1, args.K2)

    print(f"[f4] chance = 1/K2 = {1/args.K2:.3f}", flush=True)
    print(f"  {'t':>3} {'position':>10}   bridge   answer", flush=True)
    for row in per_t:
        lbl = pos_labels.get(row["t"], "")
        print(f"  {row['t']:>3} {lbl:>10}   {row['bridge_acc']:.3f}    {row['answer_acc']:.3f}",
              flush=True)

    payload = {
        "ckpt": args.ckpt.name, "variant": variant,
        "K1": args.K1, "K2": args.K2, "chance": round(1.0 / args.K2, 4),
        "probed_layer": args.layer, "eager_knockout": args.eager_knockout,
        "seq_length": len(per_t),
        "position_labels": {str(p): lbl for p, lbl in pos_labels.items()},
        "per_timestep": per_t,
    }
    OUTPUT_DIR.mkdir(exist_ok=True)
    tag = args.tag or (f"probe_f4__{args.ckpt.stem}__K{args.K1}K{args.K2}"
                       f"_L{args.layer}{'_KO' if args.eager_knockout else ''}")
    out = OUTPUT_DIR / f"{tag}.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"[f4] wrote {out.name}", flush=True)


if __name__ == "__main__":
    main()
