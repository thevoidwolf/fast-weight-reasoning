"""D2 — Planted-write ceiling test (Fable's diagnostic #2).

Build M directly from IDEAL writes on a synthetic 3-hop sequence, run
the exact chained-read math from rig_003_freki/model.py (single-M
additive write + K_chain-step chained Mᵀ read with RMSNorm between
steps), measure argmax accuracy against candidate value tokens.

This measures the ARCHITECTURAL CEILING of FREKI-K under perfect
learning. If accuracy < 0.99 at d_key=32, cleanup (or a d_key bump) is
mandatory even with perfect optimization. If accuracy ≥ 0.99, cleanup
is optional and pure aux might suffice.

Comparison across d_key ∈ {32, 64} and with vs. without "noise" writes
at control positions (SEP, FACT, QTOK, ATOK).
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


def rmsnorm(x, eps=1e-5):
    rms = x.pow(2).mean(-1, keepdim=True).add(eps).sqrt()
    return x / rms


def build_planted_M(codes_pair_first, codes_pair_second, betas, d):
    """M = sum over pairs of β_i · k_i · v_iᵀ.
    codes_pair_first[i]:  k_i (the "previous" position's code) [d]
    codes_pair_second[i]: v_i (the "current" position's code)  [d]
    betas[i]: scalar write strength.
    Returns M: [d, d]."""
    M = torch.zeros(d, d)
    for k, v, b in zip(codes_pair_first, codes_pair_second, betas):
        M = M + b * torch.outer(k, v)
    return M


def chained_read(M, q, K_chain: int):
    """Ideal read: y = RMSNorm(Mᵀ · RMSNorm(Mᵀ · ... · RMSNorm(Mᵀ · q)))"""
    y = q
    for _ in range(K_chain):
        y = M.T @ y
        y = rmsnorm(y)
    return y


def run_planted(K_hops: int, K_bank: int, d_key: int, K_chain: int,
                n_trials: int = 500, noise_writes: bool = False,
                noise_ratio: float = 2.0, beta_val: float = 1.0,
                device: str = "cpu") -> dict:
    """
    K_hops: number of chain hops (2 or 3)
    K_bank: number of facts per bank
    d_key: code dimension
    K_chain: how many M applications in read
    noise_writes: if True, add noise_ratio*K_bank extra rank-1 writes
                  with random unit-vector k and v (simulating writes at
                  control positions like SEP/FACT via conv history)
    """
    correct = 0
    for _ in range(n_trials):
        # Sample entities & values as random unit vectors in R^d.
        # For a K_hops task: total tokens per query = K_bank^K_hops (nope — actually
        # each hop has K_bank distinct facts).
        # Fact banks: bank 0 = (e_i, b_i) for i in [K_bank]
        #             bank 1 = (b_i, c_{f1(i)}) for i in [K_bank]     if K_hops == 3
        #             bank 2 = (c_l, v_{f2(l)}) for l in [K_bank]     if K_hops == 3
        #             (bank 1 = (b_j, v_{f1(j)}) if K_hops == 2)
        # We pick random unit-vector codes for every token.
        n_ents = 3 * K_bank if K_hops == 3 else 2 * K_bank
        # Layout: first K_bank codes = e_i, next K_bank = b_i, next K_bank = c_l (if K=3)
        # + K_bank codes for the values
        ent_codes = F.normalize(torch.randn(n_ents, d_key), dim=-1)
        val_codes = F.normalize(torch.randn(K_bank, d_key), dim=-1)
        e_codes = ent_codes[:K_bank]
        b_codes = ent_codes[K_bank:2*K_bank]
        if K_hops == 3:
            c_codes = ent_codes[2*K_bank:3*K_bank]

        # Injective permutations
        f1 = torch.randperm(K_bank)   # bank 1: b_j → c_{f1(j)}  (or v_{f1(j)} if K=2)
        if K_hops == 3:
            f2 = torch.randperm(K_bank)  # bank 2: c_l → v_{f2(l)}

        # Build M with ideal writes at pair-second positions.
        # Bank 0 (e→b): writes k=e_i, v=b_i
        # Bank 1 (b→c or v):
        # Bank 2 (c→v): only if K_hops=3
        keys, vals, betas = [], [], []
        for i in range(K_bank):
            keys.append(e_codes[i]); vals.append(b_codes[i]); betas.append(beta_val)
        if K_hops == 2:
            for j in range(K_bank):
                keys.append(b_codes[j]); vals.append(val_codes[f1[j]]); betas.append(beta_val)
        else:  # K_hops == 3
            for j in range(K_bank):
                keys.append(b_codes[j]); vals.append(c_codes[f1[j]]); betas.append(beta_val)
            for l in range(K_bank):
                keys.append(c_codes[l]); vals.append(val_codes[f2[l]]); betas.append(beta_val)

        # Optional noise writes (control-position writes, random content)
        if noise_writes:
            n_noise = int(noise_ratio * K_bank)
            noise_k = F.normalize(torch.randn(n_noise, d_key), dim=-1)
            noise_v = F.normalize(torch.randn(n_noise, d_key), dim=-1)
            for i in range(n_noise):
                keys.append(noise_k[i]); vals.append(noise_v[i]); betas.append(beta_val)

        M = build_planted_M(keys, vals, betas, d_key)

        # Query: pick a random target index; query is q = e_target.
        target_idx = torch.randint(K_bank, (1,)).item()
        q = e_codes[target_idx]

        # Ideal answer:
        if K_hops == 2:
            answer_val_idx = f1[target_idx].item()
        else:
            answer_val_idx = f2[f1[target_idx]].item()

        # Run FREKI chained read
        y = chained_read(M, q, K_chain=K_chain)

        # Argmax over the K_bank candidate value codes
        scores = val_codes @ y                # [K_bank]
        pred_idx = int(scores.argmax().item())
        if pred_idx == answer_val_idx:
            correct += 1

    return {"accuracy": correct / n_trials, "n_trials": n_trials}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K-bank", type=int, default=4)
    ap.add_argument("--n-trials", type=int, default=2000)
    ap.add_argument("--tag", default="d2_planted_write")
    args = ap.parse_args()

    out_dir = HERE / "outputs"
    out_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("D2 planted-write ceiling test")
    print("=" * 60)

    torch.manual_seed(0)
    scenarios = []
    for K_hops in [2, 3]:
        K_chain = K_hops
        for d_key in [32, 64, 128]:
            for noise in [False, True]:
                print(f"\n[scenario] K_hops={K_hops}  K_chain={K_chain}  "
                      f"d_key={d_key}  noise={noise}")
                res = run_planted(
                    K_hops=K_hops, K_bank=args.K_bank, d_key=d_key,
                    K_chain=K_chain, n_trials=args.n_trials,
                    noise_writes=noise,
                )
                print(f"  → accuracy = {res['accuracy']:.4f}  (chance = {1.0/args.K_bank:.3f})")
                scenarios.append({
                    "K_hops": K_hops, "K_chain": K_chain, "d_key": d_key,
                    "noise": noise, **res,
                })

    print("\n" + "=" * 60)
    print("D2 VERDICT")
    print("=" * 60)
    # Grab the K_hops=3 rows without noise
    hop3_no_noise = [s for s in scenarios if s["K_hops"] == 3 and not s["noise"]]
    hop3_noise    = [s for s in scenarios if s["K_hops"] == 3 and s["noise"]]

    print("\n3-hop / K_chain=3 ceiling (no noise, ideal writes only):")
    for s in hop3_no_noise:
        print(f"  d_key={s['d_key']:>3d}: accuracy = {s['accuracy']:.4f}")
    print("\n3-hop / K_chain=3 ceiling with noise writes:")
    for s in hop3_noise:
        print(f"  d_key={s['d_key']:>3d}: accuracy = {s['accuracy']:.4f}")

    ceiling_32 = next(s["accuracy"] for s in hop3_no_noise if s["d_key"] == 32)
    if ceiling_32 < 0.90:
        print(f"\n→ Ceiling at d=32 = {ceiling_32:.3f} < 0.90. Cleanup or d_key")
        print(f"  bump is MANDATORY even under perfect optimization.")
    elif ceiling_32 < 0.99:
        print(f"\n→ Ceiling at d=32 = {ceiling_32:.3f} in [0.90, 0.99). Cleanup or")
        print(f"  d_key=64 recommended to hit pass bar reliably.")
    else:
        print(f"\n→ Ceiling at d=32 = {ceiling_32:.3f}. Expressivity is fine. If")
        print(f"  training gets close to ideal writes, the pass bar is reachable")
        print(f"  without cleanup.")

    result_path = out_dir / f"{args.tag}.json"
    result_path.write_text(json.dumps({
        "K_bank": args.K_bank, "n_trials": args.n_trials,
        "scenarios": scenarios,
    }, indent=2))
    print(f"\nResults written to {result_path}")


if __name__ == "__main__":
    main()
