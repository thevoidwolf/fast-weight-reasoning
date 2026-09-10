"""D2b — Vocab-pool-aware planted-write ceiling test.

Fable's key prediction: within-sequence crosstalk depends on the number of
writes in M (~12 facts, 3 banks × 4 keys), NOT on the total vocab pool.
Two random 32-dim codes have the same expected overlap whether the pool
is 128 or 512.

Test protocol:
  1. Pre-generate a POOL of `n_entities_pool` random unit vectors.
  2. Per trial, draw K1+K2+K3 codes uniformly without replacement from
     the pool. Build M from ideal writes at those codes. Run chained
     read (K_chain applications of M). Measure argmax accuracy.
  3. Compare ceilings across pool sizes {128, 512, 2048} at d_key=32
     and d_key=64.

If the ceiling is vocab-flat, Fable's diagnosis is confirmed: the
n_entities=512 training failure is a scaffold/schedule issue, not a
capacity issue.
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


def run_planted_from_pool(
    K_hops: int, K_bank: int, d_key: int, K_chain: int, n_pool: int,
    n_trials: int, noise_writes: bool = True, noise_ratio: float = 2.0,
    beta_val: float = 1.0, gen: torch.Generator | None = None,
):
    gen = gen or torch.Generator().manual_seed(0)
    pool = F.normalize(torch.randn(n_pool, d_key, generator=gen), dim=-1)
    val_pool = F.normalize(torch.randn(n_pool, d_key, generator=gen), dim=-1)

    correct = 0
    for _ in range(n_trials):
        # Draw K_hops * K_bank fresh codes from pool for entities
        n_ents = K_hops * K_bank
        idx = torch.randperm(n_pool, generator=gen)[:n_ents]
        ent_codes = pool[idx]
        # And K_bank codes for the values (from the value pool)
        val_idx = torch.randperm(n_pool, generator=gen)[:K_bank]
        val_codes = val_pool[val_idx]

        e_codes = ent_codes[:K_bank]
        b_codes = ent_codes[K_bank:2*K_bank]
        if K_hops == 3:
            c_codes = ent_codes[2*K_bank:3*K_bank]

        f1 = torch.randperm(K_bank, generator=gen)
        if K_hops == 3:
            f2 = torch.randperm(K_bank, generator=gen)

        # Build M from ideal writes
        M = torch.zeros(d_key, d_key)
        for i in range(K_bank):
            M = M + beta_val * torch.outer(e_codes[i], b_codes[i])
        if K_hops == 2:
            for j in range(K_bank):
                M = M + beta_val * torch.outer(b_codes[j], val_codes[f1[j]])
        else:  # K_hops == 3
            for j in range(K_bank):
                M = M + beta_val * torch.outer(b_codes[j], c_codes[f1[j]])
            for l in range(K_bank):
                M = M + beta_val * torch.outer(c_codes[l], val_codes[f2[l]])

        if noise_writes:
            n_noise = int(noise_ratio * K_bank)
            noise_idx_k = torch.randperm(n_pool, generator=gen)[:n_noise]
            noise_idx_v = torch.randperm(n_pool, generator=gen)[:n_noise]
            for i in range(n_noise):
                M = M + beta_val * torch.outer(pool[noise_idx_k[i]], pool[noise_idx_v[i]])

        target_idx = torch.randint(K_bank, (1,), generator=gen).item()
        q = e_codes[target_idx]

        if K_hops == 2:
            answer_val_idx = f1[target_idx].item()
        else:
            answer_val_idx = f2[f1[target_idx]].item()

        y = q
        for _step in range(K_chain):
            y = M.T @ y
            y = rmsnorm(y)

        scores = val_codes @ y
        pred_idx = int(scores.argmax().item())
        if pred_idx == answer_val_idx:
            correct += 1

    return correct / n_trials


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K-bank", type=int, default=4)
    ap.add_argument("--n-trials", type=int, default=2000)
    ap.add_argument("--tag", default="d2b_pool_ceiling")
    args = ap.parse_args()

    out_dir = HERE / "outputs"
    out_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("D2b: vocab-pool-aware planted-write ceiling test")
    print("=" * 60)

    scenarios = []
    print(f"\n{'K_hops':>6} {'K_ch':>5} {'d_key':>5} {'pool':>5} {'noise':>6} {'ceiling':>8}")
    print("  " + "-" * 50)
    for K_hops, K_chain in [(2, 2), (3, 3), (4, 4)]:
        for d_key in [32, 64]:
            for n_pool in [128, 512, 2048]:
                for noise in [False, True]:
                    gen = torch.Generator().manual_seed(0)
                    acc = run_planted_from_pool(
                        K_hops=K_hops, K_bank=args.K_bank, d_key=d_key,
                        K_chain=K_chain, n_pool=n_pool,
                        n_trials=args.n_trials, noise_writes=noise,
                        gen=gen,
                    )
                    print(f"  {K_hops:>4} {K_chain:>5} {d_key:>5} {n_pool:>5} "
                          f"{str(noise):>6} {acc:>8.4f}")
                    scenarios.append({
                        "K_hops": K_hops, "K_chain": K_chain, "d_key": d_key,
                        "n_pool": n_pool, "noise": noise, "accuracy": acc,
                    })

    print("\n" + "=" * 60)
    print("D2b VERDICT")
    print("=" * 60)
    # Check: does d=32 K=3 (with noise) ceiling change across pool sizes?
    focus = [s for s in scenarios if s["K_hops"] == 3 and s["d_key"] == 32 and s["noise"]]
    print("\nd=32, K_chain=3 hops=3, noise=True across pool sizes:")
    for s in focus:
        print(f"  pool={s['n_pool']:>4}  acc={s['accuracy']:.4f}")

    variations = [s["accuracy"] for s in focus]
    max_var = max(variations) - min(variations)
    if max_var < 0.03:
        print(f"\n→ Ceiling is essentially POOL-FLAT (max variation {max_var:.3f}).")
        print("  Confirms Fable's diagnosis: n_entities scaling failure is")
        print("  scaffold/schedule, NOT capacity. Move to #2 (mastery-gated).")
    else:
        print(f"\n→ Ceiling varies by {max_var:.3f} across pool sizes.")
        print("  Some pool-dependence exists; primitive capacity is not")
        print("  perfectly vocab-flat.")

    result_path = out_dir / f"{args.tag}.json"
    result_path.write_text(json.dumps({
        "K_bank": args.K_bank, "n_trials": args.n_trials,
        "scenarios": scenarios,
    }, indent=2))
    print(f"\nResults written to {result_path}")


if __name__ == "__main__":
    main()
