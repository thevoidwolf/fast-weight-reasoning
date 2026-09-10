"""Identity-homotopy variant of the 3-hop sampler for Arm 3.

At identity mode (ρ = 0), banks 2 and 3 use identity maps with SHARED pools —
the tokens that would be b, c, and v are all drawn from the same entity pool
and paired with themselves.

  Bank 1: (e_i, b_i)  — normal, disjoint entity pool
  Bank 2: (b_j, b_j)  — identity: same token as key AND value
  Bank 3: (b_l, b_l)  — identity: same token as key AND value
                        (reuses bank-1's b pool, so chain 1→2→3 lands on itself)

Answer for e_target: hop 1 gives b_target. Hops 2 and 3 are identity maps
lookup-friendly to b_target itself, so the chain still lands on b_target.
Target = b_target (bank-1 entity range), NOT a value-token — this is a
DEPARTURE from the normal task where answer is always a v-token.

At real mode (ρ = 1), calls the standard `sample_long_3hop_injective_truncated`.

The training loop picks per-batch: with probability ρ(step), use real mode;
otherwise identity mode. As ρ anneals 0 → 1, the primitive learns the full
3-hop composition as a continuous deformation of the (much easier) identity
chain.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import torch

from common.tasks import (
    TaskCfg, PAD, FACT, SEP, QTOK, ATOK, N_CONTROL,
    sample_long_3hop_injective_truncated,
    sample_long_4hop_injective_truncated,
)


def sample_long_3hop_identity_mode(
    K1: int, K2: int, K3: int, cfg: TaskCfg, batch: int,
    gen: torch.Generator, device: str = "cpu",
):
    """Identity-mode: bank 1 is real (e→b); banks 2 and 3 are identity chains
    (b→b, b→b) using the same b-token pool.

    Layout matches `sample_long_3hop_injective_truncated` exactly so the
    model sees identical shapes; the difference is only in the answer.
    """
    n_ents_bank1 = K1 + K2   # e-tokens and b-tokens
    ents = torch.stack([
        torch.randperm(cfg.n_entities, generator=gen)[:n_ents_bank1] + N_CONTROL
        for _ in range(batch)
    ]).to(device)
    key1 = ents[:, :K1]               # e-tokens
    key2 = ents[:, K1:K1+K2]          # b-tokens (used in banks 1-value, 2-key, 2-value, 3-key, 3-value)

    # Bank 1: (e_i, b_{f1(i)}) — random injective mapping
    f1 = torch.stack([torch.randperm(K2, generator=gen)[:K1] for _ in range(batch)]).to(device)
    bridge1_dest = torch.gather(key2, 1, f1)                          # [B, K1]

    # Bank 2 (identity): (b_j, b_j) — same key and value
    # Bank 3 (identity): (b_l, b_l) — same key and value
    # We use the FIRST K3 entries of key2 as the c-position tokens (which equal b tokens
    # by design in identity mode). This ensures shared pool.
    #
    # NOTE: for a chain to succeed, the b-tokens used in banks 2 and 3 must include
    # ALL possible values from bridge1_dest. Since bridge1_dest ⊂ key2, using key2 for
    # banks 2 and 3 works. K2 == K3 (assumed).
    #
    # We construct banks 2 and 3 using the full K2 b-tokens as both keys and values
    # (self-paired). K3 controls the length of bank 3; we set bank3 also = K2 entries.

    # Query
    target_idx = torch.randint(K1, (batch,), generator=gen).to(device)
    q_ent = key1.gather(1, target_idx.unsqueeze(1))
    # Answer = bridge1 = b_target (since banks 2, 3 map b to itself)
    answer = bridge1_dest.gather(1, target_idx.unsqueeze(1)).squeeze(1)

    # Assemble sequence: same shape as real 3-hop task.
    parts = [torch.full((batch, 1), FACT, device=device, dtype=torch.long)]
    # Bank 1
    for i in range(K1):
        parts.append(key1[:, i:i+1])
        parts.append(bridge1_dest[:, i:i+1])
        if i < K1 - 1:
            parts.append(torch.full((batch, 1), SEP, device=device, dtype=torch.long))
    parts.append(torch.full((batch, 1), SEP, device=device, dtype=torch.long))
    # Bank 2 (identity: (b_j, b_j))
    for j in range(K2):
        parts.append(key2[:, j:j+1])
        parts.append(key2[:, j:j+1])
        if j < K2 - 1:
            parts.append(torch.full((batch, 1), SEP, device=device, dtype=torch.long))
    parts.append(torch.full((batch, 1), SEP, device=device, dtype=torch.long))
    # Bank 3 (identity: (b_l, b_l))
    for l in range(K3):
        # Wrap around key2 if K3 > K2 (shouldn't be, but safe).
        l_use = l % K2
        parts.append(key2[:, l_use:l_use+1])
        parts.append(key2[:, l_use:l_use+1])
        if l < K3 - 1:
            parts.append(torch.full((batch, 1), SEP, device=device, dtype=torch.long))

    parts.append(torch.full((batch, 1), QTOK, device=device, dtype=torch.long))
    parts.append(q_ent)
    parts.append(torch.full((batch, 1), ATOK, device=device, dtype=torch.long))

    seq_in = torch.cat(parts, dim=1)
    # bridge1 = bridge2 = answer in identity mode (all = b_target)
    return seq_in, answer, answer, answer


def sample_id_or_real(
    K1: int, K2: int, K3: int, cfg: TaskCfg, batch: int,
    gen: torch.Generator, rho: float, device: str = "cpu",
):
    """With probability `rho`, sample from the real 3-hop task; else identity."""
    # Draw one bernoulli PER BATCH (not per example) — cleaner training signal.
    if torch.rand(1, generator=gen).item() < rho:
        return sample_long_3hop_injective_truncated(
            K1=K1, K2=K2, K3=K3, cfg=cfg, batch=batch, gen=gen, device=device,
        )
    else:
        return sample_long_3hop_identity_mode(
            K1=K1, K2=K2, K3=K3, cfg=cfg, batch=batch, gen=gen, device=device,
        )


def sample_long_4hop_identity_mode(
    K1: int, K2: int, K3: int, K4: int, cfg: TaskCfg, batch: int,
    gen: torch.Generator, device: str = "cpu",
):
    """4-hop identity mode. Bank 1 real (e→b); banks 2, 3, 4 identity
    (b→b, b→b, b→b) sharing the b-token pool. Answer = b_target.

    All three intermediate bridges (bridge1, bridge2, bridge3) collapse
    to the answer in identity mode — training loop treats them as valid
    aux signal for "each y_h decodes to bridge_h at ATOK".
    """
    n_ents_bank1 = K1 + K2
    ents = torch.stack([
        torch.randperm(cfg.n_entities, generator=gen)[:n_ents_bank1] + N_CONTROL
        for _ in range(batch)
    ]).to(device)
    key1 = ents[:, :K1]
    key2 = ents[:, K1:K1+K2]

    f1 = torch.stack([torch.randperm(K2, generator=gen)[:K1] for _ in range(batch)]).to(device)
    bridge1_dest = torch.gather(key2, 1, f1)

    target_idx = torch.randint(K1, (batch,), generator=gen).to(device)
    q_ent = key1.gather(1, target_idx.unsqueeze(1))
    answer = bridge1_dest.gather(1, target_idx.unsqueeze(1)).squeeze(1)

    parts = [torch.full((batch, 1), FACT, device=device, dtype=torch.long)]
    # Bank 1
    for i in range(K1):
        parts.append(key1[:, i:i+1])
        parts.append(bridge1_dest[:, i:i+1])
        if i < K1 - 1:
            parts.append(torch.full((batch, 1), SEP, device=device, dtype=torch.long))
    parts.append(torch.full((batch, 1), SEP, device=device, dtype=torch.long))
    # Banks 2, 3, 4 — all identity, all use key2 pool
    for K_bank in (K2, K3, K4):
        for j in range(K_bank):
            j_use = j % K2
            parts.append(key2[:, j_use:j_use+1])
            parts.append(key2[:, j_use:j_use+1])
            if j < K_bank - 1:
                parts.append(torch.full((batch, 1), SEP, device=device, dtype=torch.long))
        parts.append(torch.full((batch, 1), SEP, device=device, dtype=torch.long))
    # Note: this leaves a trailing SEP after bank 4; the real 4-hop task
    # sampler has bank 4 followed by QTOK with no trailing SEP. Align exactly.
    parts.pop()   # drop the trailing SEP after bank 4

    parts.append(torch.full((batch, 1), QTOK, device=device, dtype=torch.long))
    parts.append(q_ent)
    parts.append(torch.full((batch, 1), ATOK, device=device, dtype=torch.long))

    seq_in = torch.cat(parts, dim=1)
    # bridges 1, 2, 3 all == answer in identity mode
    return seq_in, answer, answer, answer, answer


def sample_id_or_real_4hop(
    K1: int, K2: int, K3: int, K4: int, cfg: TaskCfg, batch: int,
    gen: torch.Generator, rho: float, device: str = "cpu",
):
    """4-hop analog of sample_id_or_real."""
    if torch.rand(1, generator=gen).item() < rho:
        return sample_long_4hop_injective_truncated(
            K1=K1, K2=K2, K3=K3, K4=K4, cfg=cfg, batch=batch,
            gen=gen, device=device,
        )
    else:
        return sample_long_4hop_identity_mode(
            K1=K1, K2=K2, K3=K3, K4=K4, cfg=cfg, batch=batch,
            gen=gen, device=device,
        )
