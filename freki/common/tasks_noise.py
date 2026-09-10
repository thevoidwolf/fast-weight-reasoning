"""Facts-embedded-in-noise + cue-ladder samplers.

Two families:

1. `sample_3hop_facts_in_noise`: noise tokens drawn from the ENTIRE
   entity+value vocab, so noise routinely collides with in-context
   chain entities. Under the delta write rule, a collided noise-key
   ERASES the stored fact (v − Mᵀk overwrite). Per Fable, this makes
   the task near-adversarial-by-construction and is not representative
   of LM data.

2. `sample_3hop_facts_in_filler_vocab`: "cue-ladder (a)" per Fable.
   Noise drawn from a DISJOINT filler vocabulary that never overlaps
   with in-context entities or values. Tests whether β can learn
   content-based write-selectivity when the filler is structurally
   distinguishable. This is the LM-plausible variant.

Both keep the 3-hop chain structure (100% supervision at ATOK) so we
avoid the LM-mix pitfall of filler rows contributing zero gradient.
"""
from __future__ import annotations

import torch

try:
    from .tasks import TaskCfg, N_CONTROL, FACT, SEP, QTOK, ATOK
except ImportError:
    from common.tasks import TaskCfg, N_CONTROL, FACT, SEP, QTOK, ATOK


def sample_3hop_facts_in_noise(
    K1: int, K2: int, K3: int, cfg: TaskCfg, batch: int,
    gen: torch.Generator, n_noise: int = 8, device: str = "cpu",
    exclude_in_context: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """3-hop chain with `n_noise` random-token filler between fact pairs.

    Layout: FACT (k b [noise]*n)*K1 (k b [noise]*n)*K2 (k v [noise]*n)*K3 QTOK q ATOK

    Bank-boundary SEPs preserved. Noise tokens are drawn uniform from
    the whole entity+value vocab. When `exclude_in_context=True` (default),
    noise is resampled per row to exclude the in-context entities/values
    for that row — prevents the "delta rule erases a fact" pathology
    that made the original noise task near-adversarial (Fable 2026-08-14).

    Bridges/answer are identical to the reference 3-hop sampler.
    """
    if K1 > K2 or K2 > K3:
        raise ValueError(f"3-hop requires K1<=K2<=K3, got K1={K1} K2={K2} K3={K3}")

    n_ents_needed = K1 + K2 + K3
    ents = torch.stack([
        torch.randperm(cfg.n_entities, generator=gen)[:n_ents_needed] + N_CONTROL
        for _ in range(batch)
    ]).to(device)
    vals = torch.randint(cfg.n_values, (batch, K3), generator=gen).to(device) \
        + (N_CONTROL + cfg.n_entities)

    key1 = ents[:, :K1]
    key2 = ents[:, K1:K1 + K2]
    key3 = ents[:, K1 + K2:K1 + K2 + K3]

    f1 = torch.stack([torch.randperm(K2, generator=gen)[:K1]
                      for _ in range(batch)]).to(device)
    f2 = torch.stack([torch.randperm(K3, generator=gen)[:K2]
                      for _ in range(batch)]).to(device)
    bridge1_dest = torch.gather(key2, 1, f1)
    bridge2_dest = torch.gather(key3, 1, f2)

    target_idx = torch.randint(K1, (batch,), generator=gen).to(device)
    f1_of_target = f1.gather(1, target_idx.unsqueeze(1))
    f2_of_f1_of_target = f2.gather(1, f1_of_target)
    answer = vals.gather(1, f2_of_f1_of_target).squeeze(1)
    q_ent = key1.gather(1, target_idx.unsqueeze(1))

    forbidden = None
    if exclude_in_context:
        forbidden = torch.cat([ents, vals, q_ent], dim=1)               # [B, K1+K2+K3+1]

    def _noise_block():
        if n_noise == 0:
            return None
        if not exclude_in_context:
            return torch.randint(N_CONTROL, cfg.vocab_size,
                                 (batch, n_noise), generator=gen).to(device)
        K = n_noise * 4
        cand = torch.randint(N_CONTROL, cfg.vocab_size,
                             (batch, K), generator=gen).to(device)     # [B, K]
        is_bad = (cand.unsqueeze(-1) == forbidden.unsqueeze(1)).any(dim=-1)   # [B, K]
        good = (~is_bad).int()
        order = torch.argsort(1 - good, dim=1, stable=True)             # good first, order preserved
        idx = order[:, :n_noise]                                         # [B, n_noise]
        return cand.gather(1, idx)

    parts = [torch.full((batch, 1), FACT, device=device, dtype=torch.long)]

    for i in range(K1):
        parts.append(key1[:, i:i + 1])
        parts.append(bridge1_dest[:, i:i + 1])
        if n_noise > 0:
            parts.append(_noise_block())
    parts.append(torch.full((batch, 1), SEP, device=device, dtype=torch.long))

    for j in range(K2):
        parts.append(key2[:, j:j + 1])
        parts.append(bridge2_dest[:, j:j + 1])
        if n_noise > 0:
            parts.append(_noise_block())
    parts.append(torch.full((batch, 1), SEP, device=device, dtype=torch.long))

    for l in range(K3):
        parts.append(key3[:, l:l + 1])
        parts.append(vals[:, l:l + 1])
        if l < K3 - 1 and n_noise > 0:
            parts.append(_noise_block())

    parts.append(torch.full((batch, 1), QTOK, device=device, dtype=torch.long))
    parts.append(q_ent)
    parts.append(torch.full((batch, 1), ATOK, device=device, dtype=torch.long))

    seq_in = torch.cat(parts, dim=1)
    bridge1 = bridge1_dest.gather(1, target_idx.unsqueeze(1)).squeeze(1)
    bridge2 = bridge2_dest.gather(1, f1_of_target).squeeze(1)
    return seq_in, answer, bridge1, bridge2


def sample_3hop_facts_in_filler_vocab(
    K1: int, K2: int, K3: int, cfg: TaskCfg, batch: int,
    gen: torch.Generator, n_noise: int = 8, filler_count: int = 32,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cue-ladder (a): noise drawn from a disjoint filler-only vocab.

    Reserves the FIRST `filler_count` slots of the value pool as
    "filler tokens" that are never used as chain values. Chain values
    are drawn from the remaining `n_values - filler_count` slots.
    Between fact pairs, `n_noise` random filler tokens are inserted.

    Tests whether β can learn content-based write-selectivity when
    filler is structurally distinguishable (specific token range).
    """
    if K1 > K2 or K2 > K3:
        raise ValueError(f"3-hop requires K1<=K2<=K3, got K1={K1} K2={K2} K3={K3}")
    if cfg.n_values <= filler_count:
        raise ValueError(f"n_values ({cfg.n_values}) must exceed filler_count ({filler_count})")

    filler_lo = N_CONTROL + cfg.n_entities
    filler_hi = filler_lo + filler_count
    value_lo = filler_hi
    n_values_avail = cfg.n_values - filler_count

    n_ents_needed = K1 + K2 + K3
    ents = torch.stack([
        torch.randperm(cfg.n_entities, generator=gen)[:n_ents_needed] + N_CONTROL
        for _ in range(batch)
    ]).to(device)
    vals = torch.randint(n_values_avail, (batch, K3), generator=gen).to(device) + value_lo

    key1 = ents[:, :K1]
    key2 = ents[:, K1:K1 + K2]
    key3 = ents[:, K1 + K2:K1 + K2 + K3]

    f1 = torch.stack([torch.randperm(K2, generator=gen)[:K1]
                      for _ in range(batch)]).to(device)
    f2 = torch.stack([torch.randperm(K3, generator=gen)[:K2]
                      for _ in range(batch)]).to(device)
    bridge1_dest = torch.gather(key2, 1, f1)
    bridge2_dest = torch.gather(key3, 1, f2)

    target_idx = torch.randint(K1, (batch,), generator=gen).to(device)
    f1_of_target = f1.gather(1, target_idx.unsqueeze(1))
    f2_of_f1_of_target = f2.gather(1, f1_of_target)
    answer = vals.gather(1, f2_of_f1_of_target).squeeze(1)
    q_ent = key1.gather(1, target_idx.unsqueeze(1))

    def _noise_block():
        if n_noise == 0:
            return None
        return torch.randint(filler_count, (batch, n_noise), generator=gen).to(device) + filler_lo

    parts = [torch.full((batch, 1), FACT, device=device, dtype=torch.long)]

    for i in range(K1):
        parts.append(key1[:, i:i + 1])
        parts.append(bridge1_dest[:, i:i + 1])
        if n_noise > 0:
            parts.append(_noise_block())
    parts.append(torch.full((batch, 1), SEP, device=device, dtype=torch.long))

    for j in range(K2):
        parts.append(key2[:, j:j + 1])
        parts.append(bridge2_dest[:, j:j + 1])
        if n_noise > 0:
            parts.append(_noise_block())
    parts.append(torch.full((batch, 1), SEP, device=device, dtype=torch.long))

    for l in range(K3):
        parts.append(key3[:, l:l + 1])
        parts.append(vals[:, l:l + 1])
        if l < K3 - 1 and n_noise > 0:
            parts.append(_noise_block())

    parts.append(torch.full((batch, 1), QTOK, device=device, dtype=torch.long))
    parts.append(q_ent)
    parts.append(torch.full((batch, 1), ATOK, device=device, dtype=torch.long))

    seq_in = torch.cat(parts, dim=1)
    bridge1 = bridge1_dest.gather(1, target_idx.unsqueeze(1)).squeeze(1)
    bridge2 = bridge2_dest.gather(1, f1_of_target).squeeze(1)
    return seq_in, answer, bridge1, bridge2


def sample_3hop_facts_with_rel(
    K1: int, K2: int, K3: int, cfg: TaskCfg, batch: int,
    gen: torch.Generator, n_noise: int = 8,
    n_rel_per_bank: int = 1,
    noise_source: str = "filler_vocab",
    filler_count: int = 32,
    rel_count: int = 3,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cue-ladder (b): each fact is a `(k, REL_bank, v)` triple with an
    explicit relation token; cue-ladder (c) via `n_rel_per_bank > 1`
    (paraphrased relations, each fact picks one uniformly).

    Value-pool layout (offset from value_lo = N_CONTROL + n_entities):
      [0, filler_count)                        — filler tokens
      [filler_count, filler_count + rel_count) — relation tokens
      [filler_count + rel_count, n_values)      — chain values

    For 3-hop with n_rel_per_bank=1: 3 REL slots (one per bank).
    For n_rel_per_bank=3: 9 REL slots (three per bank, paraphrased).

    `noise_source`:
      'filler_vocab' — noise drawn from disjoint filler pool (safe).
      'value_vocab'  — noise drawn from the full value pool
                        (INCLUDING chain values, RELs, filler). Makes
                        the REL cue essential — Fable's cue-ladder (b)
                        proper: filler overlaps content.

    Returns (seq, answer, bridge1, bridge2) — bridges unchanged.
    """
    if K1 > K2 or K2 > K3:
        raise ValueError(f"3-hop requires K1<=K2<=K3, got K1={K1} K2={K2} K3={K3}")

    filler_lo = N_CONTROL + cfg.n_entities
    filler_hi = filler_lo + filler_count
    rel_lo = filler_hi
    total_rel_slots = 3 * n_rel_per_bank
    rel_hi = rel_lo + total_rel_slots
    value_lo = rel_hi
    n_values_avail = cfg.n_values - filler_count - total_rel_slots
    if n_values_avail <= 0:
        raise ValueError(
            f"n_values={cfg.n_values} too small for filler_count={filler_count} + "
            f"rel_slots={total_rel_slots}. Need n_values > filler_count + 3*n_rel_per_bank."
        )

    n_ents_needed = K1 + K2 + K3
    ents = torch.stack([
        torch.randperm(cfg.n_entities, generator=gen)[:n_ents_needed] + N_CONTROL
        for _ in range(batch)
    ]).to(device)
    vals = torch.randint(n_values_avail, (batch, K3), generator=gen).to(device) + value_lo

    key1 = ents[:, :K1]
    key2 = ents[:, K1:K1 + K2]
    key3 = ents[:, K1 + K2:K1 + K2 + K3]

    f1 = torch.stack([torch.randperm(K2, generator=gen)[:K1]
                      for _ in range(batch)]).to(device)
    f2 = torch.stack([torch.randperm(K3, generator=gen)[:K2]
                      for _ in range(batch)]).to(device)
    bridge1_dest = torch.gather(key2, 1, f1)
    bridge2_dest = torch.gather(key3, 1, f2)

    target_idx = torch.randint(K1, (batch,), generator=gen).to(device)
    f1_of_target = f1.gather(1, target_idx.unsqueeze(1))
    f2_of_f1_of_target = f2.gather(1, f1_of_target)
    answer = vals.gather(1, f2_of_f1_of_target).squeeze(1)
    q_ent = key1.gather(1, target_idx.unsqueeze(1))

    def _rel_for_bank(bank_idx: int, count: int):
        """Sample REL tokens for each fact in a bank.
        bank_idx in {0,1,2}. Returns [B, count] of REL token ids."""
        bank_start = rel_lo + bank_idx * n_rel_per_bank
        # Random choice among n_rel_per_bank variants (per fact per row)
        picks = torch.randint(n_rel_per_bank, (batch, count), generator=gen).to(device)
        return picks + bank_start

    def _noise_block():
        if n_noise == 0:
            return None
        if noise_source == "filler_vocab":
            return torch.randint(filler_count, (batch, n_noise), generator=gen).to(device) + filler_lo
        elif noise_source == "value_vocab":
            # Full value pool including chain values, RELs, filler
            return torch.randint(cfg.n_values, (batch, n_noise), generator=gen).to(device) + filler_lo
        else:
            raise ValueError(f"unknown noise_source={noise_source!r}")

    parts = [torch.full((batch, 1), FACT, device=device, dtype=torch.long)]

    rel_b1 = _rel_for_bank(0, K1)                                     # [B, K1]
    for i in range(K1):
        parts.append(key1[:, i:i + 1])
        parts.append(rel_b1[:, i:i + 1])
        parts.append(bridge1_dest[:, i:i + 1])
        if n_noise > 0:
            parts.append(_noise_block())
    parts.append(torch.full((batch, 1), SEP, device=device, dtype=torch.long))

    rel_b2 = _rel_for_bank(1, K2)
    for j in range(K2):
        parts.append(key2[:, j:j + 1])
        parts.append(rel_b2[:, j:j + 1])
        parts.append(bridge2_dest[:, j:j + 1])
        if n_noise > 0:
            parts.append(_noise_block())
    parts.append(torch.full((batch, 1), SEP, device=device, dtype=torch.long))

    rel_b3 = _rel_for_bank(2, K3)
    for l in range(K3):
        parts.append(key3[:, l:l + 1])
        parts.append(rel_b3[:, l:l + 1])
        parts.append(vals[:, l:l + 1])
        if l < K3 - 1 and n_noise > 0:
            parts.append(_noise_block())

    parts.append(torch.full((batch, 1), QTOK, device=device, dtype=torch.long))
    parts.append(q_ent)
    parts.append(torch.full((batch, 1), ATOK, device=device, dtype=torch.long))

    seq_in = torch.cat(parts, dim=1)
    bridge1 = bridge1_dest.gather(1, target_idx.unsqueeze(1)).squeeze(1)
    bridge2 = bridge2_dest.gather(1, f1_of_target).squeeze(1)
    return seq_in, answer, bridge1, bridge2


def sample_3hop_facts_with_rel_identity(
    K1: int, K2: int, K3: int, cfg: TaskCfg, batch: int,
    gen: torch.Generator, n_noise: int = 8,
    n_rel_per_bank: int = 1,
    noise_source: str = "filler_vocab",
    filler_count: int = 32,
    rel_count: int = 3,
    device: str = "cpu",
):
    """Identity-mode variant of the REL sampler."""
    if K1 > K2 or K2 > K3:
        raise ValueError(f"3-hop requires K1<=K2<=K3, got K1={K1} K2={K2} K3={K3}")
    filler_lo = N_CONTROL + cfg.n_entities
    rel_lo = filler_lo + filler_count
    n_ents_bank1 = K1 + K2
    ents = torch.stack([
        torch.randperm(cfg.n_entities, generator=gen)[:n_ents_bank1] + N_CONTROL
        for _ in range(batch)
    ]).to(device)
    key1 = ents[:, :K1]
    key2 = ents[:, K1:K1 + K2]
    f1 = torch.stack([torch.randperm(K2, generator=gen)[:K1]
                      for _ in range(batch)]).to(device)
    bridge1_dest = torch.gather(key2, 1, f1)

    target_idx = torch.randint(K1, (batch,), generator=gen).to(device)
    q_ent = key1.gather(1, target_idx.unsqueeze(1))
    answer = bridge1_dest.gather(1, target_idx.unsqueeze(1)).squeeze(1)

    def _rel_for_bank(bank_idx, count):
        bank_start = rel_lo + bank_idx * n_rel_per_bank
        picks = torch.randint(n_rel_per_bank, (batch, count), generator=gen).to(device)
        return picks + bank_start

    def _noise_block():
        if n_noise == 0:
            return None
        if noise_source == "filler_vocab":
            return torch.randint(filler_count, (batch, n_noise), generator=gen).to(device) + filler_lo
        elif noise_source == "value_vocab":
            return torch.randint(cfg.n_values, (batch, n_noise), generator=gen).to(device) + filler_lo
        else:
            raise ValueError(f"unknown noise_source={noise_source!r}")

    parts = [torch.full((batch, 1), FACT, device=device, dtype=torch.long)]

    rel_b1 = _rel_for_bank(0, K1)
    for i in range(K1):
        parts.append(key1[:, i:i + 1])
        parts.append(rel_b1[:, i:i + 1])
        parts.append(bridge1_dest[:, i:i + 1])
        if n_noise > 0:
            parts.append(_noise_block())
    parts.append(torch.full((batch, 1), SEP, device=device, dtype=torch.long))

    rel_b2 = _rel_for_bank(1, K2)
    for j in range(K2):
        parts.append(key2[:, j:j + 1])
        parts.append(rel_b2[:, j:j + 1])
        parts.append(key2[:, j:j + 1])
        if n_noise > 0:
            parts.append(_noise_block())
    parts.append(torch.full((batch, 1), SEP, device=device, dtype=torch.long))

    idx3 = torch.arange(K3, device=device) % K2
    key3_self = key2[:, idx3]
    rel_b3 = _rel_for_bank(2, K3)
    for l in range(K3):
        parts.append(key3_self[:, l:l + 1])
        parts.append(rel_b3[:, l:l + 1])
        parts.append(key3_self[:, l:l + 1])
        if l < K3 - 1 and n_noise > 0:
            parts.append(_noise_block())

    parts.append(torch.full((batch, 1), QTOK, device=device, dtype=torch.long))
    parts.append(q_ent)
    parts.append(torch.full((batch, 1), ATOK, device=device, dtype=torch.long))

    seq_in = torch.cat(parts, dim=1)
    return seq_in, answer, answer, answer


def sample_3hop_facts_in_filler_vocab_identity(
    K1: int, K2: int, K3: int, cfg: TaskCfg, batch: int,
    gen: torch.Generator, n_noise: int = 8, filler_count: int = 32,
    device: str = "cpu",
):
    """Identity-mode variant of cue-ladder (a)."""
    if K1 > K2 or K2 > K3:
        raise ValueError(f"3-hop requires K1<=K2<=K3, got K1={K1} K2={K2} K3={K3}")
    filler_lo = N_CONTROL + cfg.n_entities
    n_ents_bank1 = K1 + K2
    ents = torch.stack([
        torch.randperm(cfg.n_entities, generator=gen)[:n_ents_bank1] + N_CONTROL
        for _ in range(batch)
    ]).to(device)
    key1 = ents[:, :K1]
    key2 = ents[:, K1:K1 + K2]
    f1 = torch.stack([torch.randperm(K2, generator=gen)[:K1]
                      for _ in range(batch)]).to(device)
    bridge1_dest = torch.gather(key2, 1, f1)

    target_idx = torch.randint(K1, (batch,), generator=gen).to(device)
    q_ent = key1.gather(1, target_idx.unsqueeze(1))
    answer = bridge1_dest.gather(1, target_idx.unsqueeze(1)).squeeze(1)

    def _noise_block():
        if n_noise == 0:
            return None
        return torch.randint(filler_count, (batch, n_noise), generator=gen).to(device) + filler_lo

    parts = [torch.full((batch, 1), FACT, device=device, dtype=torch.long)]
    for i in range(K1):
        parts.append(key1[:, i:i + 1])
        parts.append(bridge1_dest[:, i:i + 1])
        if n_noise > 0:
            parts.append(_noise_block())
    parts.append(torch.full((batch, 1), SEP, device=device, dtype=torch.long))
    for j in range(K2):
        parts.append(key2[:, j:j + 1])
        parts.append(key2[:, j:j + 1])
        if n_noise > 0:
            parts.append(_noise_block())
    parts.append(torch.full((batch, 1), SEP, device=device, dtype=torch.long))
    idx3 = torch.arange(K3, device=device) % K2
    key3_self = key2[:, idx3]
    for l in range(K3):
        parts.append(key3_self[:, l:l + 1])
        parts.append(key3_self[:, l:l + 1])
        if l < K3 - 1 and n_noise > 0:
            parts.append(_noise_block())

    parts.append(torch.full((batch, 1), QTOK, device=device, dtype=torch.long))
    parts.append(q_ent)
    parts.append(torch.full((batch, 1), ATOK, device=device, dtype=torch.long))

    seq_in = torch.cat(parts, dim=1)
    return seq_in, answer, answer, answer


def sample_3hop_facts_in_noise_identity(
    K1: int, K2: int, K3: int, cfg: TaskCfg, batch: int,
    gen: torch.Generator, n_noise: int = 8, device: str = "cpu",
):
    """Identity-mode with matching noise layout (banks 2, 3 self-paired)."""
    if K1 > K2 or K2 > K3:
        raise ValueError(f"3-hop requires K1<=K2<=K3, got K1={K1} K2={K2} K3={K3}")

    n_ents_bank1 = K1 + K2
    ents = torch.stack([
        torch.randperm(cfg.n_entities, generator=gen)[:n_ents_bank1] + N_CONTROL
        for _ in range(batch)
    ]).to(device)
    key1 = ents[:, :K1]
    key2 = ents[:, K1:K1 + K2]
    f1 = torch.stack([torch.randperm(K2, generator=gen)[:K1]
                      for _ in range(batch)]).to(device)
    bridge1_dest = torch.gather(key2, 1, f1)

    target_idx = torch.randint(K1, (batch,), generator=gen).to(device)
    q_ent = key1.gather(1, target_idx.unsqueeze(1))
    answer = bridge1_dest.gather(1, target_idx.unsqueeze(1)).squeeze(1)

    def _noise_block():
        if n_noise == 0:
            return None
        return torch.randint(N_CONTROL, cfg.vocab_size,
                             (batch, n_noise), generator=gen).to(device)

    parts = [torch.full((batch, 1), FACT, device=device, dtype=torch.long)]

    for i in range(K1):
        parts.append(key1[:, i:i + 1])
        parts.append(bridge1_dest[:, i:i + 1])
        if n_noise > 0:
            parts.append(_noise_block())
    parts.append(torch.full((batch, 1), SEP, device=device, dtype=torch.long))

    for j in range(K2):
        parts.append(key2[:, j:j + 1])
        parts.append(key2[:, j:j + 1])
        if n_noise > 0:
            parts.append(_noise_block())
    parts.append(torch.full((batch, 1), SEP, device=device, dtype=torch.long))

    idx3 = torch.arange(K3, device=device) % K2
    key3_self = key2[:, idx3]
    for l in range(K3):
        parts.append(key3_self[:, l:l + 1])
        parts.append(key3_self[:, l:l + 1])
        if l < K3 - 1 and n_noise > 0:
            parts.append(_noise_block())

    parts.append(torch.full((batch, 1), QTOK, device=device, dtype=torch.long))
    parts.append(q_ent)
    parts.append(torch.full((batch, 1), ATOK, device=device, dtype=torch.long))

    seq_in = torch.cat(parts, dim=1)
    return seq_in, answer, answer, answer
