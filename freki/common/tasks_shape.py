"""Task-shape variants for PART3 §49 (structural-prior dependence).

Three arms per the writeup:
  Arm 1 — no bank-boundary SEP (within-pair adjacency and per-bank
          ordering preserved; only the two bank-boundary SEPs removed).
  Arm 2 — fully interleaved: all (key, value) pairs across banks
          concatenated and shuffled per row. Between-pair SEP kept.
  Arm 3 — no SEP anywhere (Arm 2 with within-pair SEPs also stripped).

Bridges/values are the same tokens as in the baseline 3-hop task. The
mechanism must resolve the chain from token identity alone once
structural priors are removed. Answer/bridge tensors are identical to
`common.tasks.sample_long_3hop_injective_truncated` — only the
input-sequence layout differs.

3-hop only for now. Extending to 4-hop is straightforward.
"""
from __future__ import annotations

from typing import Literal
import torch

try:
    from .tasks import TaskCfg, N_CONTROL, FACT, SEP, QTOK, ATOK
except ImportError:
    from common.tasks import TaskCfg, N_CONTROL, FACT, SEP, QTOK, ATOK


ArmMode = Literal["arm1_no_bank_sep", "arm2_interleaved", "arm3_no_sep"]


def _draw_ents_vals(K1: int, K2: int, K3: int, cfg: TaskCfg,
                    batch: int, gen: torch.Generator, device: str):
    n_ents_needed = K1 + K2 + K3
    ents = torch.stack([
        torch.randperm(cfg.n_entities, generator=gen)[:n_ents_needed] + N_CONTROL
        for _ in range(batch)
    ]).to(device)
    vals = torch.randint(cfg.n_values, (batch, K3), generator=gen).to(device) \
        + (N_CONTROL + cfg.n_entities)
    return ents, vals


def sample_3hop_shape(
    arm: ArmMode,
    K1: int, K2: int, K3: int, cfg: TaskCfg, batch: int,
    gen: torch.Generator, device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Task-shape sampler. Returns (seq, answer, bridge1, bridge2).

    Chain construction is identical to the baseline injective task; only
    the fact list layout is changed by `arm`.
    """
    if K1 > K2 or K2 > K3:
        raise ValueError(f"3-hop requires K1<=K2<=K3, got K1={K1} K2={K2} K3={K3}")

    ents, vals = _draw_ents_vals(K1, K2, K3, cfg, batch, gen, device)
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

    pairs_b1 = torch.stack([key1, bridge1_dest], dim=-1)      # [B, K1, 2]
    pairs_b2 = torch.stack([key2, bridge2_dest], dim=-1)      # [B, K2, 2]
    pairs_b3 = torch.stack([key3, vals], dim=-1)              # [B, K3, 2]

    parts = [torch.full((batch, 1), FACT, device=device, dtype=torch.long)]

    if arm == "arm1_no_bank_sep":
        for bank_pairs, K in ((pairs_b1, K1), (pairs_b2, K2), (pairs_b3, K3)):
            for i in range(K):
                parts.append(bank_pairs[:, i, 0:1])
                parts.append(bank_pairs[:, i, 1:2])
                if i < K - 1:
                    parts.append(torch.full((batch, 1), SEP, device=device,
                                            dtype=torch.long))

    elif arm in ("arm2_interleaved", "arm3_no_sep"):
        all_pairs = torch.cat([pairs_b1, pairs_b2, pairs_b3], dim=1)  # [B, K1+K2+K3, 2]
        n_pairs = all_pairs.shape[1]
        perms = torch.stack([torch.randperm(n_pairs, generator=gen)
                             for _ in range(batch)]).to(device)         # [B, n_pairs]
        perms_exp = perms.unsqueeze(-1).expand(-1, -1, 2)                # [B, n_pairs, 2]
        shuffled = all_pairs.gather(1, perms_exp)                        # [B, n_pairs, 2]
        emit_sep = (arm == "arm2_interleaved")
        for i in range(n_pairs):
            parts.append(shuffled[:, i, 0:1])
            parts.append(shuffled[:, i, 1:2])
            if emit_sep and i < n_pairs - 1:
                parts.append(torch.full((batch, 1), SEP, device=device,
                                        dtype=torch.long))
    else:
        raise ValueError(f"unknown arm: {arm!r}")

    parts.append(torch.full((batch, 1), QTOK, device=device, dtype=torch.long))
    parts.append(q_ent)
    parts.append(torch.full((batch, 1), ATOK, device=device, dtype=torch.long))

    seq_in = torch.cat(parts, dim=1)
    bridge1 = bridge1_dest.gather(1, target_idx.unsqueeze(1)).squeeze(1)
    bridge2 = bridge2_dest.gather(1, f1_of_target).squeeze(1)
    return seq_in, answer, bridge1, bridge2


def sample_3hop_shape_identity(
    arm: ArmMode,
    K1: int, K2: int, K3: int, cfg: TaskCfg, batch: int,
    gen: torch.Generator, device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Identity-mode variant of the shape sampler.

    Same layout as `sample_3hop_shape(arm, ...)` — bank 1 remains real
    (e_i → b_{f1(i)}), banks 2 & 3 are self-paired (b_j → b_j). Answer =
    bridge1 = bridge2 = b_target. Used for identity-homotopy warmup so
    the model sees the arm's layout in every batch (no shape leak).
    """
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

    pairs_b1 = torch.stack([key1, bridge1_dest], dim=-1)                     # [B, K1, 2]
    pairs_b2 = torch.stack([key2, key2], dim=-1)                             # [B, K2, 2]
    idx3 = torch.arange(K3, device=device) % K2
    key3_self = key2[:, idx3]                                                 # [B, K3]
    pairs_b3 = torch.stack([key3_self, key3_self], dim=-1)                    # [B, K3, 2]

    parts = [torch.full((batch, 1), FACT, device=device, dtype=torch.long)]

    if arm == "arm1_no_bank_sep":
        for bank_pairs, K in ((pairs_b1, K1), (pairs_b2, K2), (pairs_b3, K3)):
            for i in range(K):
                parts.append(bank_pairs[:, i, 0:1])
                parts.append(bank_pairs[:, i, 1:2])
                if i < K - 1:
                    parts.append(torch.full((batch, 1), SEP, device=device,
                                            dtype=torch.long))
    elif arm in ("arm2_interleaved", "arm3_no_sep"):
        all_pairs = torch.cat([pairs_b1, pairs_b2, pairs_b3], dim=1)
        n_pairs = all_pairs.shape[1]
        perms = torch.stack([torch.randperm(n_pairs, generator=gen)
                             for _ in range(batch)]).to(device)
        perms_exp = perms.unsqueeze(-1).expand(-1, -1, 2)
        shuffled = all_pairs.gather(1, perms_exp)
        emit_sep = (arm == "arm2_interleaved")
        for i in range(n_pairs):
            parts.append(shuffled[:, i, 0:1])
            parts.append(shuffled[:, i, 1:2])
            if emit_sep and i < n_pairs - 1:
                parts.append(torch.full((batch, 1), SEP, device=device,
                                        dtype=torch.long))
    else:
        raise ValueError(f"unknown arm: {arm!r}")

    parts.append(torch.full((batch, 1), QTOK, device=device, dtype=torch.long))
    parts.append(q_ent)
    parts.append(torch.full((batch, 1), ATOK, device=device, dtype=torch.long))

    seq_in = torch.cat(parts, dim=1)
    return seq_in, answer, answer, answer


def sample_3hop_reuse_arm1(
    K1: int, K2: int, K3: int, cfg: TaskCfg, batch: int,
    gen: torch.Generator, device: str = "cpu",
    n_reuse: int = 2,
    reuse_role: str = "cross_bank_key",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """PART3 §50 Arm 1 — controlled cross-hop entity reuse.

    Baseline sequence structure (with bank-boundary SEPs, matches the
    reference 3-hop task exactly). Each row has `n_reuse` tokens that
    appear as keys in *two* different banks, generating cross-bank
    ambiguity that must be resolved by bank position (role).

    Safety constraints — the target's chain never sees a reused key:
      * The target's hop-1 key is not chosen from the reused set.
      * The target's hop-1 destination (bridge1) is not chosen either.
      * The bridge1 target's bank-2 destination (bridge2) also not.
    So the target chain is fully unambiguous; the mechanism only needs
    to disambiguate reuse for the DISTRACTOR facts (still hurts if it
    fires the wrong-bank fact).

    Reuse layout for one shared token t:
      Duplicate t as a KEY in a second bank (b'). Its value at that
      slot is a random unrelated bridge/value drawn from the same
      candidate pool for that bank (so the duplicate fact is
      structurally identical to real facts — same shape).

    Returns (seq, answer, bridge1, bridge2) exactly as the baseline
    sampler.
    """
    if K1 > K2 or K2 > K3:
        raise ValueError(f"3-hop requires K1<=K2<=K3, got K1={K1} K2={K2} K3={K3}")
    if reuse_role != "cross_bank_key":
        raise NotImplementedError(f"reuse_role={reuse_role!r} not implemented")
    n_ents_needed = K1 + K2 + K3
    n_reuse = min(n_reuse, K1 - 1, K2 - 1, K3 - 1)
    n_reuse = max(0, n_reuse)

    ents = torch.stack([
        torch.randperm(cfg.n_entities, generator=gen)[:n_ents_needed] + N_CONTROL
        for _ in range(batch)
    ]).to(device)
    vals = torch.randint(cfg.n_values, (batch, K3), generator=gen).to(device) \
        + (N_CONTROL + cfg.n_entities)

    key1 = ents[:, :K1].clone()
    key2 = ents[:, K1:K1 + K2].clone()
    key3 = ents[:, K1 + K2:K1 + K2 + K3].clone()

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

    for b in range(batch):
        tgt_k1_slot = int(target_idx[b].item())
        tgt_k2_slot = int(f1_of_target[b].item())
        tgt_k3_slot = int(f2_of_f1_of_target[b].item())

        source_slots = []
        for slot in range(K1):
            if slot != tgt_k1_slot:
                source_slots.append(("k1", slot))
        for slot in range(K2):
            if slot != tgt_k2_slot:
                source_slots.append(("k2", slot))
        for slot in range(K3):
            if slot != tgt_k3_slot:
                source_slots.append(("k3", slot))

        perm = torch.randperm(len(source_slots), generator=gen)[:n_reuse]
        picks = [source_slots[int(i)] for i in perm.tolist()]

        for (bank_src, slot_src) in picks:
            if bank_src == "k1":
                token = key1[b, slot_src].item()
            elif bank_src == "k2":
                token = key2[b, slot_src].item()
            else:
                token = key3[b, slot_src].item()

            candidates = []
            if bank_src != "k1":
                for slot in range(K1):
                    if slot != tgt_k1_slot:
                        candidates.append(("k1", slot))
            if bank_src != "k2":
                for slot in range(K2):
                    if slot != tgt_k2_slot:
                        candidates.append(("k2", slot))
            if bank_src != "k3":
                for slot in range(K3):
                    if slot != tgt_k3_slot:
                        candidates.append(("k3", slot))
            pick_idx = int(torch.randint(len(candidates), (1,), generator=gen).item())
            (bank_dst, slot_dst) = candidates[pick_idx]

            if bank_dst == "k1":
                if bridge1_dest[b, slot_dst].item() != token:
                    key1[b, slot_dst] = token
            elif bank_dst == "k2":
                if bridge1_dest[b, tgt_k1_slot].item() != token \
                        and bridge2_dest[b, slot_dst].item() != token:
                    key2[b, slot_dst] = token
            else:
                if bridge2_dest[b, tgt_k2_slot].item() != token \
                        and vals[b, slot_dst].item() != token:
                    key3[b, slot_dst] = token

    parts = [torch.full((batch, 1), FACT, device=device, dtype=torch.long)]
    for i in range(K1):
        parts.append(key1[:, i:i + 1])
        parts.append(bridge1_dest[:, i:i + 1])
        if i < K1 - 1:
            parts.append(torch.full((batch, 1), SEP, device=device, dtype=torch.long))
    parts.append(torch.full((batch, 1), SEP, device=device, dtype=torch.long))
    for j in range(K2):
        parts.append(key2[:, j:j + 1])
        parts.append(bridge2_dest[:, j:j + 1])
        if j < K2 - 1:
            parts.append(torch.full((batch, 1), SEP, device=device, dtype=torch.long))
    parts.append(torch.full((batch, 1), SEP, device=device, dtype=torch.long))
    for l in range(K3):
        parts.append(key3[:, l:l + 1])
        parts.append(vals[:, l:l + 1])
        if l < K3 - 1:
            parts.append(torch.full((batch, 1), SEP, device=device, dtype=torch.long))

    parts.append(torch.full((batch, 1), QTOK, device=device, dtype=torch.long))
    parts.append(q_ent)
    parts.append(torch.full((batch, 1), ATOK, device=device, dtype=torch.long))

    seq_in = torch.cat(parts, dim=1)
    bridge1 = bridge1_dest.gather(1, target_idx.unsqueeze(1)).squeeze(1)
    bridge2 = bridge2_dest.gather(1, f1_of_target).squeeze(1)
    return seq_in, answer, bridge1, bridge2


def sample_3hop_reuse_arm1_identity(
    K1: int, K2: int, K3: int, cfg: TaskCfg, batch: int,
    gen: torch.Generator, device: str = "cpu",
    n_reuse: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Identity-mode warmup with the same layout + reuse pattern as
    `sample_3hop_reuse_arm1`. Bank 2 & 3 are self-paired (b_j → b_j);
    reuse duplicates a small number of keys across banks.

    Uses baseline (bank-structured) layout — identity-homotopy is only
    for warmup, not a shape probe.
    """
    if K1 > K2 or K2 > K3:
        raise ValueError(f"3-hop requires K1<=K2<=K3, got K1={K1} K2={K2} K3={K3}")

    n_ents_bank1 = K1 + K2
    ents = torch.stack([
        torch.randperm(cfg.n_entities, generator=gen)[:n_ents_bank1] + N_CONTROL
        for _ in range(batch)
    ]).to(device)
    key1 = ents[:, :K1].clone()
    key2 = ents[:, K1:K1 + K2].clone()
    f1 = torch.stack([torch.randperm(K2, generator=gen)[:K1]
                      for _ in range(batch)]).to(device)
    bridge1_dest = torch.gather(key2, 1, f1)

    target_idx = torch.randint(K1, (batch,), generator=gen).to(device)
    q_ent = key1.gather(1, target_idx.unsqueeze(1))
    answer = bridge1_dest.gather(1, target_idx.unsqueeze(1)).squeeze(1)

    parts = [torch.full((batch, 1), FACT, device=device, dtype=torch.long)]
    for i in range(K1):
        parts.append(key1[:, i:i + 1])
        parts.append(bridge1_dest[:, i:i + 1])
        if i < K1 - 1:
            parts.append(torch.full((batch, 1), SEP, device=device, dtype=torch.long))
    parts.append(torch.full((batch, 1), SEP, device=device, dtype=torch.long))
    for j in range(K2):
        parts.append(key2[:, j:j + 1])
        parts.append(key2[:, j:j + 1])
        if j < K2 - 1:
            parts.append(torch.full((batch, 1), SEP, device=device, dtype=torch.long))
    parts.append(torch.full((batch, 1), SEP, device=device, dtype=torch.long))
    idx3 = torch.arange(K3, device=device) % K2
    key3_self = key2[:, idx3]
    for l in range(K3):
        parts.append(key3_self[:, l:l + 1])
        parts.append(key3_self[:, l:l + 1])
        if l < K3 - 1:
            parts.append(torch.full((batch, 1), SEP, device=device, dtype=torch.long))

    parts.append(torch.full((batch, 1), QTOK, device=device, dtype=torch.long))
    parts.append(q_ent)
    parts.append(torch.full((batch, 1), ATOK, device=device, dtype=torch.long))

    seq_in = torch.cat(parts, dim=1)
    return seq_in, answer, answer, answer


def sequence_length_3hop_shape(arm: ArmMode, K1: int, K2: int, K3: int) -> int:
    n_pairs = K1 + K2 + K3
    n_tokens = 2 * n_pairs                       # every pair contributes 2 tokens
    if arm == "arm1_no_bank_sep":
        n_seps = (K1 - 1) + (K2 - 1) + (K3 - 1)  # only within-bank SEPs
    elif arm == "arm2_interleaved":
        n_seps = n_pairs - 1
    elif arm == "arm3_no_sep":
        n_seps = 0
    else:
        raise ValueError(f"unknown arm: {arm!r}")
    return 1 + n_tokens + n_seps + 3             # FACT + pairs + SEPs + QTOK q ATOK
