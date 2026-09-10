"""Task samplers for the 2-hop chain composition benchmark used in the
FENRIR paper.

Task shape (long_2hop_kNkM_injective_truncated):

  FACT e_1 b_1 SEP e_2 b_2 SEP ... SEP e_K1 b_K1     (K1 hop-1 facts)
  SEP c_1 v_1 SEP c_2 v_2 SEP ... SEP c_K2 v_K2       (K2 hop-2 facts)
  QTOK q ATOK                                          (query + answer marker)

where the K1 hop-1 keys {e_i} and K2 hop-2 keys {c_j} are drawn from
DISJOINT entity pools; each b_i is exactly one of the {c_j} (an injective
map f: [K1] -> [K2] enforced by randperm, so no two hop-1 facts share a
hop-2 slot); each v_j is a value token; the query q is one of the {e_i};
the answer is v_{f(target_i)}, i.e. follow the chain e_target -> b_target
= c_f(target) -> v_f(target).

The model is teacher-forced up to and including ATOK; the loss is
cross-entropy at ATOK's output against the answer token.

Injective f (via randperm) matters: with collisions (randint), a confused
model gets accidental credit when two hop-1 targets collide on the same
hop-2 slot. Injective forces the chain to be the only path.

Chance floor is 1/K2 (uniform guess over the K2 candidate value tokens
in the fact bank).

Vocabulary layout:

  0 PAD, 1 BOS, 2 FACT, 3 SEP, 4 QTOK, 5 ATOK, 6 EOS
  7 .. 7+E-1                              : entity tokens (E entities)
  7+E .. 7+E+V-1                          : value tokens (V values)
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

# Control-token ids
PAD, BOS, FACT, SEP, QTOK, ATOK, EOS = 0, 1, 2, 3, 4, 5, 6
N_CONTROL = 7


@dataclass
class TaskCfg:
    """Task configuration shared across all K1/K2 arms."""
    n_entities: int = 128
    n_values: int = 128
    seed: int = 0

    @property
    def vocab_size(self) -> int:
        return N_CONTROL + self.n_entities + self.n_values


def _sample_entity_value_pools(cfg: TaskCfg, batch: int, k: int,
                               gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample k distinct entities per row and k values (with replacement).

    Returns:
        ents: [batch, k] long, values in [N_CONTROL, N_CONTROL + n_entities)
        vals: [batch, k] long, values in [N_CONTROL + n_entities, vocab_size)
    """
    ents = torch.stack([
        torch.randperm(cfg.n_entities, generator=gen)[:k] + N_CONTROL
        for _ in range(batch)
    ])
    vals = torch.randint(cfg.n_values, (batch, k), generator=gen) + (N_CONTROL + cfg.n_entities)
    return ents, vals


def sample_long_2hop_injective_truncated(
    K1: int, K2: int, cfg: TaskCfg, batch: int,
    gen: torch.Generator, device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample a batch of the long_2hop_kNkM_injective_truncated task.

    Args:
        K1: number of hop-1 facts (chain sources).
        K2: number of hop-2 facts (must satisfy K1 <= K2 for injectivity).
        cfg: TaskCfg (vocab sizes).
        batch: batch size.
        gen: CPU torch.Generator (drives determinism).
        device: destination device for the returned tensors.

    Returns:
        seq_in: [batch, L] long, tokens up to and including ATOK
            (teacher-forcing input; model's logits at position L-1 predict the answer).
        answer: [batch] long, the ground-truth answer token.
        bridge: [batch] long, the ground-truth bridge (intermediate) token
            (useful for aux-loss supervision or diagnostic probing; the
            training loop may ignore it).

    Deterministic given (K1, K2, cfg.seed, batch, gen's state at call time).
    """
    if K1 > K2:
        raise ValueError(f"injective sampler requires K1 <= K2, got K1={K1}, K2={K2}")

    n_ents_needed = K1 + K2
    ents, vals = _sample_entity_value_pools(cfg, batch,
                                            max(n_ents_needed, K2), gen)
    ents = ents[:, :n_ents_needed].to(device)
    vals = vals[:, :K2].to(device)

    key1 = ents[:, :K1]           # [B, K1] hop-1 keys (chain sources)
    key2 = ents[:, K1:K1 + K2]    # [B, K2] hop-2 keys (bridges)

    # Injective f: for each row a random permutation of [K2], take first K1.
    # f[i] tells us which hop-2 slot the i-th hop-1 fact's bridge points at.
    f = torch.stack([
        torch.randperm(K2, generator=gen)[:K1] for _ in range(batch)
    ]).to(device)
    bridge_dest = torch.gather(key2, 1, f)                  # [B, K1] = key2[f]

    # Target: query one of the K1 hop-1 keys.
    target_idx = torch.randint(K1, (batch,), generator=gen).to(device)
    f_of_target = f.gather(1, target_idx.unsqueeze(1))       # [B, 1]
    answer = vals.gather(1, f_of_target).squeeze(1)          # [B]
    bridge = bridge_dest.gather(1, target_idx.unsqueeze(1)).squeeze(1)  # [B]
    q_ent = key1.gather(1, target_idx.unsqueeze(1))          # [B, 1]

    # Assemble the sequence. Use a list of columns and cat at the end.
    parts = [torch.full((batch, 1), FACT, device=device, dtype=torch.long)]

    # Hop-1 facts: (key1[i], bridge_dest[i]) with SEP between pairs
    for i in range(K1):
        parts.append(key1[:, i:i + 1])
        parts.append(bridge_dest[:, i:i + 1])
        if i < K1 - 1:
            parts.append(torch.full((batch, 1), SEP, device=device, dtype=torch.long))

    # SEP between hop-1 and hop-2 fact banks
    parts.append(torch.full((batch, 1), SEP, device=device, dtype=torch.long))

    # Hop-2 facts: (key2[j], vals[j]) with SEP between pairs
    for j in range(K2):
        parts.append(key2[:, j:j + 1])
        parts.append(vals[:, j:j + 1])
        if j < K2 - 1:
            parts.append(torch.full((batch, 1), SEP, device=device, dtype=torch.long))

    # Query block
    parts.append(torch.full((batch, 1), QTOK, device=device, dtype=torch.long))
    parts.append(q_ent)
    parts.append(torch.full((batch, 1), ATOK, device=device, dtype=torch.long))

    seq_in = torch.cat(parts, dim=1)     # [B, L], last position is ATOK
    return seq_in, answer, bridge


# --------------------------------------------------------------------------
# Position layout helpers (used by probes to name positions)
# --------------------------------------------------------------------------


def positions(K1: int, K2: int) -> dict:
    """Return a dict mapping position class to sequence indices for the
    long_2hop_kNkM_injective_truncated layout with truncation-at-ATOK.

    Keys:
      k1        : list of K1 positions of the hop-1 keys
      bridge    : list of K1 positions of the hop-1 bridge tokens
      k2        : list of K2 positions of the hop-2 keys
      val       : list of K2 positions of the hop-2 value tokens
      sep       : list of SEP positions (interior hop-1 + hop-1/hop-2 boundary + interior hop-2)
      qtok      : scalar position of QTOK
      q_ent     : scalar position of the query entity
      atok      : scalar position of ATOK (last position of seq_in)
    """
    # Sequence: FACT [k1 b1 SEP]*(K1-1) k1 b1 SEP [k2 v1 SEP]*(K2-1) k2 v2 QTOK q_ent ATOK
    # Position 0: FACT
    # Positions 1..3*K1-1: K1 hop-1 pairs with interior SEPs (3 tokens per pair minus final SEP)
    #   pair i: k1 at (1 + 3i), bridge at (2 + 3i), SEP at (3 + 3i) for i < K1-1
    # Then SEP at (3 * K1) as the bank boundary
    # Then hop-2 pairs: k2 at (1 + 3*K1 + 3j), val at (2 + 3*K1 + 3j),
    #                   SEP at (3 + 3*K1 + 3j) for j < K2-1
    # Then QTOK at (3 * (K1 + K2))
    # Then q_ent at (3 * (K1 + K2) + 1)
    # Then ATOK at (3 * (K1 + K2) + 2)
    return {
        "k1":     [1 + 3 * i for i in range(K1)],
        "bridge": [2 + 3 * i for i in range(K1)],
        "k2":     [1 + 3 * K1 + 3 * j for j in range(K2)],
        "val":    [2 + 3 * K1 + 3 * j for j in range(K2)],
        "sep":    [3 + 3 * i for i in range(K1 - 1)]
                  + [3 * K1]
                  + [3 + 3 * K1 + 3 * j for j in range(K2 - 1)],
        "qtok":   3 * (K1 + K2),
        "q_ent":  3 * (K1 + K2) + 1,
        "atok":   3 * (K1 + K2) + 2,
    }


def sequence_length(K1: int, K2: int) -> int:
    """Length of seq_in (up to and including ATOK) for the given K1, K2."""
    return 3 * (K1 + K2) + 3


# --------------------------------------------------------------------------
# 3-hop task variant (session 2026-08-12 addition)
# --------------------------------------------------------------------------


def sample_long_3hop_injective_truncated(
    K1: int, K2: int, K3: int, cfg: TaskCfg, batch: int,
    gen: torch.Generator, device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """3-hop chain composition. Three fact banks:

      bank 1 : (e_i, b_i)  for i in [K1]   -- chain sources → intermediate bridges
      bank 2 : (b_j, c_j)  for j in [K2]   -- intermediate → final bridges
      bank 3 : (c_l, v_l)  for l in [K3]   -- final bridges → values

    All three key pools are disjoint. Injective maps f1: [K1] → [K2] and
    f2: [K2] → [K3] enforce that each hop lands in exactly one downstream
    slot (no accidental credit from collisions). Query e_target → answer
    v_{f2(f1(target))}.

    Chance floor is 1/K3 (uniform guess over the K3 candidate value tokens).
    Requires K1 <= K2 and K2 <= K3.
    """
    if K1 > K2 or K2 > K3:
        raise ValueError(f"3-hop sampler requires K1 <= K2 <= K3, got K1={K1}, K2={K2}, K3={K3}")

    n_ents_needed = K1 + K2 + K3       # e's, b's, c's all disjoint
    ents = torch.stack([
        torch.randperm(cfg.n_entities, generator=gen)[:n_ents_needed] + N_CONTROL
        for _ in range(batch)
    ]).to(device)
    vals = torch.randint(cfg.n_values, (batch, K3), generator=gen).to(device) + (N_CONTROL + cfg.n_entities)

    key1 = ents[:, :K1]                                      # [B, K1] chain sources
    key2 = ents[:, K1:K1 + K2]                               # [B, K2] intermediate bridges
    key3 = ents[:, K1 + K2:K1 + K2 + K3]                     # [B, K3] final bridges

    # Injective f1: [K1] -> [K2], f2: [K2] -> [K3]
    f1 = torch.stack([torch.randperm(K2, generator=gen)[:K1] for _ in range(batch)]).to(device)  # [B, K1]
    f2 = torch.stack([torch.randperm(K3, generator=gen)[:K2] for _ in range(batch)]).to(device)  # [B, K2]

    # bridge1_dest[b, i] = key2[b, f1[b, i]]  (what bank-1 fact i points at)
    bridge1_dest = torch.gather(key2, 1, f1)                 # [B, K1]
    # bridge2_dest[b, j] = key3[b, f2[b, j]]  (what bank-2 fact j points at)
    bridge2_dest = torch.gather(key3, 1, f2)                 # [B, K2]

    # Choose query
    target_idx = torch.randint(K1, (batch,), generator=gen).to(device)                 # [B]
    f1_of_target = f1.gather(1, target_idx.unsqueeze(1))                                # [B, 1]
    f2_of_f1_of_target = f2.gather(1, f1_of_target)                                     # [B, 1]
    answer = vals.gather(1, f2_of_f1_of_target).squeeze(1)                              # [B]
    q_ent = key1.gather(1, target_idx.unsqueeze(1))                                     # [B, 1]

    # Assemble sequence: FACT [bank1] SEP [bank2] SEP [bank3] QTOK q_ent ATOK
    parts = [torch.full((batch, 1), FACT, device=device, dtype=torch.long)]

    # Bank 1: (key1[i], bridge1_dest[i]) with SEP between pairs
    for i in range(K1):
        parts.append(key1[:, i:i + 1])
        parts.append(bridge1_dest[:, i:i + 1])
        if i < K1 - 1:
            parts.append(torch.full((batch, 1), SEP, device=device, dtype=torch.long))
    parts.append(torch.full((batch, 1), SEP, device=device, dtype=torch.long))    # bank boundary

    # Bank 2: (key2[j], bridge2_dest[j]) with SEP between pairs
    for j in range(K2):
        parts.append(key2[:, j:j + 1])
        parts.append(bridge2_dest[:, j:j + 1])
        if j < K2 - 1:
            parts.append(torch.full((batch, 1), SEP, device=device, dtype=torch.long))
    parts.append(torch.full((batch, 1), SEP, device=device, dtype=torch.long))    # bank boundary

    # Bank 3: (key3[l], vals[l]) with SEP between pairs
    for l in range(K3):
        parts.append(key3[:, l:l + 1])
        parts.append(vals[:, l:l + 1])
        if l < K3 - 1:
            parts.append(torch.full((batch, 1), SEP, device=device, dtype=torch.long))

    # Query block
    parts.append(torch.full((batch, 1), QTOK, device=device, dtype=torch.long))
    parts.append(q_ent)
    parts.append(torch.full((batch, 1), ATOK, device=device, dtype=torch.long))

    seq_in = torch.cat(parts, dim=1)
    # Return (seq, answer, bridge1, bridge2).
    #   bridge1 = the b-token that e_target points to (bank-1 destination)
    #   bridge2 = the c-token that bridge1 points to (bank-2 destination)
    # bridge2 is the intermediate after hop-2, needed by aux-supervision experiments.
    bridge1 = bridge1_dest.gather(1, target_idx.unsqueeze(1)).squeeze(1)
    bridge2 = bridge2_dest.gather(1, f1_of_target).squeeze(1)
    return seq_in, answer, bridge1, bridge2


def sequence_length_3hop(K1: int, K2: int, K3: int) -> int:
    return 3 * (K1 + K2 + K3) + 3


def sample_long_4hop_injective_truncated(
    K1: int, K2: int, K3: int, K4: int, cfg: TaskCfg, batch: int,
    gen: torch.Generator, device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """4-hop chain composition. Four fact banks:

      bank 1 : (e_i,   b_i)  for i in [K1]   -- e → b
      bank 2 : (b_j,   c_j)  for j in [K2]   -- b → c
      bank 3 : (c_l,   d_l)  for l in [K3]   -- c → d
      bank 4 : (d_m,   v_m)  for m in [K4]   -- d → v (final value)

    All four key pools are disjoint entity tokens. Values are drawn from
    the value-token range. Injective maps f1, f2, f3 chain the hops.

    Chance floor is 1/K4. Requires K1 <= K2 <= K3 <= K4.

    Returns (seq, answer, bridge1, bridge2, bridge3) — bridge1, bridge2,
    bridge3 are the intermediate hop-1/2/3 targets (all entity tokens).
    """
    if not (K1 <= K2 <= K3 <= K4):
        raise ValueError(f"4-hop sampler requires K1<=K2<=K3<=K4, got "
                         f"K1={K1}, K2={K2}, K3={K3}, K4={K4}")

    n_ents_needed = K1 + K2 + K3 + K4
    ents = torch.stack([
        torch.randperm(cfg.n_entities, generator=gen)[:n_ents_needed] + N_CONTROL
        for _ in range(batch)
    ]).to(device)
    vals = torch.randint(cfg.n_values, (batch, K4), generator=gen).to(device) + (N_CONTROL + cfg.n_entities)

    key1 = ents[:, :K1]
    key2 = ents[:, K1:K1 + K2]
    key3 = ents[:, K1 + K2:K1 + K2 + K3]
    key4 = ents[:, K1 + K2 + K3:K1 + K2 + K3 + K4]

    f1 = torch.stack([torch.randperm(K2, generator=gen)[:K1] for _ in range(batch)]).to(device)
    f2 = torch.stack([torch.randperm(K3, generator=gen)[:K2] for _ in range(batch)]).to(device)
    f3 = torch.stack([torch.randperm(K4, generator=gen)[:K3] for _ in range(batch)]).to(device)

    bridge1_dest = torch.gather(key2, 1, f1)                 # [B, K1]
    bridge2_dest = torch.gather(key3, 1, f2)                 # [B, K2]
    bridge3_dest = torch.gather(key4, 1, f3)                 # [B, K3]

    target_idx = torch.randint(K1, (batch,), generator=gen).to(device)
    f1_of_t   = f1.gather(1, target_idx.unsqueeze(1))                        # [B, 1]
    f2_of_f1  = f2.gather(1, f1_of_t)                                        # [B, 1]
    f3_of_f2  = f3.gather(1, f2_of_f1)                                       # [B, 1]
    answer    = vals.gather(1, f3_of_f2).squeeze(1)                          # [B]
    q_ent     = key1.gather(1, target_idx.unsqueeze(1))                      # [B, 1]

    parts = [torch.full((batch, 1), FACT, device=device, dtype=torch.long)]

    # Bank 1
    for i in range(K1):
        parts.append(key1[:, i:i + 1])
        parts.append(bridge1_dest[:, i:i + 1])
        if i < K1 - 1:
            parts.append(torch.full((batch, 1), SEP, device=device, dtype=torch.long))
    parts.append(torch.full((batch, 1), SEP, device=device, dtype=torch.long))
    # Bank 2
    for j in range(K2):
        parts.append(key2[:, j:j + 1])
        parts.append(bridge2_dest[:, j:j + 1])
        if j < K2 - 1:
            parts.append(torch.full((batch, 1), SEP, device=device, dtype=torch.long))
    parts.append(torch.full((batch, 1), SEP, device=device, dtype=torch.long))
    # Bank 3
    for l in range(K3):
        parts.append(key3[:, l:l + 1])
        parts.append(bridge3_dest[:, l:l + 1])
        if l < K3 - 1:
            parts.append(torch.full((batch, 1), SEP, device=device, dtype=torch.long))
    parts.append(torch.full((batch, 1), SEP, device=device, dtype=torch.long))
    # Bank 4
    for m in range(K4):
        parts.append(key4[:, m:m + 1])
        parts.append(vals[:, m:m + 1])
        if m < K4 - 1:
            parts.append(torch.full((batch, 1), SEP, device=device, dtype=torch.long))

    parts.append(torch.full((batch, 1), QTOK, device=device, dtype=torch.long))
    parts.append(q_ent)
    parts.append(torch.full((batch, 1), ATOK, device=device, dtype=torch.long))

    seq_in = torch.cat(parts, dim=1)
    bridge1 = bridge1_dest.gather(1, target_idx.unsqueeze(1)).squeeze(1)
    bridge2 = bridge2_dest.gather(1, f1_of_t).squeeze(1)
    bridge3 = bridge3_dest.gather(1, f2_of_f1).squeeze(1)
    return seq_in, answer, bridge1, bridge2, bridge3


def sequence_length_4hop(K1: int, K2: int, K3: int, K4: int) -> int:
    return 3 + 3 * (K1 + K2 + K3 + K4) + 4     # 4 bank-boundary SEPs + query block
