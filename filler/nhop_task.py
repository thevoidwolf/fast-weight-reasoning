"""The multi-hop chain task and the filler operator.

`_sample_Nhop_factorized_truncated_injective` builds an N-hop chain
(bank 1 -> bank 2 -> ... -> bank N -> answer) with the answer truncated off the
input; `insert_filler` splices filler tokens in before the query.

Note on `insert_filler`: its docstring describes the original 2-hop truncated
layout. It is kept byte-for-byte as the code that produced the paper's numbers;
do not "fix" it to the docstring.
"""
from __future__ import annotations

import torch

from .tasks import _sample_pairs, FACT, SEP, QTOK, ATOK, EOS, N_CONTROL  # noqa: F401


def _sample_Nhop_factorized_truncated_injective(K_list, task_cfg, batch, gen, device):
    """N-hop generalisation of the 2-hop factorized truncated sampler.

    K_list = [K_1, ..., K_N]. Bank i holds K_i (source, bridge) pairs; for banks
    1..N-1 the bridge is drawn injectively from bank (i+1)'s KEY pool, and for
    bank N the bridge is a value token. Requires K_i <= K_{i+1}. Entity pools are
    disjoint across banks.

    Layout: FACT (k_1 b_1)*K_1 SEP ... SEP (k_N v)*K_N Q e_target A a EOS
    The sequence is returned truncated just before the answer token; the model
    predicts the answer at the final position.

    Returns (seq_in, target, bridge_target).
    """
    N = len(K_list)
    assert N >= 2, f"N-hop sampler requires N >= 2, got {N}"
    for i in range(N - 1):
        assert K_list[i] <= K_list[i + 1], (
            f"injective ladder requires K[{i}]={K_list[i]} <= "
            f"K[{i+1}]={K_list[i+1]}")

    total_entities = sum(K_list)
    all_ents, all_vals = _sample_pairs(
        task_cfg, batch, max(total_entities, K_list[-1]), gen)
    all_ents = all_ents[:, :total_entities].to(device)
    all_vals = all_vals[:, :K_list[-1]].to(device)

    key_pools = []
    offset = 0
    for K in K_list:
        key_pools.append(all_ents[:, offset:offset + K])
        offset += K

    f_list = []
    for i in range(N - 1):
        f_i = torch.stack([
            torch.randperm(K_list[i + 1], generator=gen)[:K_list[i]]
            for _ in range(batch)
        ]).to(device)
        f_list.append(f_i)

    bridges = []
    for i in range(N - 1):
        bridges.append(torch.gather(key_pools[i + 1], 1, f_list[i]))
    bridges.append(all_vals[:, :K_list[N - 1]])

    target_idx = torch.randint(K_list[0], (batch,), generator=gen).to(device)
    q_ent = key_pools[0].gather(1, target_idx.unsqueeze(1))

    cur_idx = target_idx.unsqueeze(1)
    for i in range(N - 1):
        cur_idx = f_list[i].gather(1, cur_idx)
    a = all_vals.gather(1, cur_idx)

    bridge_target = bridges[0].gather(1, target_idx.unsqueeze(1))

    parts = [torch.full((batch, 1), FACT, device=device, dtype=torch.long)]
    for i in range(N):
        for j in range(K_list[i]):
            parts.append(key_pools[i][:, j:j + 1])
            parts.append(bridges[i][:, j:j + 1])
            if j < K_list[i] - 1:
                parts.append(torch.full((batch, 1), SEP, device=device, dtype=torch.long))
        if i < N - 1:
            parts.append(torch.full((batch, 1), SEP, device=device, dtype=torch.long))
    parts.append(torch.full((batch, 1), QTOK, device=device, dtype=torch.long))
    parts.append(q_ent)
    parts.append(torch.full((batch, 1), ATOK, device=device, dtype=torch.long))
    parts.append(a)
    parts.append(torch.full((batch, 1), EOS, device=device, dtype=torch.long))
    seq = torch.cat(parts, dim=1)
    ans_pos = seq.shape[1] - 2
    return seq[:, :ans_pos], a.squeeze(1), bridge_target.squeeze(1)


def insert_filler(seq_in: torch.Tensor, N: int,
                  filler_token: int = SEP) -> torch.Tensor:
    """Insert N filler positions between the last fact-bank token and QTOK.

    Truncated seq layout is `... k_last v_last QTOK q_ent` with QTOK at
    position L-2 and q_ent at L-1. Filler is placed at index L-2, so the
    resulting seq is `... k_last v_last <filler×N> QTOK q_ent`.
    """
    if N == 0:
        return seq_in
    B, L = seq_in.shape
    device = seq_in.device
    dtype = seq_in.dtype
    filler = torch.full((B, N), filler_token, device=device, dtype=dtype)
    # Split at L-2 so QTOK and q_ent get pushed right.
    return torch.cat([seq_in[:, :L - 2], filler, seq_in[:, L - 2:]], dim=1)
