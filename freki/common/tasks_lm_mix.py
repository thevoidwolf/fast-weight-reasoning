"""LM-mix sampler for Fable's proposed formation-under-sparse-supervision test.

Per-row mixture:
  With probability `chain_frac`: standard 3-hop chain (uses
  `sample_long_3hop_injective_truncated`).
  Otherwise: filler row — random tokens of matched length, wrapped in
  the same FACT ... QTOK q ATOK skeleton so the model sees a
  structurally identical input but with no meaningful chain. Filler is
  fully non-informative — the target is a fixed dummy value and we
  return a loss mask so the training loop ignores those rows.

Tests: does the mechanism form when only ~15% of rows carry real
supervision, and does β-gating learn to not-write on filler?

The returned `loss_mask` is [B] bool: True on chain rows, False on
filler. Downstream training loop masks CE contributions to chain rows
only.
"""
from __future__ import annotations

import torch

try:
    from .tasks import (
        TaskCfg, N_CONTROL, FACT, SEP, QTOK, ATOK,
        sample_long_3hop_injective_truncated, sequence_length_3hop,
    )
except ImportError:
    from common.tasks import (
        TaskCfg, N_CONTROL, FACT, SEP, QTOK, ATOK,
        sample_long_3hop_injective_truncated, sequence_length_3hop,
    )


def sample_3hop_lm_mix(
    K1: int, K2: int, K3: int, cfg: TaskCfg, batch: int,
    gen: torch.Generator, chain_frac: float = 0.15,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor]:
    """Return (seq, answer, bridge1, bridge2, loss_mask, is_chain_row).

    Every row uses the standard 3-hop sequence length. Filler rows have
    the same skeleton (FACT ... QTOK q ATOK) but random content between.
    answer/bridge1/bridge2 are meaningful only where `is_chain_row=True`;
    filler rows carry placeholders (0) so downstream indexing is safe.
    """
    seq_len = sequence_length_3hop(K1, K2, K3)
    chain_seq, chain_ans, chain_b1, chain_b2 = \
        sample_long_3hop_injective_truncated(K1, K2, K3, cfg, batch, gen, device=device)
    assert chain_seq.shape == (batch, seq_len), \
        f"unexpected chain seq shape {chain_seq.shape} vs ({batch},{seq_len})"

    is_chain = (torch.rand(batch, generator=gen).to(device) < chain_frac)

    filler_tokens = torch.randint(
        N_CONTROL, cfg.vocab_size, (batch, seq_len), generator=gen).to(device)
    filler_tokens[:, 0] = FACT
    filler_tokens[:, -1] = ATOK
    filler_tokens[:, -3] = QTOK

    seq = torch.where(is_chain.unsqueeze(1), chain_seq, filler_tokens)

    zeros = torch.zeros(batch, dtype=chain_ans.dtype, device=device)
    answer = torch.where(is_chain, chain_ans, zeros)
    bridge1 = torch.where(is_chain, chain_b1, zeros)
    bridge2 = torch.where(is_chain, chain_b2, zeros)

    loss_mask = is_chain
    return seq, answer, bridge1, bridge2, loss_mask, is_chain
