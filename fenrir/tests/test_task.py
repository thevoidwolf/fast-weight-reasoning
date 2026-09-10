"""Audit-as-code: task shape, determinism, and position layout invariants."""
from __future__ import annotations

import pytest
import torch

from tasks import (
    TaskCfg, sample_long_2hop_injective_truncated, positions, sequence_length,
    PAD, BOS, FACT, SEP, QTOK, ATOK, EOS, N_CONTROL,
)


def _mkgen(seed: int) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(seed)


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------


def test_same_seed_produces_same_batch():
    cfg = TaskCfg()
    a_seq, a_ans, a_br = sample_long_2hop_injective_truncated(
        4, 4, cfg, batch=8, gen=_mkgen(0))
    b_seq, b_ans, b_br = sample_long_2hop_injective_truncated(
        4, 4, cfg, batch=8, gen=_mkgen(0))
    assert torch.equal(a_seq, b_seq)
    assert torch.equal(a_ans, b_ans)
    assert torch.equal(a_br, b_br)


def test_different_seed_produces_different_batch():
    cfg = TaskCfg()
    a_seq, _, _ = sample_long_2hop_injective_truncated(4, 4, cfg, 8, _mkgen(0))
    b_seq, _, _ = sample_long_2hop_injective_truncated(4, 4, cfg, 8, _mkgen(1))
    assert not torch.equal(a_seq, b_seq)


# --------------------------------------------------------------------------
# Shape and length
# --------------------------------------------------------------------------


@pytest.mark.parametrize("K1,K2", [(1, 4), (2, 4), (3, 4), (4, 4), (1, 8), (4, 8)])
def test_sequence_length_matches_helper(K1, K2):
    cfg = TaskCfg()
    seq, _, _ = sample_long_2hop_injective_truncated(K1, K2, cfg, 4, _mkgen(0))
    assert seq.shape == (4, sequence_length(K1, K2))


def test_last_position_is_atok():
    cfg = TaskCfg()
    seq, _, _ = sample_long_2hop_injective_truncated(4, 4, cfg, 8, _mkgen(0))
    assert (seq[:, -1] == ATOK).all(), "seq_in must end with ATOK (teacher-forced predict)"


def test_first_position_is_fact():
    cfg = TaskCfg()
    seq, _, _ = sample_long_2hop_injective_truncated(4, 4, cfg, 8, _mkgen(0))
    assert (seq[:, 0] == FACT).all(), "seq_in must start with FACT"


# --------------------------------------------------------------------------
# Position layout invariants
# --------------------------------------------------------------------------


@pytest.mark.parametrize("K1,K2", [(1, 4), (2, 4), (3, 4), (4, 4)])
def test_positions_align_with_actual_sequence(K1, K2):
    """Every position class in positions() must land on the expected token
    type in the sampled sequence."""
    cfg = TaskCfg()
    seq, _, _ = sample_long_2hop_injective_truncated(K1, K2, cfg, 8, _mkgen(0))
    P = positions(K1, K2)

    # SEP positions should all be SEP
    for pos in P["sep"]:
        assert (seq[:, pos] == SEP).all(), f"pos {pos} expected SEP, got {seq[:, pos]}"

    # QTOK / ATOK / q_ent positions
    assert (seq[:, P["qtok"]] == QTOK).all()
    assert (seq[:, P["atok"]] == ATOK).all()

    # k1 and bridge positions are entity tokens; k2 is entity, val is value token
    entity_lo, entity_hi = N_CONTROL, N_CONTROL + cfg.n_entities
    value_lo, value_hi = N_CONTROL + cfg.n_entities, cfg.vocab_size
    for pos in P["k1"]:
        assert ((seq[:, pos] >= entity_lo) & (seq[:, pos] < entity_hi)).all()
    for pos in P["bridge"]:
        assert ((seq[:, pos] >= entity_lo) & (seq[:, pos] < entity_hi)).all()
    for pos in P["k2"]:
        assert ((seq[:, pos] >= entity_lo) & (seq[:, pos] < entity_hi)).all()
    for pos in P["val"]:
        assert ((seq[:, pos] >= value_lo) & (seq[:, pos] < value_hi)).all()


# --------------------------------------------------------------------------
# Injectivity and chain correctness
# --------------------------------------------------------------------------


def test_injective_f_has_no_repeated_hop2_slots():
    """The injective sampler must map K1 hop-1 facts to K1 DISTINCT hop-2
    slots. So the K1 bridge tokens per row must be distinct entities that
    all appear as k2 tokens."""
    cfg = TaskCfg()
    K1, K2 = 4, 4
    seq, _, _ = sample_long_2hop_injective_truncated(K1, K2, cfg, 64, _mkgen(0))
    P = positions(K1, K2)
    for b in range(seq.shape[0]):
        bridges = seq[b, P["bridge"]]
        k2s = seq[b, P["k2"]]
        # All bridges must be distinct
        assert len(set(bridges.tolist())) == K1, \
            f"row {b}: bridge tokens {bridges.tolist()} have duplicates"
        # Every bridge must appear as a k2 token
        for br in bridges.tolist():
            assert br in k2s.tolist(), f"row {b}: bridge {br} not in k2s {k2s.tolist()}"


def test_answer_is_chain_correct():
    """For each sampled row: q -> b -> v. Verify by walking the sequence."""
    cfg = TaskCfg()
    K1, K2 = 4, 4
    seq, answer, bridge = sample_long_2hop_injective_truncated(K1, K2, cfg, 32, _mkgen(0))
    P = positions(K1, K2)
    for b in range(seq.shape[0]):
        q = seq[b, P["q_ent"]].item()
        # find q among k1 tokens
        k1_toks = seq[b, P["k1"]].tolist()
        assert q in k1_toks, f"row {b}: query {q} not in k1 tokens"
        idx = k1_toks.index(q)
        # bridge_dest at that position
        br_from_seq = seq[b, P["bridge"][idx]].item()
        assert br_from_seq == bridge[b].item(), \
            f"row {b}: bridge mismatch: seq says {br_from_seq}, returned {bridge[b].item()}"
        # find bridge in k2 tokens
        k2_toks = seq[b, P["k2"]].tolist()
        assert br_from_seq in k2_toks, f"row {b}: bridge {br_from_seq} not in k2 tokens"
        j = k2_toks.index(br_from_seq)
        # answer is val at that position
        val_from_seq = seq[b, P["val"][j]].item()
        assert val_from_seq == answer[b].item(), \
            f"row {b}: answer mismatch: seq says {val_from_seq}, returned {answer[b].item()}"


def test_answer_matches_chance_at_uniform_guess():
    """Uniform guessing over the K2 value tokens should hit 1/K2 accuracy.
    Sanity check that answers are diverse (not all the same value)."""
    cfg = TaskCfg()
    K1, K2 = 4, 4
    _, answer, _ = sample_long_2hop_injective_truncated(K1, K2, cfg, 512, _mkgen(0))
    # answers should span multiple distinct tokens
    assert len(set(answer.tolist())) >= 32, \
        f"only {len(set(answer.tolist()))} distinct answers in 512 samples: too concentrated"


# --------------------------------------------------------------------------
# Rejects invalid config
# --------------------------------------------------------------------------


def test_k1_greater_than_k2_raises():
    cfg = TaskCfg()
    with pytest.raises(ValueError):
        sample_long_2hop_injective_truncated(5, 4, cfg, 8, _mkgen(0))
