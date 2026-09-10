"""Audit-as-code: within-repo parity between sequential and chunked forms
of the FENRIR rev variant.

Cross-repo parity against alternate reference implementations lives
outside this repo in the publisher's audit tooling.
"""
from __future__ import annotations

import pytest
import torch

from chunked import rev_sequential, rev_chunked


def _make_inputs(B: int, L: int, d: int, dtype: torch.dtype, device: str, seed: int):
    """Random q, k, v, beta at the given precision and device."""
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(B, L, d, dtype=dtype, device=device, generator=g) / d ** 0.5
    k = torch.randn(B, L, d, dtype=dtype, device=device, generator=g) / d ** 0.5
    v = torch.randn(B, L, d, dtype=dtype, device=device, generator=g) / d ** 0.5
    beta = torch.sigmoid(torch.randn(B, L, dtype=dtype, device=device, generator=g))
    return q, k, v, beta


# --------------------------------------------------------------------------
# Forward parity: sequential vs chunked (rev variant)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("chunk_size", [1, 7, 8, 16, 32, 64])
def test_rev_sequential_matches_chunked_fp64(chunk_size):
    """At fp64 the two forms should match to near-machine precision."""
    B, L, d = 2, 48, 8
    q, k, v, beta = _make_inputs(B, L, d, torch.float64, "cpu", seed=0)
    M0 = torch.randn(B, d, d, dtype=torch.float64) / d

    y_seq, M_seq = rev_sequential(q, k, v, beta, M0)
    y_chk, M_chk, _ = rev_chunked(q, k, v, beta, M0, chunk_size=chunk_size)

    y_diff = (y_seq - y_chk).abs().max().item()
    M_diff = (M_seq - M_chk).abs().max().item()
    assert y_diff < 1e-9, f"y diverges: chunk_size={chunk_size} max diff={y_diff:.3e}"
    assert M_diff < 1e-9, f"M diverges: chunk_size={chunk_size} max diff={M_diff:.3e}"


def test_rev_sequential_matches_chunked_fp32():
    """At fp32 the chunked internal precision (fp32) should still match
    the sequential fp32 loop to reasonable tolerance."""
    B, L, d = 2, 32, 8
    q, k, v, beta = _make_inputs(B, L, d, torch.float32, "cpu", seed=0)
    y_seq, _ = rev_sequential(q, k, v, beta)
    y_chk, _, _ = rev_chunked(q, k, v, beta, chunk_size=8)
    diff = (y_seq - y_chk).abs().max().item()
    assert diff < 1e-4, f"fp32 y diff = {diff:.3e}"


def test_ragged_final_chunk_is_handled():
    """chunk_size=7 with L=32 gives 4 chunks of 7 plus a chunk of 4."""
    B, L, d = 1, 32, 8
    q, k, v, beta = _make_inputs(B, L, d, torch.float64, "cpu", seed=1)
    y_seq, _ = rev_sequential(q, k, v, beta)
    y_chk, _, _ = rev_chunked(q, k, v, beta, chunk_size=7)
    assert y_seq.shape == y_chk.shape == (B, L, d)
    assert (y_seq - y_chk).abs().max().item() < 1e-9


# --------------------------------------------------------------------------
# Eager-return parity: chunked's need_eager output must equal the
# per-position M_{t-1} k_t computed sequentially
# --------------------------------------------------------------------------


def test_chunked_eager_output_matches_sequential_computation():
    """When need_eager=True, chunked must return the same per-position
    M_{t-1} k_t values that the sequential mixer stores in
    _cache['eager']. This is the audit hook for mid-scan probes on
    chunked training checkpoints.
    """
    B, L, d = 1, 24, 8
    q, k, v, beta = _make_inputs(B, L, d, torch.float64, "cpu", seed=2)

    # Sequential reference: compute M_{t-1} k_t at every t by hand.
    M = torch.zeros(B, d, d, dtype=torch.float64)
    seq_eagers = []
    for t in range(L):
        Mk = torch.einsum('bij,bj->bi', M, k[:, t])
        seq_eagers.append(Mk.clone())
        w = beta[:, t, None] * (k[:, t] + Mk)
        M = M + torch.einsum('bi,bj->bij', w, v[:, t])
    seq_eager = torch.stack(seq_eagers, dim=1)

    _, _, chk_eager = rev_chunked(q, k, v, beta, chunk_size=8, need_eager=True)
    diff = (seq_eager - chk_eager).abs().max().item()
    assert diff < 1e-9, f"chunked eager diverges from sequential: max diff={diff:.3e}"


# --------------------------------------------------------------------------
# Gradient parity: chunked must give the same grads as sequential
# --------------------------------------------------------------------------


def test_gradient_parity_fp32():
    """The chunked forward's autograd should match the sequential loop's
    to reasonable fp32 tolerance."""
    B, L, d = 2, 24, 8

    def grads_from(fn):
        torch.manual_seed(1)
        qg = (torch.randn(B, L, d) / d ** 0.5).requires_grad_()
        kg = (torch.randn(B, L, d) / d ** 0.5).requires_grad_()
        vg = (torch.randn(B, L, d) / d ** 0.5).requires_grad_()
        bg = torch.randn(B, L).requires_grad_()
        out = fn(qg, kg, vg, torch.sigmoid(bg))
        out.pow(2).mean().backward()
        return [t.grad.clone() for t in (qg, kg, vg, bg)]

    g_seq = grads_from(lambda q_, k_, v_, b_: rev_sequential(q_, k_, v_, b_)[0])
    g_chk = grads_from(lambda q_, k_, v_, b_: rev_chunked(q_, k_, v_, b_, chunk_size=8)[0])
    max_err = max((a - b).abs().max().item() for a, b in zip(g_seq, g_chk))
    assert max_err < 1e-4, f"gradient parity: max err = {max_err:.3e}"


def test_gradient_parity_with_checkpoint():
    """Same test with use_checkpoint=True (recompute intra-chunk in backward)."""
    B, L, d = 2, 24, 8

    def grads_from(use_ckpt: bool):
        torch.manual_seed(2)
        qg = (torch.randn(B, L, d) / d ** 0.5).requires_grad_()
        kg = (torch.randn(B, L, d) / d ** 0.5).requires_grad_()
        vg = (torch.randn(B, L, d) / d ** 0.5).requires_grad_()
        bg = torch.randn(B, L).requires_grad_()
        out = rev_chunked(qg, kg, vg, torch.sigmoid(bg),
                          chunk_size=8, use_checkpoint=use_ckpt)[0]
        out.pow(2).mean().backward()
        return [t.grad.clone() for t in (qg, kg, vg, bg)]

    g_nockpt = grads_from(False)
    g_ckpt = grads_from(True)
    max_err = max((a - b).abs().max().item() for a, b in zip(g_nockpt, g_ckpt))
    assert max_err < 1e-4, f"checkpoint parity: max err = {max_err:.3e}"
