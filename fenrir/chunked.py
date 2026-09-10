"""Chunk-parallel kernel for the FENRIR rev variant.

The rev variant's per-position recurrence is:

    a_t   = M_{t-1} k_t              # reverse lookup (M k, contract axis 1)
    w_t   = beta_t * (k_t + a_t)
    M_t   = M_{t-1} + w_t v_t^T      # outer-product write
    y_t   = q_t^T M_t                # read-after-write, degree-1

Within a chunk of size c, M_{t-1} = M0 + sum_{s<t} w_s v_s^T, so

    w_t = beta_t * (k_t + M0 k_t + sum_{s<t} (v_s . k_t) w_s)

Stacking rows w_t^T into W (shape [c, d]):

    (I - diag(beta) T) W = diag(beta) (K + P)
      P      := K M0^T               (row t = (M0 k_t)^T)      shape [c, d]
      T[t,s] := v_s . k_t  for s < t (strictly lower triangular) shape [c, c]

I - diag(beta) T is UNIT LOWER TRIANGULAR, so this is one batched
solve_triangular per chunk (torch.linalg.solve_triangular), no python
loop over positions. Then

    Y   = Q M0 + tril(Q W^T, incl diag) V       # read-after-write
    M_c = M0 + W^T V

Per-chunk cost is O(d^2 c + d c^2 + c^3); serial depth is L/c instead of L.

Only the REV variant admits this clean triangular closure. The FWD variant
(eager term M^T k) requires a bilinear reduction over past (w_s . k_t) v_s
that does not reduce to a single triangular solve without further algebra,
so this file implements rev only. Training scripts for the fwd variant
fall back to the sequential path in model.py.

Numerics: the recurrence runs in fp32 regardless of the caller's dtype
because M is a long-horizon accumulator and bf16 accumulation degrades
retrieval on long sequences. Inputs are cast in, outputs cast back to
the caller's dtype.

Backward memory: with use_checkpoint=True only chunk-boundary states are
kept live through autograd bookkeeping; intra-chunk intermediates are
recomputed in backward (torch.utils.checkpoint, non-reentrant).
"""
from __future__ import annotations

from typing import Optional

import torch
from torch.utils.checkpoint import checkpoint as _ckpt


# --------------------------------------------------------------------------
# Sequential reference (identical math to the rev path in model.py)
# --------------------------------------------------------------------------


def rev_sequential(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   beta: torch.Tensor,
                   M0: Optional[torch.Tensor] = None):
    """Reference sequential loop for the rev variant.

    q, k, v: [B, L, d]; beta: [B, L]; M0: [B, d, d] or None.
    Returns (y [B, L, d], M_final [B, d, d]). Runs in the input dtype.

    Used by the chunked-vs-sequential parity test in tests/test_parity.py.
    """
    B, L, d = k.shape
    M = torch.zeros(B, d, d, dtype=k.dtype, device=k.device) if M0 is None \
        else M0.clone()
    ys = []
    for t in range(L):
        Mk = torch.einsum('bij,bj->bi', M, k[:, t])
        w = beta[:, t, None] * (k[:, t] + Mk)
        M = M + torch.einsum('bi,bj->bij', w, v[:, t])
        ys.append(torch.einsum('bi,bij->bj', q[:, t], M))
    return torch.stack(ys, dim=1), M


# --------------------------------------------------------------------------
# Chunk-parallel form
# --------------------------------------------------------------------------


def _chunk_step(M0, Kc, Vc, Qc, bc, need_eager: bool):
    """One chunk. M0 [B, d, d]; Kc, Vc, Qc [B, c, d]; bc [B, c].
    Returns (Yc [B, c, d], M_new [B, d, d], eager [B, c, d] or None).

    The optional `eager` return holds the per-position M_{t-1} k_t values,
    matching what the sequential mixer stores in _cache['eager'] under
    probe_cache. Enables mid-scan probes on chunked training runs.
    """
    c = Kc.shape[1]
    P = Kc @ M0.transpose(-1, -2)                      # row t = (M0 k_t)^T
    T = torch.tril(Kc @ Vc.transpose(-1, -2), -1)      # T[t, s] = v_s . k_t, s < t
    eye = torch.eye(c, dtype=Kc.dtype, device=Kc.device)
    Aeq = eye - bc.unsqueeze(-1) * T                   # unit lower triangular
    rhs = bc.unsqueeze(-1) * (Kc + P)
    W = torch.linalg.solve_triangular(Aeq, rhs, upper=False,
                                      unitriangular=True)      # [B, c, d]
    Yc = Qc @ M0 + torch.tril(Qc @ W.transpose(-1, -2)) @ Vc
    M_new = M0 + W.transpose(-1, -2) @ Vc
    eager = (P + T @ W) if need_eager else None        # per-position M_{t-1} k_t
    return Yc, M_new, eager


def rev_chunked(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                beta: torch.Tensor,
                M0: Optional[torch.Tensor] = None,
                chunk_size: int = 64,
                use_checkpoint: bool = False,
                need_eager: bool = False):
    """Chunk-parallel forward for the rev variant.

    Numerically equal to rev_sequential up to non-associative-reduction
    noise at the working precision. Runs internally in fp32; y is cast
    back to q.dtype.

    Args:
        q, k, v: [B, L, d] projected queries, keys, values.
        beta:    [B, L] scalar write gates in (0, 1).
        M0:      [B, d, d] initial state (defaults to zeros).
        chunk_size: intra-chunk length. Larger = fewer solve_triangular
            calls but larger per-chunk matrices.
        use_checkpoint: recompute intra-chunk intermediates in backward
            (torch.utils.checkpoint, non-reentrant) to save memory.
        need_eager: also return the per-position M_{t-1} k_t values,
            matching the sequential mixer's _cache['eager'].

    Returns:
        (y [B, L, d], M_final [B, d, d], eager [B, L, d] or None).
    """
    B, L, d = k.shape
    out_dtype = q.dtype
    cdt = torch.float64 if q.dtype == torch.float64 else torch.float32
    qf, kf, vf, bf = q.to(cdt), k.to(cdt), v.to(cdt), beta.to(cdt)
    M = torch.zeros(B, d, d, dtype=cdt, device=k.device) if M0 is None \
        else M0.to(cdt)

    ys = []
    eagers = [] if need_eager else None
    for s0 in range(0, L, chunk_size):
        e = min(s0 + chunk_size, L)
        args = (M, kf[:, s0:e], vf[:, s0:e], qf[:, s0:e], bf[:, s0:e])
        if use_checkpoint and torch.is_grad_enabled():
            Yc, M, eager = _ckpt(_chunk_step, *args, need_eager,
                                 use_reentrant=False)
        else:
            Yc, M, eager = _chunk_step(*args, need_eager)
        ys.append(Yc)
        if need_eager:
            eagers.append(eager)

    y = torch.cat(ys, dim=1).to(out_dtype)
    eager_out = torch.cat(eagers, dim=1) if need_eager else None
    return y, M, eager_out
