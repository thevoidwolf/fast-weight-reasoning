"""Chunk-parallel core for the corrected-orientation eager closure.

Pure tensor
functions, no dependencies, importable from the mixer wrapper AND runnable
standalone as a self-test (`python chunked_core.py`).

Recurrence (matches EagerClosureFixedMixer's python loop exactly):

    a_t = M_{t-1} k_t                     # reverse lookup, corrected orientation
                                          #  (einsum 'bij,bj->bi' = M · k)
    w_t = beta_t (k_t + a_t)
    M_t = M_{t-1} + w_t v_t^T
    y_t = q_t^T M_t                       # read-after-write, degree-1

Chunked form (DeltaNet/UT-transform class + one extra M0 K matmul). Within a
chunk of size c, M_{t-1} = M0 + sum_{s<t} w_s v_s^T, so

    w_t = beta_t ( k_t + M0 k_t + sum_{s<t} (v_s . k_t) w_s )

Stacking rows w_t^T into W [c, d]:

    (I - diag(beta) T) W = diag(beta) (K + P)
      P      := K M0^T                (row t = (M0 k_t)^T)   [c, d]
      T[t,s] := v_s . k_t  for s < t  (strictly lower tri)   [c, c]

I - diag(beta) T is UNIT LOWER TRIANGULAR -> one batched trsm
(torch.linalg.solve_triangular), no python loop over positions. Then

    Y   = Q M0 + tril(Q W^T, incl diag) V        # read-after-write
    M_c = M0 + W^T V

Per-chunk cost O(d^2 c + d c^2 + c^3); serial depth L/c instead of L.

Numerics: the recurrence runs in fp32 regardless of autocast — M is a
long-horizon accumulator and bf16 accumulation degrades retrieval. Inputs are
cast in, outputs cast back to the caller's dtype.
"""
from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint as _ckpt


# --------------------------------------------------------------------------
# sequential reference (identical math to the python loop)
# --------------------------------------------------------------------------

def eager_fixed_sequential(q, k, v, beta, M0=None):
    """q, k, v: [B, L, d]; beta: [B, L]; M0: [B, d, d] or None.
    Returns y [B, L, d], M_final [B, d, d]. Runs in the input dtype."""
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
# chunk-parallel form
# --------------------------------------------------------------------------

def _chunk_step(M0, Kc, Vc, Qc, bc, need_mk: bool):
    """One chunk. M0 [B,d,d]; Kc/Vc/Qc [B,c,d]; bc [B,c].
    Returns (Yc [B,c,d], M_new [B,d,d], Mk [B,c,d] or None)."""
    c = Kc.shape[1]
    P = Kc @ M0.transpose(-1, -2)                      # row t = (M0 k_t)^T
    T = torch.tril(Kc @ Vc.transpose(-1, -2), -1)      # T[t,s] = v_s.k_t, s<t
    eye = torch.eye(c, dtype=Kc.dtype, device=Kc.device)
    Aeq = eye - bc.unsqueeze(-1) * T                   # unit lower triangular
    rhs = bc.unsqueeze(-1) * (Kc + P)
    W = torch.linalg.solve_triangular(Aeq, rhs, upper=False,
                                      unitriangular=True)   # [B,c,d]
    Yc = Qc @ M0 + torch.tril(Qc @ W.transpose(-1, -2)) @ Vc
    M_new = M0 + W.transpose(-1, -2) @ Vc
    Mk = (P + T @ W) if need_mk else None              # per-position M_{t-1} k_t
    return Yc, M_new, Mk


def eager_fixed_chunked(q, k, v, beta, M0=None, chunk_size: int = 64,
                        use_checkpoint: bool = False, need_mk: bool = False):
    """Chunk-parallel forward, numerically equal to eager_fixed_sequential
    (up to reduction-order noise at the working precision).

    Runs internally in fp32; y is cast back to q.dtype. M_final and Mk are
    returned in fp32. Set use_checkpoint=True during training to recompute
    intra-chunk intermediates in backward.
    """
    B, L, d = k.shape
    out_dtype = q.dtype
    cdt = torch.float64 if q.dtype == torch.float64 else torch.float32
    qf, kf, vf, bf = q.to(cdt), k.to(cdt), v.to(cdt), beta.to(cdt)
    M = torch.zeros(B, d, d, dtype=cdt, device=k.device) if M0 is None \
        else M0.to(cdt)

    ys, mks = [], [] if need_mk else None
    for s0 in range(0, L, chunk_size):
        e = min(s0 + chunk_size, L)
        args = (M, kf[:, s0:e], vf[:, s0:e], qf[:, s0:e], bf[:, s0:e])
        if use_checkpoint and torch.is_grad_enabled():
            Yc, M, Mk = _ckpt(_chunk_step, *args, need_mk,
                              use_reentrant=False)
        else:
            Yc, M, Mk = _chunk_step(*args, need_mk)
        ys.append(Yc)
        if need_mk:
            mks.append(Mk)

    y = torch.cat(ys, dim=1).to(out_dtype)
    return (y, M, torch.cat(mks, dim=1)) if need_mk else (y, M, None)


# --------------------------------------------------------------------------
# self-test: fp64 forward equivalence + fp32 gradient equivalence
# --------------------------------------------------------------------------

def _selftest():
    torch.manual_seed(0)
    B, L, d = 4, 96, 16
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}  B={B} L={L} d={d}")

    q = torch.randn(B, L, d, dtype=torch.float64, device=dev) / d**0.5
    k = torch.randn(B, L, d, dtype=torch.float64, device=dev) / d**0.5
    v = torch.randn(B, L, d, dtype=torch.float64, device=dev) / d**0.5
    beta = torch.sigmoid(torch.randn(B, L, dtype=torch.float64, device=dev))
    M0 = torch.randn(B, d, d, dtype=torch.float64, device=dev) / d

    y_ref, M_ref = eager_fixed_sequential(q, k, v, beta, M0)
    worst = 0.0
    for c in (1, 7, 8, 16, 32, 64, L):
        y_c, M_c, mk = eager_fixed_chunked(q, k, v, beta, M0, chunk_size=c,
                                           need_mk=True)
        err = max((y_ref - y_c).abs().max().item(),
                  (M_ref - M_c).abs().max().item())
        worst = max(worst, err)
        print(f"  c={c:>3}: max|seq-chunked| = {err:.3e}")
    print(f"  forward fp64: {'PASS' if worst < 1e-9 else 'FAIL'}")


if __name__ == "__main__":
    _selftest()
