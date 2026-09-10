"""Diagnostics for the filler chapter.

`occlusion_probe` measures QTOK-flip: the fraction of predictions that change
when the query token is half-occluded, at a fixed filler length N=200. Above
0.15 means the readout is query-conditioned (leaning on tokens next to the
query); at or below 0.15 it is state-persistent (using the memory). An analogous
atok_flip occludes the answer marker.

Operationalization:
  - "half-occluded" = multiply that token's embedding by 0.5 before the stack.
  - the "query token" is the query entity just before the answer marker
    (position L-2 in the filler-inserted sequence); ATOK is at L-1. The
    prediction is read at the final position.
  - flip = fraction of examples whose argmax prediction changes under occlusion.

`div_norm` measures state divergence: the maximum residual-stream norm at the
last position when evaluated at N=2000 fillers.
"""
from __future__ import annotations

import torch

from .nhop_task import _sample_Nhop_factorized_truncated_injective, insert_filler


def _forward_from_embed(model, emb):
    """Run the stack from a (possibly perturbed) embedding tensor, reusing the
    model's own submodules. Mirrors NovelStack.forward minus the embed step."""
    x = emb
    for blk in model.blocks:
        x = blk(x)
    return model.head(model.norm_f(x))


@torch.no_grad()
def occlusion_probe(model, task_cfg, K_list, batch, gen, device,
                    N: int = 200, scale: float = 0.5) -> dict:
    """QTOK-flip / ATOK-flip probe (see module docstring)."""
    was_training = model.training
    model.eval()
    seq_base, _, _ = _sample_Nhop_factorized_truncated_injective(
        K_list, task_cfg, batch, gen, device)
    seq = insert_filler(seq_base, N)                     # [B, L]
    emb = model.embed(seq)                               # [B, L, D]
    base_pred = _forward_from_embed(model, emb)[:, -1].argmax(-1)

    def flip_at(pos: int) -> float:
        e = emb.clone()
        e[:, pos] = e[:, pos] * scale                    # half-occlude one token
        pred = _forward_from_embed(model, e)[:, -1].argmax(-1)
        return float((pred != base_pred).float().mean().item())

    out = {"qtok_flip": flip_at(-2),   # query entity, just before the answer marker
           "atok_flip": flip_at(-1)}   # the answer marker itself
    if was_training:
        model.train()
    return out


@torch.no_grad()
def div_norm(model, task_cfg, K_list, gen, device,
             N: int = 2000, batch: int = 4) -> float:
    """Max residual-stream norm at the last position under N=2000 fillers.
    Above 1e3 = diverging; below 1e2 = healthy (paper's thresholds)."""
    seq_base, _, _ = _sample_Nhop_factorized_truncated_injective(
        K_list, task_cfg, batch, gen, device)
    seq_far = insert_filler(seq_base, N)
    _, hidden = model(seq_far, return_hidden=True)
    return float(hidden[-1].float().norm(dim=-1).max().item())
