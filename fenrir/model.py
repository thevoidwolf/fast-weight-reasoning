"""FENRIR mixer, reference implementation for the paper.

Two variants:
  - fwd: eager term = M^T k  (forward lookup: match k against stored KEYS, return VALUES)
  - rev: eager term = M   k  (reverse lookup: match k against stored VALUES, return KEYS)

Both share the same write (M += beta * outer(k_eff, v)) and the same read
(y = M^T q, standard fast-weight forward read). The variant flag only
changes the address perturbation direction.

All convention claims here are verified by tests/test_conventions.py.
The mixer's forward pass IS the ground truth for every formula in the paper.
If a paper claim disagrees with the code below, the paper is wrong.

Storage convention:
  M has shape [B, d_key, d_value].
  M[b, i, j] accumulates the outer product k_eff[b, i] * v[b, j],
  so axis 0 (i) is the KEY axis and axis 1 (j) is the VALUE axis.

Since both variants use k as input to the eager term (and add it to k_eff
which must be d_key-shaped), we require d_key == d_value.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from chunked import rev_chunked


# --------------------------------------------------------------------------
# Config + norm
# --------------------------------------------------------------------------


@dataclass
class MixerConfig:
    """Hyperparameters shared across the scaffold and the mixer."""
    d_model: int = 256
    d_key: int = 32
    d_value: int = 32           # must equal d_key for the eager term to type-check
    expand: int = 2             # in_proj widens d_model → 2 * expand * d_model
    d_conv: int = 4             # depthwise conv1d kernel size


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * x / rms


# --------------------------------------------------------------------------
# The FENRIR mixer
# --------------------------------------------------------------------------


VALID_VARIANTS = ("fwd", "rev")


class FenrirMixer(nn.Module):
    """FENRIR fast-weight mixer with variant-selectable eager term.

    Recurrence (per position t, given projected q, k, v, beta):
        eager   = { M^T k   if variant == "fwd"     # forward lookup
                  { M   k   if variant == "rev"     # reverse lookup
        k_eff   = k + eager
        M       = M + beta * outer(k_eff, v)         # storage: M[i,j] = k_eff[i]*v[j]
        y       = M^T q                               # standard fast-weight read

    The variant flag ONLY changes the eager term. Write and read are identical.

    Scaffold (in_proj, conv1d, SiLU, z-gate, out_proj) is a Mamba S4-style
    block layout.
    """

    def __init__(self, cfg: MixerConfig, variant: str = "rev",
                 chunked: bool = False, chunk_size: int = 64,
                 use_checkpoint: bool = False,
                 use_eager: bool = True):
        super().__init__()
        if variant not in VALID_VARIANTS:
            raise ValueError(f"variant must be one of {VALID_VARIANTS}, got {variant!r}")
        if cfg.d_key != cfg.d_value:
            raise ValueError(
                f"FENRIR eager term requires d_key == d_value (got "
                f"d_key={cfg.d_key}, d_value={cfg.d_value}). "
                f"k_eff = k + eager needs eager to be d_key-shaped."
            )
        if chunked and variant == "fwd":
            raise ValueError(
                "chunked=True is not supported for variant='fwd'. "
                "See chunked.py for why (fwd's per-position closure does "
                "not reduce to a single triangular solve). Train fwd with "
                "chunked=False."
            )
        if chunked and not use_eager:
            raise ValueError(
                "use_eager=False is only supported on the sequential path. "
                "The chunked kernel is derived from the eager-term recurrence. "
                "Run the no-eager ablation with chunked=False."
            )
        self.cfg = cfg
        self.variant = variant
        self.chunked = chunked
        self.chunk_size = chunk_size
        self.use_checkpoint = use_checkpoint
        self.use_eager = use_eager
        self.d = cfg.d_key
        self.d_inner = cfg.d_model * cfg.expand

        # Scaffold: in_proj widens d_model → 2*d_inner (x_workspace + z)
        self.in_proj = nn.Linear(cfg.d_model, 2 * self.d_inner, bias=False)
        # Depthwise causal conv1d over the sequence axis
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=cfg.d_conv,
            groups=self.d_inner, padding=cfg.d_conv - 1, bias=True,
        )
        # Per-position projections: q, k, v from x_conv; beta from x_conv
        self.q_proj = nn.Linear(self.d_inner, self.d, bias=False)
        self.k_proj = nn.Linear(self.d_inner, self.d, bias=False)
        self.v_proj = nn.Linear(self.d_inner, self.d, bias=False)
        self.beta_proj = nn.Linear(self.d_inner, 1, bias=True)
        # Read-out back to d_inner, then z-gate, then out_proj to d_model
        self.readout_proj = nn.Linear(self.d, self.d_inner, bias=False)
        self.out_proj = nn.Linear(self.d_inner, cfg.d_model, bias=False)

        # Set probe_cache=True from the outside to have the forward pass
        # store q/k/v/β/eager/y at each timestep in self._cache.
        # Off by default so training doesn't accumulate cache memory.
        self.probe_cache: bool = False
        self._cache: dict = {}

    # --- the two eager-term operators, each a single einsum ---------------

    @staticmethod
    def _eager_rev(M: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """M · k, contract on axis 1 (value axis) of M. Returns d_key-shaped.
        Semantics: match k against stored values, return stored keys."""
        return torch.einsum('bij,bj->bi', M, k)

    @staticmethod
    def _eager_fwd(M: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """M^T · k, contract on axis 0 (key axis) of M. Returns d_value-shaped.
        Semantics: match k against stored keys, return stored values."""
        return torch.einsum('bij,bi->bj', M, k)

    def _eager(self, M: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Dispatch to the variant's eager term."""
        return self._eager_rev(M, k) if self.variant == "rev" else self._eager_fwd(M, k)

    # --- forward pass -----------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, d_model] to [B, L, d_model]. Sequential per-position scan.

        A chunk-parallel kernel for the rev variant lives in chunked.py and
        is used by train.py when --chunked is on (about 6x faster in practice).
        The sequential path here is the authoritative reference implementation.
        """
        B, L, _ = x.shape
        # Scaffold: in_proj, causal conv, SiLU
        xz = self.in_proj(x)                                    # [B, L, 2*d_inner]
        x_workspace, z = xz.chunk(2, dim=-1)                    # each [B, L, d_inner]
        # Causal conv1d: pad by kernel_size-1 on the left (via pytorch's padding
        # arg with our right-side trim below), then slice back to L
        x_conv = self.conv1d(x_workspace.transpose(1, 2))[..., :L].transpose(1, 2)
        x_conv = F.silu(x_conv)                                 # [B, L, d_inner]

        # Per-position projections
        q = self.q_proj(x_conv)                                 # [B, L, d]
        k = self.k_proj(x_conv)
        v = self.v_proj(x_conv)
        beta = torch.sigmoid(self.beta_proj(x_conv).squeeze(-1))  # [B, L] in (0, 1)

        # State scan: chunked (rev only) or sequential (both variants)
        if self.chunked:
            y, _, eager_stacked = rev_chunked(
                q, k, v, beta,
                chunk_size=self.chunk_size,
                use_checkpoint=self.use_checkpoint,
                need_eager=self.probe_cache,
            )
            if self.probe_cache:
                self._cache = {
                    "q": q.detach(), "k": k.detach(), "v": v.detach(),
                    "beta": beta.detach(),
                    "eager": eager_stacked.to(y.dtype),
                    "y": y.detach(),
                }
        else:
            M = torch.zeros(B, self.d, self.d, device=x.device, dtype=x.dtype)
            y_seq = []
            eager_seq = [] if self.probe_cache else None
            for t in range(L):
                if self.use_eager:
                    eager = self._eager(M, k[:, t])             # [B, d]
                    k_eff = k[:, t] + eager                     # [B, d]
                else:
                    eager = torch.zeros_like(k[:, t])           # ablation: k_eff = k
                    k_eff = k[:, t]
                if eager_seq is not None:
                    eager_seq.append(eager.detach())
                # Write: M[i, j] += beta * k_eff[i] * v[j]
                outer = torch.einsum('bi,bj->bij', k_eff, v[:, t])  # [B, d, d]
                M = M + beta[:, t].view(-1, 1, 1) * outer
                # Read: y[j] = sum_i q[i] * M[i, j] = (M^T q)[j]
                y_t = torch.einsum('bi,bij->bj', q[:, t], M)    # [B, d]
                y_seq.append(y_t)

            y = torch.stack(y_seq, dim=1)                        # [B, L, d]
            if self.probe_cache:
                self._cache = {
                    "q": q.detach(), "k": k.detach(), "v": v.detach(),
                    "beta": beta.detach(),
                    "eager": torch.stack(eager_seq, dim=1),
                    "y": y.detach(),
                }

        # Read-out projection, z-gate, out-projection
        y = self.readout_proj(y)                                 # [B, L, d_inner]
        y = y * F.silu(z)                                        # z-gate
        return self.out_proj(y)                                  # [B, L, d_model]


# --------------------------------------------------------------------------
# Model stack: N FENRIR blocks, residual + RMSNorm, LM head
# --------------------------------------------------------------------------


class FenrirBlock(nn.Module):
    """Pre-norm residual block: x = x + FenrirMixer(RMSNorm(x))."""

    def __init__(self, cfg: MixerConfig, variant: str,
                 chunked: bool = False, chunk_size: int = 64,
                 use_checkpoint: bool = False,
                 use_eager: bool = True):
        super().__init__()
        self.norm = RMSNorm(cfg.d_model)
        self.mixer = FenrirMixer(cfg, variant=variant, chunked=chunked,
                                 chunk_size=chunk_size,
                                 use_checkpoint=use_checkpoint,
                                 use_eager=use_eager)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mixer(self.norm(x))


class FenrirStack(nn.Module):
    """FENRIR model: token embedding → L FENRIR blocks → RMSNorm → LM head.

    Constructor takes the same MixerConfig and a variant string; every block
    uses the same variant (a single-variant model). Mixing variants across
    layers is out of scope for the paper.
    """

    def __init__(self, vocab_size: int, cfg: MixerConfig, variant: str,
                 n_layers: int = 3, chunked: bool = False,
                 chunk_size: int = 64, use_checkpoint: bool = False,
                 use_eager: bool = True):
        super().__init__()
        self.cfg = cfg
        self.variant = variant
        self.n_layers = n_layers
        self.use_eager = use_eager
        self.embed = nn.Embedding(vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([
            FenrirBlock(cfg, variant, chunked=chunked, chunk_size=chunk_size,
                        use_checkpoint=use_checkpoint, use_eager=use_eager)
            for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(cfg.d_model)
        # Tied LM head: use the embedding matrix transposed
        self.lm_head = nn.Linear(cfg.d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: [B, L] long → logits [B, L, vocab_size]."""
        x = self.embed(tokens)                                   # [B, L, d_model]
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return self.lm_head(x)
