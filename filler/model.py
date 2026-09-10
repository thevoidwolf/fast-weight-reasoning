"""Model for the filler chapter: the eager-closure fast-weight mixer on a
pre-norm residual stack (NovelStack).

  - MixerConfig / RMSNorm / _ChainedWrapperMixin  : shared scaffold
  - EagerClosureFixedMixer                        : the sequential reference
  - EagerClosureFixedChunkedMixer                 : the chunk-parallel version
                                                    used for training (fp32
                                                    recurrence, O(L/c) depth)
  - NovelBlock / NovelStack                       : x + mixer(norm(x)), stacked

The mixer is torch-only (a fast-weight matrix state), so it runs on CPU with no
mamba_ssm dependency.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .chunked_core import eager_fixed_chunked


# --------------------------------------------------------------------------
# Config + norm
# --------------------------------------------------------------------------

@dataclass
class MixerConfig:
    d_model: int = 128
    d_state: int = 32       # only used by a Mamba2 baseline (not used here)
    d_conv: int = 4
    expand: int = 2
    headdim: int = 64
    d_key: int = 32         # fast-weight key dim
    d_value: int = 32       # fast-weight value dim


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * x / rms


# --------------------------------------------------------------------------
# Shared mixer prelude (in_proj -> depthwise conv1d -> silu, plus a z gate)
# --------------------------------------------------------------------------

class _ChainedWrapperMixin(nn.Module):
    """Shared prelude: in_proj -> conv1d -> silu. Subclasses add per-position
    projections + a scan loop and finish with readout_proj + z-gate + out_proj."""
    def __init__(self, cfg: MixerConfig, sharpen: str = "l2"):
        super().__init__()
        self.cfg = cfg
        self.d = cfg.d_key
        self.d_inner = cfg.d_model * cfg.expand
        self.sharpen_mode = sharpen

        self.in_proj = nn.Linear(cfg.d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=cfg.d_conv,
            groups=self.d_inner, padding=cfg.d_conv - 1, bias=True,
        )
        self.out_proj = nn.Linear(self.d_inner, cfg.d_model, bias=False)

    def _sharpen(self, x, dim=-1, eps=1e-6):
        if self.sharpen_mode == "l2":
            return x / (x.norm(dim=dim, keepdim=True) + eps)
        return x

    def _prelude(self, x):
        B, L, _ = x.shape
        xz = self.in_proj(x)
        x_workspace, z = xz.chunk(2, dim=-1)
        x_conv = self.conv1d(x_workspace.transpose(1, 2))[..., :L].transpose(1, 2)
        x_conv = F.silu(x_conv)
        return B, L, x_conv, z


# --------------------------------------------------------------------------
# The eager-closure mixer: state-dependent chain-materializing write
# --------------------------------------------------------------------------

class EagerClosureFixedMixer(_ChainedWrapperMixin):
    """Sequential reference. The write address folds in a reverse lookup into
    the current fast-weight state (a_t = M_{t-1} k_t), so a fact written now can
    chain onto one already stored:  w_t = beta_t (k_t + a_t);  M += outer(w, v);
    read y_t = q_t . M_t."""

    def __init__(self, cfg: MixerConfig, sharpen: str = "l2"):
        super().__init__(cfg, sharpen=sharpen)
        self.q_proj = nn.Linear(self.d_inner, self.d, bias=False)
        self.k_proj = nn.Linear(self.d_inner, self.d, bias=False)
        self.v_proj = nn.Linear(self.d_inner, self.d, bias=False)
        self.beta_proj = nn.Linear(self.d_inner, 1, bias=True)
        self.readout_proj = nn.Linear(self.d, self.d_inner, bias=False)

    def forward(self, x):
        B, L, x_conv, z = self._prelude(x)
        q = self.q_proj(x_conv)
        k = self.k_proj(x_conv)
        v = self.v_proj(x_conv)
        beta = torch.sigmoid(self.beta_proj(x_conv).squeeze(-1))

        M = torch.zeros(B, self.d, self.d, device=x.device, dtype=x.dtype)
        y_seq = []
        for t in range(L):
            Mk = torch.einsum('bij,bj->bi', M, k[:, t])
            if getattr(self, "eager_knockout", False):
                Mk = torch.zeros_like(Mk)
            k_eff = k[:, t] + Mk
            outer = torch.einsum('bi,bj->bij', k_eff, v[:, t])
            M = M + beta[:, t].view(-1, 1, 1) * outer
            y_seq.append(torch.einsum('bi,bij->bj', q[:, t], M))

        y = torch.stack(y_seq, dim=1)
        y = self.readout_proj(y)
        y = y * F.silu(z)
        return self.out_proj(y)


_CHUNK = int(os.environ.get("EMBER_CHUNK_SIZE", "64"))
_CKPT = os.environ.get("EMBER_CHUNK_CKPT", "1") == "1"


def _knockout_forward(q, k, v, beta):
    """F1 arm with Mk zeroed: w_t = beta_t k_t (plain input-only write),
    y = q M read-after-write. Sequential on purpose (probe-only path)."""
    B, L, d = k.shape
    M = torch.zeros(B, d, d, dtype=k.dtype, device=k.device)
    ys = []
    for t in range(L):
        w = beta[:, t, None] * k[:, t]
        M = M + torch.einsum('bi,bj->bij', w, v[:, t])
        ys.append(torch.einsum('bi,bij->bj', q[:, t], M))
    return torch.stack(ys, dim=1)


class EagerClosureFixedChunkedMixer(_ChainedWrapperMixin):
    """Chunk-parallel version of EagerClosureFixedMixer. Same math (verified to
    fp64 machine epsilon by chunked_core's self-test); the recurrence runs in
    fp32 even under bf16 autocast. This is the mixer the filler runs use."""

    def __init__(self, cfg: MixerConfig, sharpen: str = "l2",
                 chunk_size: int = _CHUNK, use_checkpoint: bool = _CKPT):
        super().__init__(cfg, sharpen=sharpen)
        self.q_proj = nn.Linear(self.d_inner, self.d, bias=False)
        self.k_proj = nn.Linear(self.d_inner, self.d, bias=False)
        self.v_proj = nn.Linear(self.d_inner, self.d, bias=False)
        self.beta_proj = nn.Linear(self.d_inner, 1, bias=True)
        self.readout_proj = nn.Linear(self.d, self.d_inner, bias=False)
        self.chunk_size = chunk_size
        self.use_checkpoint = use_checkpoint

    def forward(self, x):
        B, L, x_conv, z = self._prelude(x)
        q = self.q_proj(x_conv)
        k = self.k_proj(x_conv)
        v = self.v_proj(x_conv)
        beta = torch.sigmoid(self.beta_proj(x_conv).squeeze(-1))

        if getattr(self, "eager_knockout", False):
            y = _knockout_forward(q.float(), k.float(), v.float(),
                                  beta.float()).to(q.dtype)
        else:
            y, _, _ = eager_fixed_chunked(
                q, k, v, beta,
                chunk_size=self.chunk_size,
                use_checkpoint=self.use_checkpoint and self.training,
                need_mk=False)

        y = self.readout_proj(y.to(x_conv.dtype))
        y = y * F.silu(z)
        return self.out_proj(y)


MIXER_REGISTRY = {
    "eager_closure_fixed": EagerClosureFixedMixer,
    "eager_closure_fixed_chunked": EagerClosureFixedChunkedMixer,
}


# --------------------------------------------------------------------------
# Scaffold: NovelBlock = x + mixer(norm(x)); NovelStack stacks them
# --------------------------------------------------------------------------

class NovelBlock(nn.Module):
    def __init__(self, cfg: MixerConfig, mixer_cls, layer_idx: int = 0):
        super().__init__()
        self.norm = RMSNorm(cfg.d_model)
        try:
            self.mixer = mixer_cls(cfg, layer_idx=layer_idx)
        except TypeError:
            self.mixer = mixer_cls(cfg)

    def forward(self, x):
        return x + self.mixer(self.norm(x))


class NovelStack(nn.Module):
    """Token embedding -> n_layers of NovelBlock -> final norm -> LM head.

    forward(tokens, return_hidden=True) also returns the per-layer hidden states,
    used by the divergence-norm diagnostic.
    """
    def __init__(self, vocab_size: int, cfg: MixerConfig, mixer_name: str,
                 n_layers: int = 3):
        super().__init__()
        if mixer_name not in MIXER_REGISTRY:
            raise KeyError(f"unknown mixer {mixer_name!r}; have {sorted(MIXER_REGISTRY)}")
        mixer_cls = MIXER_REGISTRY[mixer_name]
        self.embed = nn.Embedding(vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([NovelBlock(cfg, mixer_cls, layer_idx=i)
                                     for i in range(n_layers)])
        self.norm_f = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, vocab_size, bias=False)

    def forward(self, tokens, return_hidden: bool = False):
        x = self.embed(tokens)
        hidden_states = [] if return_hidden else None
        for blk in self.blocks:
            x = blk(x)
            if return_hidden:
                hidden_states.append(x)
        logits = self.head(self.norm_f(x))
        if return_hidden:
            return logits, hidden_states
        return logits
