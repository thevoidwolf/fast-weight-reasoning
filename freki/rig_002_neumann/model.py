"""Neumann-Read primitive — Candidate 2 for FENRIR redesign.

Motivation. K-hop chain composition is exactly what the transitive-closure
operator computes: `(I − Mᵀ)⁻¹ q = q + Mᵀq + (Mᵀ)²q + (Mᵀ)³q + …`
Truncating the series to K_read terms gives *depth-K compositional retrieval
in a single primitive read*, without needing K training layers or K per-token
processing steps to compose facts.

Per-token recurrence:

    M_0   = 0
    M_t   = M_{t-1} + β_t · k_t · v_tᵀ                         (additive write)
    y_t   = Σ_{k=0..K_read} α_k · (M_tᵀ)^k · q_t              (Neumann-series read)

Parallel form (fully vectorized — no Python for-loop over sequence).

  Write is a plain cumulative sum of rank-1 outer products over time:
      outer_t = β_t · k_t ⊗ v_t                                # [B, L, d, d]
      M_all   = cumsum(outer, dim=1)                            # [B, L, d, d]
  So all M_t are materialized in one shot from a single cumsum.

  Read is per-token independent once M_all is materialized — iterated
  Mᵀ · y done as a single batched einsum per iteration across all L
  positions in parallel:
      y_iter = einsum('blij,bli->blj', M_all, y_iter)

  Total scan cost: O(L · d²) work, O(1) sequential depth (cumsum is
  parallel-scan under the hood). Preserves FENRIR's parallel-scan
  differentiator.

α coefficients are learnable per-layer (init α_0 = α_1 = 1, higher = 0;
training decides how much closure signal helps).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class NeumannConfig:
    d_model: int = 256
    d_key: int = 32
    expand: int = 2
    d_conv: int = 4
    beta_bias_init: float = -3.0
    n_layers: int = 3
    K_read: int = 3                          # truncation depth of Neumann series
    norm_between_iters: bool = False         # RMSNorm y between M^T applications
    alpha_init: tuple = (1.0, 1.0, 0.0, 0.0) # per-order initial α (len == K_read+1)


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * x / rms


class NeumannMixer(nn.Module):
    def __init__(self, cfg: NeumannConfig):
        super().__init__()
        self.cfg = cfg
        self.d = cfg.d_key
        self.d_inner = cfg.d_model * cfg.expand
        self.K_read = cfg.K_read

        self.in_proj = nn.Linear(cfg.d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=cfg.d_conv,
            groups=self.d_inner, padding=cfg.d_conv - 1, bias=True,
        )
        self.q_proj = nn.Linear(self.d_inner, self.d, bias=False)
        self.k_proj = nn.Linear(self.d_inner, self.d, bias=False)
        self.v_proj = nn.Linear(self.d_inner, self.d, bias=False)
        self.beta_proj = nn.Linear(self.d_inner, 1, bias=True)
        nn.init.constant_(self.beta_proj.bias, cfg.beta_bias_init)

        a_init = list(cfg.alpha_init)
        if len(a_init) < self.K_read + 1:
            a_init = a_init + [0.0] * (self.K_read + 1 - len(a_init))
        elif len(a_init) > self.K_read + 1:
            a_init = a_init[:self.K_read + 1]
        self.alpha = nn.Parameter(torch.tensor(a_init, dtype=torch.float32))

        if cfg.norm_between_iters:
            self.iter_norm = RMSNorm(self.d)
        else:
            self.iter_norm = None

        self.readout_proj = nn.Linear(self.d, self.d_inner, bias=False)
        self.out_proj = nn.Linear(self.d_inner, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        xz = self.in_proj(x)
        x_workspace, z = xz.chunk(2, dim=-1)
        x_conv = self.conv1d(x_workspace.transpose(1, 2))[..., :L].transpose(1, 2)
        x_conv = F.silu(x_conv)

        q = self.q_proj(x_conv)                                    # [B, L, d]
        k = self.k_proj(x_conv)
        v = self.v_proj(x_conv)
        beta = torch.sigmoid(self.beta_proj(x_conv).squeeze(-1))    # [B, L]

        # ---- Vectorized write: cumulative sum of rank-1 outer products ----
        # outer[b,l,i,j] = β_l · k[b,l,i] · v[b,l,j]
        outer = torch.einsum('bl,bli,blj->blij', beta, k, v)        # [B, L, d, d]
        M_all = torch.cumsum(outer, dim=1)                          # [B, L, d, d]

        # ---- Vectorized read: iterated Mᵀ · y in parallel across positions ----
        # y_iter starts as q at each position (0-th Neumann term)
        y_iter = q                                                   # [B, L, d]
        y_acc = self.alpha[0] * y_iter
        for i in range(1, self.K_read + 1):
            # per-t apply Mᵀ to y_iter:   out[b,l,j] = Σ_i M_all[b,l,i,j] · y_iter[b,l,i]
            y_iter = torch.einsum('blij,bli->blj', M_all, y_iter)   # [B, L, d]
            if self.iter_norm is not None:
                y_iter = self.iter_norm(y_iter)
            y_acc = y_acc + self.alpha[i] * y_iter

        y = self.readout_proj(y_acc)                                # [B, L, d_inner]
        y = y * F.silu(z)                                            # z-gate
        return self.out_proj(y)


class NeumannBlock(nn.Module):
    def __init__(self, cfg: NeumannConfig):
        super().__init__()
        self.norm = RMSNorm(cfg.d_model)
        self.mixer = NeumannMixer(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mixer(self.norm(x))


class NeumannStack(nn.Module):
    def __init__(self, vocab_size: int, cfg: NeumannConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([NeumannBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return self.lm_head(x)


def build_model(vocab_size: int, cfg: NeumannConfig | None = None) -> NeumannStack:
    cfg = cfg or NeumannConfig()
    return NeumannStack(vocab_size=vocab_size, cfg=cfg)
