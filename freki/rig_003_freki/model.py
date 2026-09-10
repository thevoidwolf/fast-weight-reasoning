"""FREKI — Fixed Read via Explicit K-chain Iteration.

Successor primitive to the FENRIR pathology. Both prior candidates in this
line (MatProd, Neumann-Read) failed because training refused to use
compositional structure when a single-hop bypass existed:
 - Neumann-Read's learnable α_k for k≥2 stayed near zero after 4000 steps
 - MatProd's O(β^K) suppression let the model treat 2-hop cross terms as noise

Fix: remove the bypass. The read function is HARD-WIRED to K chained
Mᵀ applications with RMSNorm between them. There is no α_0 or α_1 term
for the model to fall back to. Every retrieval MUST go through the K-hop
chain, forcing training to align the write in a way that composition works.

Per-token recurrence (K=2 for the 2-hop chain task):

    M_0   = 0
    M_t   = M_{t-1} + β_t · k_t · v_tᵀ                         (additive write)
    y_t   = RMSNorm(Mᵀ · RMSNorm(Mᵀ · q_t))                     (fixed K=2 chain)

Parallel form. cumsum write + batched einsum reads — no custom kernel.
Preserves FENRIR's parallel-scan property.

Bet (validated 2026-08-12): forcing the compositional read shape onto the
primitive is the only way to make gradient descent align the writes
properly, because there is no lower-loss escape route. Result: 10/10 pass
at 12K steps, k4 ≥ 0.99 every seed. Baseline FENRIR-rev at same config:
20% pass at n=20.

Naming: FREKI ("the ravenous one") was Odin's companion wolf in Norse
myth. Contrast with FENRIR the adversarial wolf whose pathology this
primitive escapes.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FrekiConfig:
    d_model: int = 256
    d_key: int = 32
    expand: int = 2
    d_conv: int = 4
    beta_bias_init: float = -0.5              # σ≈0.38 to keep ||M|| moderate
    n_layers: int = 3
    K_chain: int = 2                          # number of forced Mᵀ chain applications
    tie_kv_proj: bool = False                 # if True, v_proj shares weights with k_proj (alignment fix)


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * x / rms


class FrekiMixer(nn.Module):
    def __init__(self, cfg: FrekiConfig):
        super().__init__()
        self.cfg = cfg
        self.d = cfg.d_key
        self.d_inner = cfg.d_model * cfg.expand
        self.K_chain = cfg.K_chain

        self.in_proj = nn.Linear(cfg.d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=cfg.d_conv,
            groups=self.d_inner, padding=cfg.d_conv - 1, bias=True,
        )
        self.q_proj = nn.Linear(self.d_inner, self.d, bias=False)
        self.k_proj = nn.Linear(self.d_inner, self.d, bias=False)
        self.v_proj = nn.Linear(self.d_inner, self.d, bias=False)
        if cfg.tie_kv_proj:
            # Share weights: value at position t IS the projected key. Aligns
            # v_proj(bridge_pos in fact 1) with k_proj(bridge_pos in fact 2)
            # by construction — the 2-hop chain reads and writes use the same
            # entity embedding by construction.
            self.v_proj.weight = self.k_proj.weight
        self.beta_proj = nn.Linear(self.d_inner, 1, bias=True)
        nn.init.constant_(self.beta_proj.bias, cfg.beta_bias_init)

        # One RMSNorm per chain-step to keep magnitudes bounded through iterated Mᵀ
        self.chain_norms = nn.ModuleList([RMSNorm(self.d) for _ in range(self.K_chain)])

        self.readout_proj = nn.Linear(self.d, self.d_inner, bias=False)
        self.out_proj = nn.Linear(self.d_inner, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        xz = self.in_proj(x)
        x_workspace, z = xz.chunk(2, dim=-1)
        x_conv = self.conv1d(x_workspace.transpose(1, 2))[..., :L].transpose(1, 2)
        x_conv = F.silu(x_conv)

        q = self.q_proj(x_conv)
        k = self.k_proj(x_conv)
        v = self.v_proj(x_conv)
        beta = torch.sigmoid(self.beta_proj(x_conv).squeeze(-1))

        # Vectorized additive write: M_t = cumsum(β_t · k_t v_tᵀ)
        outer = torch.einsum('bl,bli,blj->blij', beta, k, v)        # [B, L, d, d]
        M_all = torch.cumsum(outer, dim=1)                          # [B, L, d, d]

        # Forced K_chain-step chain read with RMSNorm between steps
        y = q                                                        # [B, L, d]
        for step in range(self.K_chain):
            y = torch.einsum('blij,bli->blj', M_all, y)              # Mᵀ · y (per-t)
            y = self.chain_norms[step](y)

        y = self.readout_proj(y)
        y = y * F.silu(z)
        return self.out_proj(y)


class FrekiBlock(nn.Module):
    def __init__(self, cfg: FrekiConfig):
        super().__init__()
        self.norm = RMSNorm(cfg.d_model)
        self.mixer = FrekiMixer(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mixer(self.norm(x))


class FrekiStack(nn.Module):
    def __init__(self, vocab_size: int, cfg: FrekiConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([FrekiBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return self.lm_head(x)


def build_model(vocab_size: int, cfg: FrekiConfig | None = None) -> FrekiStack:
    cfg = cfg or FrekiConfig()
    return FrekiStack(vocab_size=vocab_size, cfg=cfg)
