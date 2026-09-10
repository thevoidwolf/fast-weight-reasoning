"""Hop-Stratified FREKI (v2) — per-hop k/v projections.

Aux supervision (FENRIR-style deep supervision on final tgt via shared
lm_head at each block's residual output) is orchestrated in run.py via
forward hooks — the model file stays clean.

Fix for the K_chain > 2 failure mode of vanilla FREKI (rig_003) and for
v1's symmetry deadlock.

v1 diagnosis. K_chain separate M matrices with a softmax write-gate all
sharing a single (k_proj, v_proj) chance-plateaued: at init the softmax
gates are uniform, base_outer is shared across banks, so every M[h] gets
identical writes ⇒ identical reads ⇒ no gradient signal to differentiate
gates. Symmetry deadlock from step 0.

v2 fix. Give each hop its own k_proj[h] and v_proj[h]. At init the K
banks receive DIFFERENT content (different random projection means
different outer products) so the softmax gate has a real signal to
specialize: routing a token to bank h now has a real, distinct effect on
the chained read compared to routing to bank h'. The gate is still
present and still routes writes; the projections just guarantee the
banks are distinguishable content-wise from step 0.

    k_h, v_h = k_proj[h](x), v_proj[h](x)
    for h in range(K_chain):
        M[h]  = M[h]  + gate[h] · β · k_h · v_hᵀ                (routed, per-hop content)

    y = q
    for h in range(K_chain):
        y = RMSNorm(M[h]ᵀ · y)                                   (per-hop read)

Parallel form. Each M[h] is cumsum(gate[h] · β · k_h · v_hᵀ) over time.
Reads are K einsums. Per-hop projections add K_chain × 2 × d_inner × d
params per layer (small for d_inner=512, d=32, K=3 → ~100K per layer).

Compared to FREKI-family: this is the natural generalization to arbitrary
compositional depth. FREKI is the special case K_chain=2 with a single M
(equivalent to hop-stratified with all gates collapsed to the same M).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HopStratConfig:
    d_model: int = 256
    d_key: int = 32
    expand: int = 2
    d_conv: int = 4
    beta_bias_init: float = -0.5
    n_layers: int = 3
    K_chain: int = 3                       # number of hops = number of M matrices


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * x / rms


class HopStratMixer(nn.Module):
    def __init__(self, cfg: HopStratConfig):
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
        # Per-hop k / v projections — different content per bank at init
        # breaks the v1 symmetry deadlock among softmax-gated write routes.
        self.k_projs = nn.ModuleList(
            [nn.Linear(self.d_inner, self.d, bias=False) for _ in range(self.K_chain)]
        )
        self.v_projs = nn.ModuleList(
            [nn.Linear(self.d_inner, self.d, bias=False) for _ in range(self.K_chain)]
        )
        self.beta_proj = nn.Linear(self.d_inner, 1, bias=True)
        nn.init.constant_(self.beta_proj.bias, cfg.beta_bias_init)

        # Per-token softmax gate over K_chain hop-M matrices — routes writes.
        self.gate_proj = nn.Linear(self.d_inner, self.K_chain, bias=True)
        # Symmetric init: bias 0 means uniform 1/K_chain across hops at start.
        nn.init.zeros_(self.gate_proj.bias)

        # RMSNorm inserted between per-hop reads (as in FREKI)
        self.chain_norms = nn.ModuleList([RMSNorm(self.d) for _ in range(self.K_chain)])

        self.readout_proj = nn.Linear(self.d, self.d_inner, bias=False)
        self.out_proj = nn.Linear(self.d_inner, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        xz = self.in_proj(x)
        x_workspace, z = xz.chunk(2, dim=-1)
        x_conv = self.conv1d(x_workspace.transpose(1, 2))[..., :L].transpose(1, 2)
        x_conv = F.silu(x_conv)

        q = self.q_proj(x_conv)                                     # [B, L, d]
        beta = torch.sigmoid(self.beta_proj(x_conv).squeeze(-1))     # [B, L]
        gate = F.softmax(self.gate_proj(x_conv), dim=-1)             # [B, L, K_chain]

        # Per-hop writes: M[h]_t = cumsum(gate[..., h] · β · k_h · v_hᵀ)
        # Each hop has its own k/v projection so banks are content-distinct
        # even when gate is uniform at init.
        M_all = []
        for h in range(self.K_chain):
            k_h = self.k_projs[h](x_conv)                             # [B, L, d]
            v_h = self.v_projs[h](x_conv)                             # [B, L, d]
            outer_h = torch.einsum('bl,bli,blj->blij', beta, k_h, v_h)  # [B, L, d, d]
            gated = gate[:, :, h, None, None] * outer_h               # [B, L, d, d]
            M_h = torch.cumsum(gated, dim=1)                          # [B, L, d, d]
            M_all.append(M_h)

        # Per-hop chained read: y_{h+1} = RMSNorm(M[h]ᵀ · y_h)
        y = q                                                          # [B, L, d]
        for h in range(self.K_chain):
            y = torch.einsum('blij,bli->blj', M_all[h], y)             # M[h]ᵀ · y
            y = self.chain_norms[h](y)

        y = self.readout_proj(y)
        y = y * F.silu(z)
        return self.out_proj(y)


class HopStratBlock(nn.Module):
    def __init__(self, cfg: HopStratConfig):
        super().__init__()
        self.norm = RMSNorm(cfg.d_model)
        self.mixer = HopStratMixer(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mixer(self.norm(x))


class HopStratStack(nn.Module):
    def __init__(self, vocab_size: int, cfg: HopStratConfig):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([HopStratBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return self.lm_head(x)


def build_model(vocab_size: int, cfg: HopStratConfig | None = None) -> HopStratStack:
    cfg = cfg or HopStratConfig()
    return HopStratStack(vocab_size=vocab_size, cfg=cfg)
