"""MatProd primitive — Candidate 1 for FENRIR redesign.

Per-token recurrence (sequential reference):

    P_t   = I + β_t · k_t · v_tᵀ        (rank-1 perturbation of identity)
    M_0   = I
    M_t   = M_{t-1} · P_t                (pure multiplicative state, no additive B_t)
    y_t   = M_tᵀ · q_t

Distinct from FENRIR-rev (`M_t = A_t·M_{t-1} + B_t` with additive B_t), from
DeltaNet (`A_t = I − β·k·kᵀ`, symmetric Householder), and from DeltaProduct
(products of Householders per token). The state is a *pure product* of
rank-1-perturbed identities, general k v^T not constrained to Householder
form — marginal-encoding is not in the solution space.

Composition emerges from the multiplicative structure directly:
  P_1 · P_2  =  I + β1 k1 v1ᵀ + β2 k2 v2ᵀ + β1 β2 ⟨v1, k2⟩ k1 v2ᵀ
              └──────single-hop terms──────┘   └──2-hop cross term──┘

Sequential-only prototype. Chunked kernel (matrix-chain prefix scan) is
last-mile if this shows signal.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MatProdConfig:
    d_model: int = 128
    d_key: int = 32
    expand: int = 2
    d_conv: int = 4
    beta_bias_init: float = -3.0    # sigmoid(-3) ≈ 0.047, ~5% perturbation per step
    n_layers: int = 2
    additive_b: bool = False        # if True: M_t = M_{t-1}·P_t + β·k·vᵀ (hybrid variant)


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * x / rms


class MatProdMixer(nn.Module):
    def __init__(self, cfg: MatProdConfig):
        super().__init__()
        self.cfg = cfg
        self.d = cfg.d_key
        self.d_inner = cfg.d_model * cfg.expand

        self.in_proj = nn.Linear(cfg.d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=cfg.d_conv,
            groups=self.d_inner, padding=cfg.d_conv - 1, bias=True,
        )
        self.q_proj = nn.Linear(self.d_inner, self.d, bias=False)
        self.k_proj = nn.Linear(self.d_inner, self.d, bias=False)
        self.v_proj = nn.Linear(self.d_inner, self.d, bias=False)
        self.beta_proj = nn.Linear(self.d_inner, 1, bias=True)
        # Bias-init β to keep perturbations small at start — the multiplicative
        # state can compound aggressively otherwise.
        nn.init.constant_(self.beta_proj.bias, cfg.beta_bias_init)

        self.readout_proj = nn.Linear(self.d, self.d_inner, bias=False)
        self.out_proj = nn.Linear(self.d_inner, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        xz = self.in_proj(x)
        x_workspace, z = xz.chunk(2, dim=-1)
        x_conv = self.conv1d(x_workspace.transpose(1, 2))[..., :L].transpose(1, 2)
        x_conv = F.silu(x_conv)

        q = self.q_proj(x_conv)                                  # [B, L, d]
        k = self.k_proj(x_conv)
        v = self.v_proj(x_conv)
        beta = torch.sigmoid(self.beta_proj(x_conv).squeeze(-1))  # [B, L]

        # State scan: M_0 = I, M_t = M_{t-1} · (I + β_t k_t v_tᵀ)
        # Equivalently: M_t = M_{t-1} + β_t · (M_{t-1} k_t) ⊗ v_t
        M = torch.eye(self.d, device=x.device, dtype=x.dtype) \
                .unsqueeze(0).expand(B, -1, -1).contiguous()
        y_seq = []
        for t in range(L):
            Mk = torch.einsum('bij,bj->bi', M, k[:, t])          # [B, d]
            update = beta[:, t, None, None] * torch.einsum(
                'bi,bj->bij', Mk, v[:, t])                        # [B, d, d]
            M = M + update
            if self.cfg.additive_b:
                # Hybrid: also add plain outer-product write (linear storage)
                add_b = beta[:, t, None, None] * torch.einsum(
                    'bi,bj->bij', k[:, t], v[:, t])
                M = M + add_b
            # Read: y_t = Mᵀ q_t   (retrieves value for query direction)
            y_t = torch.einsum('bij,bi->bj', M, q[:, t])         # [B, d]
            y_seq.append(y_t)

        y = torch.stack(y_seq, dim=1)                             # [B, L, d]
        y = self.readout_proj(y)
        y = y * F.silu(z)
        return self.out_proj(y)


class MatProdBlock(nn.Module):
    def __init__(self, cfg: MatProdConfig):
        super().__init__()
        self.norm = RMSNorm(cfg.d_model)
        self.mixer = MatProdMixer(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mixer(self.norm(x))


class MatProdStack(nn.Module):
    def __init__(self, vocab_size: int, cfg: MatProdConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([MatProdBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return self.lm_head(x)


def build_model(vocab_size: int, cfg: MatProdConfig | None = None) -> MatProdStack:
    cfg = cfg or MatProdConfig()
    return MatProdStack(vocab_size=vocab_size, cfg=cfg)
