"""Rotation-state primitive — Candidate 4 (Wild Bet B) for FENRIR redesign.

State M is constrained to SO(d) — the orthogonal group. Per-token update
is a right-multiplication by a rotation matrix confined to the 2D plane
span{k, v}. Composition of rotations IS group multiplication, matching
the "K-hop chain composition = product of edge transforms" intuition
from the very first turn of this design session.

Per-token recurrence:

    M_0    = I
    e1     = k / ||k||
    v_⊥    = v − ⟨e1, v⟩·e1
    e2     = v_⊥ / ||v_⊥||
    θ_t    = (π/2) · tanh(β_proj(x))                    (learned angle in (−π/2, π/2))
    R_t    = I + (cos θ_t − 1)·(e1 e1ᵀ + e2 e2ᵀ) + sin θ_t·(e2 e1ᵀ − e1 e2ᵀ)
    M_t    = M_{t−1} · R_t                              (multiplicative — always in SO(d))

    y_t    = RMSNorm(Mᵀ · RMSNorm(Mᵀ · q_t))            (chained read, FREKI style)

Properties.
- **Norm-preserving by construction.** R_t is exactly orthogonal (Rodrigues
  formula for rotation in the k, v plane), so ||M_t · x|| = ||x|| for all t
  and all inputs. Zero possibility of state-explosion or state-collapse.
- **Composition semantics.** If R_t is trained to send k_t → v_t and
  v_{t−1} = k_t (chain condition), then R_t · R_{t−1} · k_{t−1} = v_t
  — the chain retrieves naturally through group multiplication.
- **Chance-plateau attractor may be unreachable.** With M ∈ SO(d), the
  read output has fixed norm and specific direction — the "uniform-over-
  values" degenerate solution isn't in the primitive's output distribution.

Parallelism. Right-multiplicative recurrence `M_t = M_{t−1} · R_t` is
associative — admits Blelloch prefix scan across positions in O(log L)
sequential depth. Sequential loop is used here for prototype; associative
scan is last-mile engineering if the primitive shows signal.

Distinct from every survey row: DeltaProduct uses PRODUCTS of Householder
reflections (discrete generators), we use continuous SO(d) rotations
parametrized by a learned angle. Ladder-above DeltaProduct in the
Grazzi expressivity sense.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RotationConfig:
    d_model: int = 256
    d_key: int = 32
    expand: int = 2
    d_conv: int = 4
    n_layers: int = 3
    K_chain: int = 2                          # forced-compose chained read depth
    theta_scale: float = 1.0                  # θ = theta_scale · (π/2) · tanh(β_proj)


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * x / rms


class RotationMixer(nn.Module):
    def __init__(self, cfg: RotationConfig):
        super().__init__()
        self.cfg = cfg
        self.d = cfg.d_key
        self.d_inner = cfg.d_model * cfg.expand
        self.K_chain = cfg.K_chain
        self.theta_max = cfg.theta_scale * math.pi / 2

        self.in_proj = nn.Linear(cfg.d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=cfg.d_conv,
            groups=self.d_inner, padding=cfg.d_conv - 1, bias=True,
        )
        self.q_proj = nn.Linear(self.d_inner, self.d, bias=False)
        self.k_proj = nn.Linear(self.d_inner, self.d, bias=False)
        self.v_proj = nn.Linear(self.d_inner, self.d, bias=False)
        self.theta_proj = nn.Linear(self.d_inner, 1, bias=True)
        nn.init.zeros_(self.theta_proj.bias)   # θ starts at 0, R starts at I

        # RMSNorm between chained-read Mᵀ applications
        self.chain_norms = nn.ModuleList([RMSNorm(self.d) for _ in range(self.K_chain)])

        self.readout_proj = nn.Linear(self.d, self.d_inner, bias=False)
        self.out_proj = nn.Linear(self.d_inner, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        xz = self.in_proj(x)
        x_workspace, z = xz.chunk(2, dim=-1)
        x_conv = self.conv1d(x_workspace.transpose(1, 2))[..., :L].transpose(1, 2)
        x_conv = F.silu(x_conv)

        q = self.q_proj(x_conv)                                      # [B, L, d]
        k = self.k_proj(x_conv)
        v = self.v_proj(x_conv)
        theta = self.theta_max * torch.tanh(self.theta_proj(x_conv).squeeze(-1))  # [B, L]
        cos_t = torch.cos(theta)                                     # [B, L]
        sin_t = torch.sin(theta)

        # Orthonormalize (k, v) → (e1, e2) per token
        eps = 1e-6
        e1 = k / (k.norm(dim=-1, keepdim=True) + eps)                 # [B, L, d]
        # Gram-Schmidt: v_⊥ = v − ⟨e1, v⟩ e1
        proj = (e1 * v).sum(dim=-1, keepdim=True) * e1                 # [B, L, d]
        v_perp = v - proj
        e2 = v_perp / (v_perp.norm(dim=-1, keepdim=True) + eps)       # [B, L, d]

        # Sequential state scan: M_t = M_{t−1} · R_t
        #   R_t = I + (cos θ − 1)(e1 e1ᵀ + e2 e2ᵀ) + sin θ · (e2 e1ᵀ − e1 e2ᵀ)
        # M · R_t = M + (cos θ − 1)·(M · P_plane) + sin θ · (M · Q_gen)
        # where P_plane = e1 e1ᵀ + e2 e2ᵀ (projector onto rotation plane)
        # and Q_gen = e2 e1ᵀ − e1 e2ᵀ (skew rotation generator)
        # Both products M · P_plane and M · Q_gen are two rank-1 corrections to M,
        # so the update is O(d²) per step, not O(d³).
        M_list = []
        M = torch.eye(self.d, device=x.device, dtype=x.dtype) \
                .unsqueeze(0).expand(B, -1, -1).contiguous()
        for t in range(L):
            e1_t, e2_t = e1[:, t], e2[:, t]                            # [B, d] each
            c_t, s_t = cos_t[:, t], sin_t[:, t]                        # [B]
            # M · e1_t → [B, d]; then outer with e1_t → [B, d, d]  (M P_plane part 1)
            M_e1 = torch.einsum('bij,bj->bi', M, e1_t)                # [B, d]
            M_e2 = torch.einsum('bij,bj->bi', M, e2_t)                # [B, d]
            # (cos θ − 1)·(M e1 e1ᵀ + M e2 e2ᵀ)
            delta_plane = (c_t - 1)[:, None, None] * (
                torch.einsum('bi,bj->bij', M_e1, e1_t) +
                torch.einsum('bi,bj->bij', M_e2, e2_t)
            )
            # sin θ · (M e2 e1ᵀ − M e1 e2ᵀ)
            delta_rot = s_t[:, None, None] * (
                torch.einsum('bi,bj->bij', M_e2, e1_t) -
                torch.einsum('bi,bj->bij', M_e1, e2_t)
            )
            M = M + delta_plane + delta_rot                            # M · R_t
            M_list.append(M)

        M_all = torch.stack(M_list, dim=1)                             # [B, L, d, d]

        # Chained read (FREKI style): K_chain iterations of Mᵀ · y with RMSNorm
        y = q
        for step in range(self.K_chain):
            y = torch.einsum('blij,bli->blj', M_all, y)                # Mᵀ · y (per position)
            y = self.chain_norms[step](y)

        y = self.readout_proj(y)
        y = y * F.silu(z)
        return self.out_proj(y)


class RotationBlock(nn.Module):
    def __init__(self, cfg: RotationConfig):
        super().__init__()
        self.norm = RMSNorm(cfg.d_model)
        self.mixer = RotationMixer(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mixer(self.norm(x))


class RotationStack(nn.Module):
    def __init__(self, vocab_size: int, cfg: RotationConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([RotationBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return self.lm_head(x)


def build_model(vocab_size: int, cfg: RotationConfig | None = None) -> RotationStack:
    cfg = cfg or RotationConfig()
    return RotationStack(vocab_size=vocab_size, cfg=cfg)
