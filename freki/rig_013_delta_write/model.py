"""rig_013 — DeltaNet-style write (Fable's fallback: H3, crosstalk fix).

If rig_012's shared-bus cleanup converges aux/sym but forward path stays
at chance, the problem isn't projection alignment — it's crosstalk in
the flat M. Fix: replace additive write with a delta-rule write that
erases what's already there before inserting.

Additive write (FREKI):    M_t = M_{t-1} + β_t · k_t · v_t^T
Delta-rule write:          M_t = M_{t-1} + β_t · (v_t - M_{t-1}^T · k_t) · k_t^T
                                = M_{t-1} · (I - β_t k_t k_t^T)^T + β_t v_t k_t^T (if ||k||=1)

The delta form removes prior content at the direction of k_t before
adding v_t; reduces crosstalk between writes. Yang et al. 2024 (DeltaNet)
show it has efficient chunkwise parallel form; here we use a sequential
loop for simplicity (~2-3× slower than cumsum but tractable at task size).

We keep the shared-bus cleanup and symmetric-consistency aux from rig_012
— the two fixes are orthogonal.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DeltaBusConfig:
    d_model: int = 256
    d_key: int = 32
    expand: int = 2
    d_conv: int = 4
    beta_bias_init: float = -0.5
    n_layers: int = 3
    K_chain: int = 3
    # Memory-aware β gate (Fable 2026-08-14): β_t = σ(logit − w_p·‖M^T k_t‖²)
    # protects existing content from delta-rule erasure by penalizing writes at
    # keys where memory already has energy. Default OFF preserves the passing recipe.
    use_protect_gate: bool = False
    protect_w_init: float = 0.1


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * x / rms


class DeltaBusMixer(nn.Module):
    def __init__(self, cfg: DeltaBusConfig, embed_matrix: nn.Embedding):
        super().__init__()
        self.cfg = cfg
        self.d = cfg.d_key
        self.d_inner = cfg.d_model * cfg.expand
        self.K_chain = cfg.K_chain
        self.embed_ref = embed_matrix

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

        if cfg.use_protect_gate:
            self.protect_w = nn.Parameter(torch.tensor(float(cfg.protect_w_init)))
        else:
            self.protect_w = None

        self.chain_norms = nn.ModuleList([RMSNorm(self.d) for _ in range(self.K_chain)])

        self.up_proj = nn.Linear(self.d, cfg.d_model, bias=False)
        self.bus_adapter = nn.Linear(cfg.d_model, self.d_inner, bias=False)

        self.readout_proj = nn.Linear(self.d, self.d_inner, bias=False)
        self.out_proj = nn.Linear(self.d_inner, cfg.d_model, bias=False)

        self.cleanup_T: float = 1.0
        self._last_aux_logits: list[torch.Tensor] | None = None
        self._last_sym_logits: list[torch.Tensor] | None = None

    def _self_keys(self) -> torch.Tensor:
        E = self.embed_ref.weight
        return self.k_proj(F.silu(self.bus_adapter(E)))

    def _delta_scan(self, k, v, beta):
        """Sequential DeltaNet scan.
        k, v : [B, L, d]
        beta : [B, L]
        Returns M_all : [B, L, d, d] where M_all[:, t] = M_t (state after t-th write).
        """
        B, L, d = k.shape
        M = torch.zeros(B, d, d, device=k.device, dtype=k.dtype)
        M_all = torch.zeros(B, L, d, d, device=k.device, dtype=k.dtype)
        for t in range(L):
            k_t = k[:, t]                                    # [B, d]
            v_t = v[:, t]                                    # [B, d]
            b_t = beta[:, t].unsqueeze(-1)                   # [B, 1]
            # Delta: u_t = v_t - M^T @ k_t
            Mt_k = torch.einsum('bij,bi->bj', M, k_t)         # M^T @ k_t = [B, d]
            u_t = v_t - Mt_k
            # M += β · k_t ⊗ u_t
            M = M + b_t.unsqueeze(-1) * torch.einsum('bi,bj->bij', k_t, u_t)
            M_all[:, t] = M
        return M_all

    def _delta_scan_protected(self, k, v, beta_logit):
        """Delta scan with memory-aware β gate (Fable 2026-08-14).

        β_t = σ(logit_t − w_p · ‖M^T k_t‖² / d)

        Novelty gate: penalize writes where memory already has energy at k_t,
        preventing delta-rule erasure on repeated/collided keys.
        Normalize by d so w_p scale is invariant to d_key.

        k, v       : [B, L, d]
        beta_logit : [B, L] (pre-sigmoid)
        Returns M_all + also stores a diagnostic β trajectory on self._last_beta.
        """
        B, L, d = k.shape
        M = torch.zeros(B, d, d, device=k.device, dtype=k.dtype)
        M_all = torch.zeros(B, L, d, d, device=k.device, dtype=k.dtype)
        beta_track = torch.zeros(B, L, device=k.device, dtype=k.dtype)
        for t in range(L):
            k_t = k[:, t]                                    # [B, d]
            v_t = v[:, t]                                    # [B, d]
            Mt_k = torch.einsum('bij,bi->bj', M, k_t)         # [B, d]
            penalty = (Mt_k * Mt_k).sum(dim=-1) / d           # [B], ||M^T k||^2 / d
            beta_t = torch.sigmoid(beta_logit[:, t] - self.protect_w * penalty)   # [B]
            b_t = beta_t.unsqueeze(-1)                       # [B, 1]
            beta_track[:, t] = beta_t
            u_t = v_t - Mt_k
            M = M + b_t.unsqueeze(-1) * torch.einsum('bi,bj->bij', k_t, u_t)
            M_all[:, t] = M
        self._last_beta = beta_track
        return M_all

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        xz = self.in_proj(x)
        x_workspace, z = xz.chunk(2, dim=-1)
        x_conv = self.conv1d(x_workspace.transpose(1, 2))[..., :L].transpose(1, 2)
        x_conv = F.silu(x_conv)

        q = self.q_proj(x_conv)
        k = self.k_proj(x_conv)
        v = self.v_proj(x_conv)
        beta_logit = self.beta_proj(x_conv).squeeze(-1)                  # [B, L] pre-sigmoid

        # Optional: normalize k for cleaner delta arithmetic
        k_n = F.normalize(k, dim=-1)

        if self.cfg.use_protect_gate:
            M_all = self._delta_scan_protected(k_n, v, beta_logit)
        else:
            beta = torch.sigmoid(beta_logit)
            M_all = self._delta_scan(k_n, v, beta)

        E = self.embed_ref.weight
        self_keys = self._self_keys()

        y = q
        aux_logits_per_hop: list[torch.Tensor] = []
        sym_logits_per_hop: list[torch.Tensor] = []
        for h in range(self.K_chain):
            y = torch.einsum('blij,bli->blj', M_all, y)
            y = self.chain_norms[h](y)

            if h < self.K_chain - 1:
                u = self.up_proj(y)
                logits_h = torch.einsum('vd,bld->blv', E, u) / self.cleanup_T
                aux_logits_per_hop.append(logits_h)
                probs = F.softmax(logits_h, dim=-1)
                h_br = torch.einsum('blv,vd->bld', probs, E)
                y = self.k_proj(F.silu(self.bus_adapter(h_br)))
                sym_logits = torch.einsum('bld,vd->blv', y, self_keys) / self.cleanup_T
                sym_logits_per_hop.append(sym_logits)

        self._last_aux_logits = aux_logits_per_hop
        self._last_sym_logits = sym_logits_per_hop

        y = self.readout_proj(y)
        y = y * F.silu(z)
        return self.out_proj(y)


class DeltaBusBlock(nn.Module):
    def __init__(self, cfg: DeltaBusConfig, embed_matrix: nn.Embedding):
        super().__init__()
        self.norm = RMSNorm(cfg.d_model)
        self.mixer = DeltaBusMixer(cfg, embed_matrix)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mixer(self.norm(x))


class DeltaBusStack(nn.Module):
    def __init__(self, vocab_size: int, cfg: DeltaBusConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([DeltaBusBlock(cfg, self.embed) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

    def set_cleanup_T(self, T: float):
        for blk in self.blocks:
            blk.mixer.cleanup_T = T

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return self.lm_head(x)


def build_model(vocab_size: int, cfg: DeltaBusConfig | None = None) -> DeltaBusStack:
    cfg = cfg or DeltaBusConfig()
    return DeltaBusStack(vocab_size=vocab_size, cfg=cfg)
