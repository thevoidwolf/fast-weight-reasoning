"""rig_008 — FREKI + embedding-anchored cleanup bridge (Fable's #1, arm 2).

Architectural change relative to rig_003 FREKI: between chain steps,
snap the intermediate y_h to the token simplex via a softmax over the
tied embedding table, then map back to key-space:

    u_h      = up_proj(y_h)                       # d_key → d_model
    logits_h = E · u_h / T                        # E = tied embed [vocab, d_model]
    y'_h     = down_proj(E^T · softmax(logits_h)) # snap to simplex, back to d_key

The softmax cleans up rank-limited crosstalk (VSA-style cleanup memory).
Aux supervision uses `logits_h @ ATOK` directly as the aux CE input, so
the supervised tensor IS the forward-path tensor — no bypass.

Temperature T follows an anneal schedule: soft at init (~5.0), sharp
by mid-training (~1.0). Model itself exposes a settable `T` attribute
that the training loop updates each step.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CleanupConfig:
    d_model: int = 256
    d_key: int = 32
    expand: int = 2
    d_conv: int = 4
    beta_bias_init: float = -0.5
    n_layers: int = 3
    K_chain: int = 3


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * x / rms


class CleanupMixer(nn.Module):
    def __init__(self, cfg: CleanupConfig, embed_matrix: nn.Embedding):
        super().__init__()
        self.cfg = cfg
        self.d = cfg.d_key
        self.d_inner = cfg.d_model * cfg.expand
        self.K_chain = cfg.K_chain
        self.embed_ref = embed_matrix   # weight-tied to lm_head via stack init

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

        self.chain_norms = nn.ModuleList([RMSNorm(self.d) for _ in range(self.K_chain)])

        # Cleanup bridge — one shared instance across chain steps.
        # (Per-hop would double these; sharing keeps params minimal.)
        self.up_proj   = nn.Linear(self.d, cfg.d_model, bias=False)
        self.down_proj = nn.Linear(cfg.d_model, self.d, bias=False)

        self.readout_proj = nn.Linear(self.d, self.d_inner, bias=False)
        self.out_proj = nn.Linear(self.d_inner, cfg.d_model, bias=False)

        # Settable at forward time by training loop:
        self.cleanup_T: float = 1.0
        # For aux capture:
        self._last_aux_logits: list[torch.Tensor] | None = None

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

        outer = torch.einsum('bl,bli,blj->blij', beta, k, v)
        M_all = torch.cumsum(outer, dim=1)

        E = self.embed_ref.weight    # [vocab, d_model]

        y = q
        aux_logits_per_hop: list[torch.Tensor] = []
        for h in range(self.K_chain):
            y = torch.einsum('blij,bli->blj', M_all, y)               # Mᵀ · y
            y = self.chain_norms[h](y)

            # Cleanup bridge — skip on FINAL hop (main lm_head decodes y_K directly).
            if h < self.K_chain - 1:
                u = self.up_proj(y)                                    # [B, L, d_model]
                logits_h = torch.einsum('vd,bld->blv', E, u) / self.cleanup_T
                aux_logits_per_hop.append(logits_h)                    # [B, L, V]
                probs = F.softmax(logits_h, dim=-1)                    # [B, L, V]
                # y' = down_proj(E^T · probs)
                snapped = torch.einsum('blv,vd->bld', probs, E)        # [B, L, d_model]
                y = self.down_proj(snapped)                            # [B, L, d]

        self._last_aux_logits = aux_logits_per_hop

        y = self.readout_proj(y)
        y = y * F.silu(z)
        return self.out_proj(y)


class CleanupBlock(nn.Module):
    def __init__(self, cfg: CleanupConfig, embed_matrix: nn.Embedding):
        super().__init__()
        self.norm = RMSNorm(cfg.d_model)
        self.mixer = CleanupMixer(cfg, embed_matrix)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mixer(self.norm(x))


class CleanupStack(nn.Module):
    def __init__(self, vocab_size: int, cfg: CleanupConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([CleanupBlock(cfg, self.embed) for _ in range(cfg.n_layers)])
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


def build_model(vocab_size: int, cfg: CleanupConfig | None = None) -> CleanupStack:
    cfg = cfg or CleanupConfig()
    return CleanupStack(vocab_size=vocab_size, cfg=cfg)
