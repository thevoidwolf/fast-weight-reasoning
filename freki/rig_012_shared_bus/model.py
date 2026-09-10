"""rig_012 — Shared-bus cleanup (Fable's H1+H6 fix).

Diagnosis (rig_010 at n_entities=512, contrastive-aux + d_key=64):
    aux CE = 1.82 (below 12-way chance 2.48) but k4 = 0.28 (chance).
    Aux found a subspace the forward path doesn't consume.

Root cause (Fable): cleanup output is only STATISTICALLY coupled to
next-hop k_proj(bridge) — never structurally. Aux is CE on logits
BEFORE down_proj, so any anisotropy down_proj introduces is invisible
to aux. At n=128 gradient signal aligns them anyway; at n=512
Johnson-Lindenstrauss bites the 32-dim k-subspace, aux finds a
different slice of the 256-dim hidden that separates classes cleanly,
forward holds a broken pointer.

Fix: replace down_proj with the SAME projection path the write uses.

Old cleanup:
    u        = up_proj(y)                        # d_key → d_model
    logits_h = E · u / T
    p        = softmax(logits_h)
    y'       = down_proj(E^T · p)                # d_model → d_key, DECOUPLED from writes

New (shared-bus) cleanup:
    u        = up_proj(y)
    logits_h = E · u / T
    p        = softmax(logits_h)
    h_br     = E^T · p                           # in embedding space, d_model
    y'       = k_proj(silu(bus_adapter(h_br)))   # same k_proj that writes use

The write path is `k_i = k_proj(silu(conv1d(in_proj_x(x))[i]))`. We skip
conv1d for cleanup (it mixes with neighbors of a specific position; for
the "self-key" of the bridge token, we approximate). bus_adapter is a
small d_model→d_inner shim; it and in_proj_x should span the same
column space if the model wants alignment.

Symmetric-consistency aux: at each hop h, compute
    self_keys = k_proj(silu(bus_adapter(E)))     # [vocab, d_key]
    sym_logits = y'_h @ self_keys^T / T          # [B, L, vocab]
    sym_ce = CE(sym_logits @ ATOK, bridge)
This forces the cleaned output to be re-identifiable as the same token
through the actual read path. Not just decodable through up_proj.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SharedBusConfig:
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


class SharedBusMixer(nn.Module):
    def __init__(self, cfg: SharedBusConfig, embed_matrix: nn.Embedding):
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

        self.chain_norms = nn.ModuleList([RMSNorm(self.d) for _ in range(self.K_chain)])

        # Shared-bus cleanup: up_proj → softmax → E^T · p → bus_adapter → silu → k_proj.
        self.up_proj = nn.Linear(self.d, cfg.d_model, bias=False)
        self.bus_adapter = nn.Linear(cfg.d_model, self.d_inner, bias=False)

        self.readout_proj = nn.Linear(self.d, self.d_inner, bias=False)
        self.out_proj = nn.Linear(self.d_inner, cfg.d_model, bias=False)

        self.cleanup_T: float = 1.0
        # Aux capture:
        self._last_aux_logits: list[torch.Tensor] | None = None
        self._last_sym_logits: list[torch.Tensor] | None = None

    def _self_keys(self) -> torch.Tensor:
        """Compute the k-projection of every vocab embedding as its 'self-key'.
        [vocab, d_key]. Reused for symmetric-consistency aux and for cleanup.
        """
        E = self.embed_ref.weight                          # [V, d_model]
        return self.k_proj(F.silu(self.bus_adapter(E)))    # [V, d_key]

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

        E = self.embed_ref.weight
        self_keys = self._self_keys()                       # [V, d_key]

        y = q
        aux_logits_per_hop: list[torch.Tensor] = []
        sym_logits_per_hop: list[torch.Tensor] = []
        for h in range(self.K_chain):
            y = torch.einsum('blij,bli->blj', M_all, y)     # Mᵀ · y
            y = self.chain_norms[h](y)

            if h < self.K_chain - 1:
                u = self.up_proj(y)                          # [B, L, d_model]
                logits_h = torch.einsum('vd,bld->blv', E, u) / self.cleanup_T
                aux_logits_per_hop.append(logits_h)
                probs = F.softmax(logits_h, dim=-1)          # [B, L, V]
                h_br = torch.einsum('blv,vd->bld', probs, E) # [B, L, d_model]
                # Shared bus: same silu(k_proj(bus_adapter(·))) path as writes use.
                y = self.k_proj(F.silu(self.bus_adapter(h_br)))
                # Symmetric-consistency aux: does the cleaned y re-identify as bridge?
                sym_logits = torch.einsum('bld,vd->blv', y, self_keys) / self.cleanup_T
                sym_logits_per_hop.append(sym_logits)

        self._last_aux_logits = aux_logits_per_hop
        self._last_sym_logits = sym_logits_per_hop

        y = self.readout_proj(y)
        y = y * F.silu(z)
        return self.out_proj(y)


class SharedBusBlock(nn.Module):
    def __init__(self, cfg: SharedBusConfig, embed_matrix: nn.Embedding):
        super().__init__()
        self.norm = RMSNorm(cfg.d_model)
        self.mixer = SharedBusMixer(cfg, embed_matrix)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mixer(self.norm(x))


class SharedBusStack(nn.Module):
    def __init__(self, vocab_size: int, cfg: SharedBusConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([SharedBusBlock(cfg, self.embed) for _ in range(cfg.n_layers)])
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


def build_model(vocab_size: int, cfg: SharedBusConfig | None = None) -> SharedBusStack:
    cfg = cfg or SharedBusConfig()
    return SharedBusStack(vocab_size=vocab_size, cfg=cfg)
