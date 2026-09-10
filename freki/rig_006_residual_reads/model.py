"""ResRead — FREKI variant with residual (summed) reads instead of chained-final.

Motivation. The K_chain=3 depth wall (rig_003 K_chain=3 = 0/5, rig_005 v1
and v2 = 0/5, rig_005 aux-v1 and aux-v2 = 0/5) is diagnosed as an
AND-condition on training signal: with the read

    y = RMSNorm(M · RMSNorm(M · RMSNorm(M · q)))

all three M-lookups must produce useful signal before any k-hop accuracy
leaves chance. No partial-credit scaffold like the k1-first pattern that
lets FREKI-2 escape chance on 2-hop.

Fix under test here. Sum contributions from every read depth so 1-hop
signal reaches the output on its own:

    y_1 = RMSNorm_1(M · q)
    y_2 = RMSNorm_2(M · y_1)
    y_3 = RMSNorm_3(M · y_2)
    y   = readout(sum(y_1 + y_2 + y_3))

With residual reads, depth-1 signal contributes directly to the final
output. If M learns even a partial hop-1 lookup, k1 accuracy can leave
chance immediately — which then supplies gradient signal for M to also
sharpen hops 2 and 3. Same math machinery as FREKI-K_chain=3, one
summation replacing the chain's final output selection.

This is a single-M design (like FREKI). HopStrat's K per-hop banks are
orthogonal and independently failed the same wall — first we isolate the
read-structure change.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ResReadConfig:
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


class ResReadMixer(nn.Module):
    def __init__(self, cfg: ResReadConfig):
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
        self.beta_proj = nn.Linear(self.d_inner, 1, bias=True)
        nn.init.constant_(self.beta_proj.bias, cfg.beta_bias_init)

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

        # Vectorized additive write — single M shared across all read steps.
        outer = torch.einsum('bl,bli,blj->blij', beta, k, v)            # [B, L, d, d]
        M_all = torch.cumsum(outer, dim=1)                              # [B, L, d, d]

        # Chained-but-summed read. y_h = RMSNorm(M · y_{h-1}), y_0 = q.
        # Sum every y_h into y_sum — each depth contributes independently.
        y = q
        y_sum = torch.zeros_like(q)
        for h in range(self.K_chain):
            y = torch.einsum('blij,bli->blj', M_all, y)                 # Mᵀ · y_{h-1}
            y = self.chain_norms[h](y)
            y_sum = y_sum + y

        y = self.readout_proj(y_sum)
        y = y * F.silu(z)
        return self.out_proj(y)


class ResReadBlock(nn.Module):
    def __init__(self, cfg: ResReadConfig):
        super().__init__()
        self.norm = RMSNorm(cfg.d_model)
        self.mixer = ResReadMixer(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mixer(self.norm(x))


class ResReadStack(nn.Module):
    def __init__(self, vocab_size: int, cfg: ResReadConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([ResReadBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return self.lm_head(x)


def build_model(vocab_size: int, cfg: ResReadConfig | None = None) -> ResReadStack:
    cfg = cfg or ResReadConfig()
    return ResReadStack(vocab_size=vocab_size, cfg=cfg)
