"""Eval ladder for §49 task-shape arms. Uses `sample_3hop_shape(arm, ...)`
in place of the baseline 3-hop sampler; otherwise identical to
`common.harness.eval_ladder`.
"""
from __future__ import annotations

import torch
from torch import nn

from common.harness import left_pad
from common.tasks import TaskCfg
from common.tasks_shape import sample_3hop_shape, sample_3hop_reuse_arm1


def eval_ladder_shape(model: nn.Module, task_cfg: TaskCfg, K1_max: int,
                      K2: int, K3: int,
                      batch: int, n_batches: int, device: str,
                      eval_seed: int, arm: str,
                      pad_seq_to: int = 0,
                      task_family: str = "shape",
                      n_reuse: int = 2) -> dict:
    model.eval()
    accs = {}
    for K1 in range(1, K1_max + 1):
        gen = torch.Generator(device="cpu").manual_seed(eval_seed + K1)
        correct = total = 0
        k2_use = max(K2, K1)
        k3_use = max(K3, k2_use)
        for _ in range(n_batches):
            if task_family == "shape":
                seq, tgt, *_ = sample_3hop_shape(
                    arm, K1, k2_use, k3_use, task_cfg, batch, gen, device=device)
            elif task_family == "reuse":
                seq, tgt, *_ = sample_3hop_reuse_arm1(
                    K1, k2_use, k3_use, task_cfg, batch, gen, device=device,
                    n_reuse=n_reuse)
            else:
                raise ValueError(f"unknown task_family={task_family!r}")
            if pad_seq_to > 0:
                seq = left_pad(seq, pad_seq_to)
            with torch.autocast(device_type=device, dtype=torch.bfloat16,
                                enabled=(device == "cuda"
                                         and getattr(model, "_hrs_autocast", True))):
                pred = model(seq)[:, -1].argmax(-1)
            correct += (pred == tgt).sum().item()
            total += batch
        accs[f"k{K1}"] = correct / total
    model.train()
    return accs
