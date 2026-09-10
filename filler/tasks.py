"""Task primitives for the filler chapter.

The control-token
ids, the TaskCfg, and the entity/value pair sampler. The multi-hop chain sampler
that builds on these lives in `nhop_task.py`.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

# Control-token ids. Entities occupy [N_CONTROL, N_CONTROL+n_entities); values
# occupy the block after that.
PAD, BOS, FACT, SEP, QTOK, ATOK, EOS = 0, 1, 2, 3, 4, 5, 6
N_CONTROL = 7


@dataclass
class TaskCfg:
    n_entities: int = 128
    n_values: int = 128
    k_facts_long: int = 32
    seed: int = 0

    @property
    def vocab_size(self) -> int:
        return N_CONTROL + self.n_entities + self.n_values

    def entity_ids(self):
        return range(N_CONTROL, N_CONTROL + self.n_entities)

    def value_ids(self):
        return range(N_CONTROL + self.n_entities,
                     N_CONTROL + self.n_entities + self.n_values)


def _sample_pairs(cfg: TaskCfg, batch: int, k: int, gen: torch.Generator):
    """k distinct entities per row, and k values (with replacement) per row."""
    ents = torch.stack([
        torch.randperm(cfg.n_entities, generator=gen)[:k] + N_CONTROL
        for _ in range(batch)
    ])
    vals = torch.randint(cfg.n_values, (batch, k), generator=gen) \
        + (N_CONTROL + cfg.n_entities)
    return ents, vals
