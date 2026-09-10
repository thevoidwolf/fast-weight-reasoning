"""One training run of one arm (baseline or filler), shared by both experiments.

The two arms are identical except the filler arm splices a ramped, random number
of filler tokens into the target-loss samples during training (the baseline arm
never sees filler at train time). Both use the same two-task joint loss
(0.5 * K1-probe + 0.5 * K-hop target), the same eager-closure chunked mixer, and
the same eval metrics. Recipe constants match the research drivers (rig_129
baseline / rig_136 filler).
"""
from __future__ import annotations

import contextlib

import torch
import torch.nn.functional as F

from .model import MixerConfig, NovelStack
from .nhop_task import _sample_Nhop_factorized_truncated_injective, insert_filler
from .probe import div_norm, occlusion_probe
from . import tasks
from .util import _cosine_lr, count_params, pick_device, seed_all, timer, write_result

STEPS = 15_000
FILLER_MAX = 200
EVAL_N = 200
WARMUP_STEPS = 2000
RAMP_STEPS = 2000
DIV_N = 2000


def k_lists(k_hops: int):
    """(target, probe) candidate-count ladders for a K-hop chain.
    target = [4]*(K-1)+[8]; probe = [1]+[4]*(K-2)+[8]."""
    assert k_hops >= 2
    target = [4] * (k_hops - 1) + [8]
    probe = [1] + [4] * (k_hops - 2) + [8]
    return target, probe


def _filler_cap_at(step, warmup, ramp, fmax):
    if step < warmup:
        return 0
    if step >= warmup + ramp:
        return fmax
    return int(round(fmax * (step - warmup) / ramp))


def _autocast(device):
    if device == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def run_arm(*, recipe: str, k_hops: int = 6, seed: int = 0, steps: int = STEPS,
            batch: int | None = None, filler_max: int = FILLER_MAX,
            eval_n: int = EVAL_N, warmup_steps: int = WARMUP_STEPS,
            ramp_steps: int = RAMP_STEPS, d_model: int = 256,
            eval_every: int = 1500, quiet: bool = False,
            return_model: bool = False):
    assert recipe in ("baseline", "filler")
    device = pick_device()
    torch.backends.cuda.matmul.allow_tf32 = True
    seed_all(seed)

    if batch is None:
        batch = 28 if recipe == "filler" else 48

    task_cfg = tasks.TaskCfg(k_facts_long=8, seed=seed)
    cfg = MixerConfig(d_model=d_model, d_key=32, d_value=32)
    model = NovelStack(task_cfg.vocab_size, cfg,
                       mixer_name="eager_closure_fixed_chunked",
                       n_layers=3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01,
                            betas=(0.9, 0.95))

    K_TARGET, K_PROBE = k_lists(k_hops)
    g_probe = torch.Generator(device="cpu").manual_seed(seed + 100)
    g_target = torch.Generator(device="cpu").manual_seed(seed + 200)
    g_filler = torch.Generator(device="cpu").manual_seed(seed + 300)

    n_params = count_params(model)
    chance = 1.0 / task_cfg.n_values
    if not quiet:
        print(f"[{recipe} K_hops={k_hops} seed={seed}] params={n_params:,} "
              f"batch={batch} steps={steps} K_target={K_TARGET} chance={chance:.4f} "
              f"device={device}", flush=True)

    def sample(K, g):
        return _sample_Nhop_factorized_truncated_injective(K, task_cfg, batch, g, device)

    curve = []
    with timer() as t:
        for step in range(steps):
            lr = _cosine_lr(step, warmup=200, total=steps, base=3e-4, floor=3e-5)
            for pg in opt.param_groups:
                pg["lr"] = lr
            with _autocast(device):
                r_p = sample(K_PROBE, g_probe)
                probe_loss = F.cross_entropy(model(r_p[0])[:, -1], r_p[1])

                r_t = sample(K_TARGET, g_target)
                if recipe == "filler":
                    cap = _filler_cap_at(step, warmup_steps, ramp_steps, filler_max)
                    if cap > 0:
                        N = int(torch.randint(0, cap + 1, (1,), generator=g_filler).item())
                        seq_t = insert_filler(r_t[0], N) if N > 0 else r_t[0]
                    else:
                        seq_t = r_t[0]
                else:
                    seq_t = r_t[0]
                target_loss = F.cross_entropy(model(seq_t)[:, -1], r_t[1])
                loss = 0.5 * probe_loss + 0.5 * target_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if (step + 1) % eval_every == 0 or (step + 1) == steps:
                with _autocast(device), torch.no_grad():
                    r0 = sample(K_TARGET, torch.Generator(device="cpu").manual_seed(seed + 900))
                    acc0 = float((model(r0[0])[:, -1].argmax(-1) == r0[1]).float().mean())
                    ra = sample(K_TARGET, torch.Generator(device="cpu").manual_seed(seed + 950))
                    seq_a = insert_filler(ra[0], eval_n)
                    accA = float((model(seq_a)[:, -1].argmax(-1) == ra[1]).float().mean())
                probe = occlusion_probe(model, task_cfg, K_TARGET, 96,
                                        torch.Generator(device="cpu").manual_seed(42),
                                        device, N=eval_n)
                dn = div_norm(model, task_cfg, K_TARGET,
                              torch.Generator(device="cpu").manual_seed(seed + 800),
                              device, N=DIV_N)
                curve.append({"step": step + 1, "target_acc_N0": acc0,
                              "target_acc_N200": accA,
                              "qtok_flip": probe["qtok_flip"],
                              "atok_flip": probe["atok_flip"],
                              "div_norm_at_N2000": dn})
                if not quiet:
                    dm = " *** DIV" if dn > 1e5 else ""
                    print(f"  step {step+1:>5d}  target_N0={acc0:.4f}  "
                          f"target_N200={accA:.4f}  qtok={probe['qtok_flip']:.4f}  "
                          f"div={dn:.2f}{dm}", flush=True)

    final = curve[-1] if curve else {}
    passed = bool(final and final["target_acc_N200"] >= 0.80
                  and final["qtok_flip"] <= 0.15
                  and final["div_norm_at_N2000"] < 1e3)
    payload = {"tag": f"{recipe}_hops{k_hops}", "recipe": recipe, "k_hops": k_hops,
               "seed": seed, "steps": steps, "batch": batch, "params": n_params,
               "chance": chance, "K_target": K_TARGET,
               "final_target_N200": final.get("target_acc_N200"),
               "final_qtok_flip": final.get("qtok_flip"),
               "final_div_norm": final.get("div_norm_at_N2000"),
               "passed": passed, "trajectory": curve,
               "wallclock_s": round(t["elapsed_s"], 3)}
    path = write_result(f"{recipe}_hops{k_hops}__s{seed}", payload)
    if not quiet:
        print(f"  -> {path.name}  pass={passed}  "
              f"target_N200={payload['final_target_N200']}", flush=True)
    if return_model:
        return payload, model, task_cfg
    return payload
