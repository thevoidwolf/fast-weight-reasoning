"""rig_009_id_homotopy — Arm 3: identity-homotopy curriculum, no aux.

Task-only fix. FREKI K_chain=3 (unmodified). Train the standard 3-hop
task but with a per-batch curriculum: with probability ρ(step), use the
real 3-hop sample; else use identity-mode where banks 2 and 3 are
self-paired (b→b, b→b) so the answer is reachable via just hop 1.

Schedule:
  ρ(step) = 0.0            for step < warm_id_steps
  ρ(step) = (step-warm_id_steps) / anneal_steps    linear ramp
  ρ(step) = 1.0            after warm_id_steps + anneal_steps

Answer semantics: identity-mode answer is bridge1 (a b-entity token,
NOT a v-token). Eval uses the standard 3-hop ladder (real mode only)
throughout — accuracy climb during identity phase is expected to be
low on the eval metric but training loss should progress.

Total effective steps at ρ=1 (real-mode-only): steps - warm_id_steps - anneal_steps.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import torch
import torch.nn.functional as F

from common.harness import RunCfg, cosine_lr, eval_ladder, left_pad
from common.tasks import TaskCfg
from rig_003_freki.model import FrekiConfig, build_model
from tasks_id import sample_id_or_real


def rho_schedule(step: int, warm_id_steps: int, anneal_steps: int) -> float:
    if step < warm_id_steps:
        return 0.0
    ramp_progress = (step - warm_id_steps) / max(1, anneal_steps)
    return min(1.0, ramp_progress)


def train_one_arm3(
    build_model_fn, seed: int, cfg: RunCfg, hs_cfg: FrekiConfig,
    out_dir: Path, tag: str,
    warm_id_steps: int, anneal_steps: int,
):
    log = lambda s: print(s, flush=True)

    torch.manual_seed(seed)
    task_cfg = TaskCfg(seed=seed)
    model = build_model_fn(task_cfg.vocab_size).to(cfg.device)
    model._hrs_autocast = cfg.autocast
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay, betas=cfg.betas)

    train_tasks = [(K1_r, cfg.K2) for K1_r in range(1, cfg.K1 + 1)]
    train_gens = {tk: torch.Generator(device="cpu").manual_seed(seed + 1000 + i)
                  for i, tk in enumerate(train_tasks)}

    log(f"[{tag}] params={n_params:,} tasks={train_tasks} steps={cfg.steps} "
        f"id_homotopy(warm={warm_id_steps}, anneal={anneal_steps})")
    curve = []
    start = time.time()

    for step in range(cfg.steps):
        lr = cosine_lr(step, cfg.warmup, cfg.steps, cfg.lr, cfg.lr_floor)
        for pg in opt.param_groups:
            pg["lr"] = lr

        rho = rho_schedule(step, warm_id_steps, anneal_steps)

        tk = train_tasks[step % len(train_tasks)]
        k1r = tk[0]
        k2_use = max(cfg.K2, k1r)
        k3_use = max(cfg.K3, k2_use)
        seq, tgt, _, _ = sample_id_or_real(
            K1=k1r, K2=k2_use, K3=k3_use, cfg=task_cfg, batch=cfg.batch,
            gen=train_gens[tk], rho=rho, device=cfg.device,
        )
        if cfg.pad_seq_to > 0:
            seq = left_pad(seq, cfg.pad_seq_to)

        with torch.autocast(device_type=cfg.device, dtype=torch.bfloat16,
                            enabled=(cfg.device == "cuda" and cfg.autocast)):
            logits = model(seq)
            loss = F.cross_entropy(logits[:, -1], tgt)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        if (step + 1) % cfg.eval_every == 0 or (step + 1) == cfg.steps:
            eval_K1_max = cfg.eval_K1_max if cfg.eval_K1_max > 0 else cfg.K1
            # Eval always on the real 3-hop task.
            accs = eval_ladder(model, task_cfg, eval_K1_max, cfg.K2,
                               cfg.eval_batch_size, cfg.eval_batches,
                               cfg.device, seed + 10_000,
                               pad_seq_to=cfg.pad_seq_to,
                               hops=cfg.hops, K3=cfg.K3)
            curve.append({
                "step": step + 1,
                "loss": float(loss.item()),
                "rho": rho,
                "acc": accs,
            })
            acc_str = " ".join(f"{k}={v:.3f}" for k, v in accs.items())
            log(f"  step {step + 1:>4d} loss {loss.item():.3f} ρ {rho:.2f} "
                f"lr {lr:.1e} {acc_str}")

    wallclock = time.time() - start
    final_acc = curve[-1]["acc"]
    payload = {
        "tag": tag, "rig": "rig009_id_homotopy", "seed": seed,
        "params": n_params, "cfg": asdict(cfg),
        "curriculum": {
            "warm_id_steps": warm_id_steps,
            "anneal_steps": anneal_steps,
        },
        "vocab_size": task_cfg.vocab_size,
        "train_tasks": [list(t) for t in train_tasks],
        "final_acc": final_acc,
        "wallclock_s": round(wallclock, 3),
        "curve": curve,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{tag}.json").write_text(json.dumps(payload, indent=2))
    log(f"[{tag}] done in {wallclock:.1f}s  final k{cfg.K1}={final_acc.get(f'k{cfg.K1}'):.3f}")
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--K1", type=int, default=4)
    ap.add_argument("--K2", type=int, default=4)
    ap.add_argument("--K3", type=int, default=4)
    ap.add_argument("--K-chain", type=int, default=3)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--d-key", type=int, default=32)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--beta-bias-init", type=float, default=-0.5)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--warm-id-steps", type=int, default=2000,
                    help="Steps at pure identity mode before annealing")
    ap.add_argument("--anneal-steps", type=int, default=4000,
                    help="Steps over which ρ linearly ramps 0→1")
    ap.add_argument("--tag-prefix", default="rig009_arm3")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = RunCfg(
        K1=args.K1, K2=args.K2, steps=args.steps, batch=args.batch,
        mode="mixed", eval_every=args.eval_every, device=args.device,
        hops=3, K3=args.K3,
    )
    hs_cfg = FrekiConfig(
        d_model=args.d_model, d_key=args.d_key, n_layers=args.n_layers,
        beta_bias_init=args.beta_bias_init, K_chain=args.K_chain,
    )
    out_dir = HERE / "outputs"

    def make_builder(hs_cfg):
        return lambda vocab_size: build_model(vocab_size, hs_cfg)

    per_seed = []
    for seed in args.seeds:
        payload = train_one_arm3(
            build_model_fn=make_builder(hs_cfg), seed=seed,
            cfg=cfg, hs_cfg=hs_cfg, out_dir=out_dir,
            tag=f"{args.tag_prefix}_s{seed}",
            warm_id_steps=args.warm_id_steps,
            anneal_steps=args.anneal_steps,
        )
        per_seed.append(payload)

    summary = {
        "rig": "rig009_id_homotopy",
        "seeds": args.seeds,
        "hs_cfg": hs_cfg.__dict__,
        "run_cfg": cfg.__dict__,
        "curriculum": {
            "warm_id_steps": args.warm_id_steps,
            "anneal_steps": args.anneal_steps,
        },
        "final_k4_per_seed": [p["final_acc"].get(f"k{args.K1}") for p in per_seed],
        "final_full_per_seed": [p["final_acc"] for p in per_seed],
        "wallclock_s_per_seed": [p["wallclock_s"] for p in per_seed],
    }
    (out_dir / f"{args.tag_prefix}_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
