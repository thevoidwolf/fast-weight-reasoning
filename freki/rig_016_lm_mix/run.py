"""rig_016 — LM-mix formation test (Fable's proposal).

Same DeltaNet + shared-bus + sym recipe as rig_013. Each mini-batch is
now a mix: `chain_frac` (default 0.15) of rows are true 3-hop chains
under the standard sampler; the rest are random-token filler wrapped
in the same FACT ... QTOK q ATOK skeleton. Main + aux losses are
masked to chain rows only.

Question: does the mechanism form when only 15% of rows carry real
chain supervision? If yes, this is the strongest evidence to date that
the recipe transfers to sparse-supervision LM training; if not, the
next problem is a self-supervised aux surrogate.

Decision rule (per Fable): k4 ≥ 0.90 for ≥2/3 seeds on chain-slot eval
→ green-light small-LM run. 0/3 → recipe needs dense task supervision.
1/3 → seed-lottery is back; instrument β and the crash-together
signature.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "rig_013_delta_write"))

import torch
import torch.nn.functional as F

from common.harness import RunCfg, cosine_lr, eval_ladder, left_pad
from common.tasks import TaskCfg, N_CONTROL
from common.tasks_lm_mix import sample_3hop_lm_mix
from model import DeltaBusConfig as SharedBusConfig, build_model


def _contrastive_aux_mask(seq: torch.Tensor, vocab_size: int,
                          n_entities: int) -> torch.Tensor:
    B, L = seq.shape
    device = seq.device
    mask = torch.zeros(B, vocab_size, device=device, dtype=torch.bool)
    mask.scatter_(1, seq, True)
    entity_lo, entity_hi = N_CONTROL, N_CONTROL + n_entities
    keep = torch.zeros(vocab_size, device=device, dtype=torch.bool)
    keep[entity_lo:entity_hi] = True
    mask = mask & keep.unsqueeze(0)
    return mask


def train_one_lm_mix(
    build_model_fn, seed: int, cfg: RunCfg, hs_cfg: SharedBusConfig,
    out_dir: Path, tag: str,
    chain_frac: float,
    aux_lambda_init: float, aux_decay_tau: float,
    sym_lambda_init: float,
    T_init: float, T_final: float, aux_layer: int,
    cfg_n_entities: int = 512, cfg_n_values: int = 128,
    contrastive_aux: bool = True,
    lm_filler_lambda: float = 0.0,
):
    log = lambda s: print(s, flush=True)
    torch.manual_seed(seed)
    task_cfg = TaskCfg(seed=seed, n_entities=cfg_n_entities, n_values=cfg_n_values)
    model = build_model_fn(task_cfg.vocab_size).to(cfg.device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay, betas=cfg.betas)

    train_tasks = [(K1_r, cfg.K2) for K1_r in range(1, cfg.K1 + 1)]
    train_gens = {tk: torch.Generator(device="cpu").manual_seed(seed + 1000 + i)
                  for i, tk in enumerate(train_tasks)}

    log(f"[{tag}] params={n_params:,} tasks={train_tasks} steps={cfg.steps} "
        f"chain_frac={chain_frac} aux(λ_init={aux_lambda_init}, "
        f"τ={aux_decay_tau}) sym(λ={sym_lambda_init}) "
        f"n_ent={cfg_n_entities} d_key={hs_cfg.d_key}")

    curve = []
    start = time.time()
    for step in range(cfg.steps):
        lr = cosine_lr(step, cfg.warmup, cfg.steps, cfg.lr, cfg.lr_floor)
        for pg in opt.param_groups:
            pg["lr"] = lr

        progress = step / max(1, cfg.steps - 1)
        T = T_init + (T_final - T_init) * progress
        model.set_cleanup_T(T)

        tk = train_tasks[step % len(train_tasks)]
        k1r = tk[0]
        k2_use = max(cfg.K2, k1r)
        k3_use = max(cfg.K3, k2_use)
        seq, tgt, bridge1, bridge2, loss_mask, _is_chain = sample_3hop_lm_mix(
            k1r, k2_use, k3_use, task_cfg, cfg.batch, train_gens[tk],
            chain_frac=chain_frac, device=cfg.device)
        if cfg.pad_seq_to > 0:
            seq = left_pad(seq, cfg.pad_seq_to)

        aux_lambda = aux_lambda_init * math.exp(-step / aux_decay_tau)
        sym_lambda = sym_lambda_init * math.exp(-step / aux_decay_tau)
        apply_aux = aux_lambda >= 1e-6

        n_chain = int(loss_mask.sum().item())

        with torch.autocast(device_type=cfg.device, dtype=torch.bfloat16,
                            enabled=(cfg.device == "cuda" and cfg.autocast)):
            logits = model(seq)

            if n_chain == 0:
                loss = logits[:, -1].sum() * 0.0
                main_val = float("nan")
                aux_val = float("nan")
                sym_val = float("nan")
            else:
                per_row_ce = F.cross_entropy(logits[:, -1], tgt, reduction="none")
                main_loss = (per_row_ce * loss_mask.float()).sum() / max(1, n_chain)
                main_val = float(main_loss.item())

                if apply_aux:
                    aux_stack = model.blocks[aux_layer].mixer._last_aux_logits
                    sym_stack = model.blocks[aux_layer].mixer._last_sym_logits
                    aux_targets = [bridge1, bridge2]

                    if contrastive_aux:
                        mask = _contrastive_aux_mask(seq, task_cfg.vocab_size,
                                                     cfg_n_entities)
                        neg_inf = torch.finfo(aux_stack[0].dtype).min

                    aux_losses, sym_losses = [], []
                    for h in range(len(aux_stack)):
                        lo = aux_stack[h][:, -1, :]
                        so = sym_stack[h][:, -1, :]
                        if contrastive_aux:
                            lo = lo.masked_fill(~mask, neg_inf)
                            so = so.masked_fill(~mask, neg_inf)
                        per_row_a = F.cross_entropy(lo, aux_targets[h],
                                                    reduction="none")
                        per_row_s = F.cross_entropy(so, aux_targets[h],
                                                    reduction="none")
                        aux_losses.append((per_row_a * loss_mask.float()).sum() / max(1, n_chain))
                        sym_losses.append((per_row_s * loss_mask.float()).sum() / max(1, n_chain))
                    aux_mean = sum(aux_losses) / len(aux_losses)
                    sym_mean = sum(sym_losses) / len(sym_losses)
                    loss = main_loss + aux_lambda * aux_mean + sym_lambda * sym_mean
                    aux_val = float(aux_mean.item())
                    sym_val = float(sym_mean.item())
                else:
                    loss = main_loss
                    aux_val = float("nan")
                    sym_val = float("nan")

        if lm_filler_lambda > 0.0:
            filler_mask = ~loss_mask
            if int(filler_mask.sum().item()) > 0:
                V = logits.shape[-1]
                fl_logits = logits[filler_mask, :-1, :].reshape(-1, V)
                fl_tgt = seq[filler_mask, 1:].reshape(-1)
                lm_filler_loss = F.cross_entropy(fl_logits, fl_tgt)
                loss = loss + lm_filler_lambda * lm_filler_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        if (step + 1) % cfg.eval_every == 0 or (step + 1) == cfg.steps:
            eval_K1_max = cfg.eval_K1_max if cfg.eval_K1_max > 0 else cfg.K1
            accs = eval_ladder(model, task_cfg, eval_K1_max, cfg.K2,
                               cfg.eval_batch_size, cfg.eval_batches,
                               cfg.device, seed + 10_000,
                               pad_seq_to=cfg.pad_seq_to,
                               hops=cfg.hops, K3=cfg.K3)
            curve.append({
                "step": step + 1, "loss": main_val,
                "aux_loss": aux_val, "sym_loss": sym_val,
                "aux_lambda": aux_lambda, "sym_lambda": sym_lambda,
                "cleanup_T": T, "n_chain": n_chain, "acc": accs,
            })
            acc_str = " ".join(f"{k}={v:.3f}" for k, v in accs.items())
            log(f"  step {step + 1:>5d} nchain {n_chain:>3d} "
                f"loss {main_val:.3f} aux {aux_val:.3f} sym {sym_val:.3f} "
                f"λa {aux_lambda:.4f} T {T:.2f} lr {lr:.1e} {acc_str}")

    wallclock = time.time() - start
    final_acc = curve[-1]["acc"]
    payload = {
        "tag": tag, "rig": "rig016_lm_mix", "seed": seed,
        "params": n_params, "cfg": asdict(cfg),
        "chain_frac": chain_frac,
        "aux": {"style": "shared_bus_cleanup_plus_sym_aux_masked_to_chain",
                "layer": aux_layer, "lambda_init": aux_lambda_init,
                "sym_lambda_init": sym_lambda_init,
                "decay_tau": aux_decay_tau, "T_init": T_init, "T_final": T_final,
                "contrastive": contrastive_aux},
        "vocab_size": task_cfg.vocab_size,
        "train_tasks": [list(t) for t in train_tasks],
        "final_acc": final_acc, "wallclock_s": round(wallclock, 3), "curve": curve,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{tag}.json").write_text(json.dumps(payload, indent=2))
    log(f"[{tag}] done in {wallclock:.1f}s  final k{cfg.K1}={final_acc.get(f'k{cfg.K1}'):.3f}")
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--chain-frac", type=float, default=0.15)
    ap.add_argument("--K1", type=int, default=4)
    ap.add_argument("--K2", type=int, default=4)
    ap.add_argument("--K3", type=int, default=4)
    ap.add_argument("--K-chain", type=int, default=3)
    ap.add_argument("--hops", type=int, default=3)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--d-key", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--beta-bias-init", type=float, default=-0.5)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--aux-lambda-init", type=float, default=0.15)   # half per Fable
    ap.add_argument("--sym-lambda-init", type=float, default=0.15)
    ap.add_argument("--aux-decay-tau", type=float, default=4000.0)
    ap.add_argument("--aux-layer", type=int, default=0)
    ap.add_argument("--T-init", type=float, default=5.0)
    ap.add_argument("--T-final", type=float, default=1.0)
    ap.add_argument("--n-entities", type=int, default=512)
    ap.add_argument("--n-values", type=int, default=128)
    ap.add_argument("--contrastive-aux", action="store_true", default=True)
    ap.add_argument("--no-contrastive-aux", dest="contrastive_aux",
                    action="store_false")
    ap.add_argument("--lm-filler-lambda", type=float, default=0.0,
                    help=">0 enables per-position LM loss on filler rows.")
    ap.add_argument("--tag-prefix", default="rig016_lm_mix")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = RunCfg(
        K1=args.K1, K2=args.K2, steps=args.steps, batch=args.batch,
        mode="mixed", eval_every=args.eval_every, device=args.device,
        hops=args.hops, K3=args.K3,
    )
    hs_cfg = SharedBusConfig(
        d_model=args.d_model, d_key=args.d_key, n_layers=args.n_layers,
        beta_bias_init=args.beta_bias_init, K_chain=args.K_chain,
    )
    out_dir = HERE / "outputs"

    def make_builder(hs_cfg):
        return lambda vocab_size: build_model(vocab_size, hs_cfg)

    per_seed = []
    for seed in args.seeds:
        payload = train_one_lm_mix(
            build_model_fn=make_builder(hs_cfg), seed=seed,
            cfg=cfg, hs_cfg=hs_cfg, out_dir=out_dir,
            tag=f"{args.tag_prefix}_s{seed}",
            chain_frac=args.chain_frac,
            aux_lambda_init=args.aux_lambda_init,
            sym_lambda_init=args.sym_lambda_init,
            aux_decay_tau=args.aux_decay_tau,
            T_init=args.T_init, T_final=args.T_final,
            aux_layer=args.aux_layer,
            cfg_n_entities=args.n_entities, cfg_n_values=args.n_values,
            contrastive_aux=args.contrastive_aux,
            lm_filler_lambda=args.lm_filler_lambda,
        )
        per_seed.append(payload)

    summary = {
        "rig": "rig016_lm_mix", "seeds": args.seeds,
        "chain_frac": args.chain_frac,
        "hs_cfg": hs_cfg.__dict__, "run_cfg": cfg.__dict__,
        "n_entities": args.n_entities,
        "final_acc_per_seed": [p["final_acc"] for p in per_seed],
    }
    (out_dir / f"{args.tag_prefix}_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[summary] {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
