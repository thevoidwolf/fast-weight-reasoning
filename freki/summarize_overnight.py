#!/usr/bin/env python3
"""Summarize overnight results across rig_013/014/016 outputs.

Run this in the morning to get a compact table of every completed run
without opening files individually. Prints markdown-flavoured table
grouped by experiment.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RIGS = ["rig_013_delta_write", "rig_014_task_shape", "rig_016_lm_mix"]


def _fmt_acc(acc: dict) -> str:
    return " ".join(f"{k}={v:.3f}" for k, v in sorted(acc.items()))


def _load_runs(rig_dir: Path) -> list[dict]:
    out = []
    for f in sorted((rig_dir / "outputs").glob("*.json")):
        if "summary" in f.name:
            continue
        try:
            payload = json.loads(f.read_text())
        except Exception as e:
            out.append({"file": f.name, "error": str(e)})
            continue
        row = {
            "file": f.name,
            "tag": payload.get("tag", "?"),
            "seed": payload.get("seed", "?"),
            "final_acc": payload.get("final_acc", {}),
            "steps": payload.get("cfg", {}).get("steps", "?"),
            "n_entities": payload.get("cfg", {}).get("n_values", "?"),  # placeholder
        }
        cfg = payload.get("cfg", {})
        cur = payload.get("curve") or []
        if cur:
            row["last_step"] = cur[-1].get("step", "?")
            row["last_loss"] = cur[-1].get("loss", float("nan"))
        row["hops"] = cfg.get("hops", "?")
        row["K_chain"] = payload.get("cfg", {}).get("K_chain", "?")
        row["wallclock_s"] = payload.get("wallclock_s", "?")
        if "arm" in payload:
            row["arm"] = payload["arm"]
        if "chain_frac" in payload:
            row["chain_frac"] = payload["chain_frac"]
        out.append(row)
    return out


def main():
    print("# Overnight results — 2026-08-14 → 2026-08-15\n")
    for rig in RIGS:
        rig_dir = HERE / rig
        if not rig_dir.exists():
            continue
        runs = _load_runs(rig_dir)
        if not runs:
            continue
        print(f"## {rig}\n")
        print(f"| tag | seed | steps | final_acc | wallclock |")
        print(f"|---|---|---|---|---|")
        for r in runs:
            acc = _fmt_acc(r.get("final_acc", {}))
            print(f"| {r['tag']} | {r['seed']} | {r.get('steps', '?')} | "
                  f"{acc} | {r.get('wallclock_s', '?')}s |")
        print()

    # Also list any log files that haven't completed (no JSON payload)
    print("## In-progress or failed runs (log-only, no JSON)\n")
    for rig in RIGS:
        rig_dir = HERE / rig / "outputs"
        if not rig_dir.exists():
            continue
        for log in sorted(rig_dir.glob("*.log")):
            stem = log.stem
            has_json = any((rig_dir / f"{stem}*.json").glob("*")) or \
                       any(rig_dir.glob(f"{stem}*.json"))
            if not has_json:
                sz = log.stat().st_size
                tail = ""
                try:
                    with open(log, "rb") as f:
                        f.seek(max(0, sz - 400))
                        tail = f.read().decode(errors="replace").strip().splitlines()[-1:]
                    tail = tail[0] if tail else ""
                except Exception:
                    pass
                print(f"- `{log.name}` ({sz:,} bytes) — last: {tail[:120]}")


if __name__ == "__main__":
    main()
