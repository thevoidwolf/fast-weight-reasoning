"""Aggregate all outputs/*.json into a single report readable by humans
and by the paper's Table population step.

Groups by experiment family:
  - Table 1 (mixed): table1__*.json
  - Table 1b (joint): table1b__*.json
  - Ablation --no-eager: ablation_no_eager__*.json
  - Inference eager-KO: eval_ko__*.json
  - F3 probes: probe_f3__*.json AND scan_*__f3.json
  - F4 probes: probe_f4__*.json AND scan_*__f4*.json
  - F5 probes: probe_f5__*.json AND scan_*__f5*.json

Writes outputs/AGGREGATE.md (a human-readable summary) and prints the same
to stdout.
"""
from __future__ import annotations

import json
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def load_all() -> dict[str, list[dict]]:
    """Bucket every JSON in outputs/ by family (based on filename prefix)."""
    families = {
        "table1_mixed": [], "table1b_joint": [],
        "ablation_no_eager": [], "eval_ko": [],
        "f3_probe": [], "f4_probe": [], "f5_probe": [],
        "misc": [],
    }
    for j in sorted(OUTPUT_DIR.glob("*.json")):
        try:
            data = json.loads(j.read_text())
        except json.JSONDecodeError:
            continue
        name = j.stem
        if name.startswith("table1__"):
            families["table1_mixed"].append({"tag": name, **data})
        elif name.startswith("table1b__"):
            families["table1b_joint"].append({"tag": name, **data})
        elif name.startswith("ablation_no_eager__"):
            families["ablation_no_eager"].append({"tag": name, **data})
        elif name.startswith("eval_ko__"):
            families["eval_ko"].append({"tag": name, **data})
        elif "f3" in name and ("probe_f3__" in name or "__f3" in name):
            families["f3_probe"].append({"tag": name, **data})
        elif "f4" in name and ("probe_f4__" in name or "__f4" in name):
            families["f4_probe"].append({"tag": name, **data})
        elif "f5" in name and ("probe_f5__" in name or "__f5" in name):
            families["f5_probe"].append({"tag": name, **data})
        else:
            families["misc"].append({"tag": name, **data})
    return families


def _fmt_ladder(acc: dict) -> str:
    """{'k1':0.99,'k2':0.5,...} -> 'k1=0.99 k2=0.50 k3=... k4=...'"""
    return " ".join(f"{k}={v:.3f}" for k, v in sorted(acc.items()))


def render(families: dict[str, list[dict]]) -> str:
    lines = ["# FENRIR reproduction outputs -- aggregated summary\n"]

    def _table(header: str, key: str, extractor):
        lines.append(f"## {header} ({len(families[key])} runs)\n")
        if not families[key]:
            lines.append("(none yet)\n")
            return
        for row in families[key]:
            lines.append(extractor(row))
        lines.append("")

    def train_row(r):
        headline = r.get("final_acc_headline")
        pass_marker = "PASS" if (headline and headline >= 0.9) else "fail"
        wall = r.get("wallclock_s", 0.0)
        step95 = r.get("step_to_95_headline", "None")
        use_eager = r.get("use_eager", True)
        return (f"- **{r['tag']}** ({pass_marker})  "
                f"final: {_fmt_ladder(r.get('final_acc', {}))}  "
                f"| step_to_95={step95}  | wall={wall:.0f}s  | use_eager={use_eager}")

    _table("Table 1 -- mixed curriculum", "table1_mixed", train_row)
    _table("Table 1b -- joint curriculum", "table1b_joint", train_row)
    _table("Ablation --no-eager (training-time causal control)", "ablation_no_eager", train_row)

    def eval_ko_row(r):
        on_row = _fmt_ladder(r["acc_eager_on"])
        off_row = _fmt_ladder(r["acc_eager_off"])
        return (f"- **{r['tag']}**  eager ON: {on_row}  |  "
                f"eager OFF: {off_row}")

    _table("Inference eager-KO (on trained ckpts)", "eval_ko", eval_ko_row)

    def f3_row(r):
        acc_by_layer = r.get("acc_by_layer", {})
        acc_str = " ".join(f"L{k}={float(v):.3f}" for k, v in sorted(acc_by_layer.items()))
        return f"- **{r['tag']}**  {acc_str}"

    _table("F3 logit lens probes", "f3_probe", f3_row)

    def f4_row(r):
        per_t = r.get("per_timestep", [])
        if not per_t:
            return f"- **{r['tag']}**  (empty)"
        atok = per_t[-1]
        early = per_t[len(per_t) // 4] if len(per_t) > 4 else per_t[0]
        return (f"- **{r['tag']}**  KO={r.get('eager_knockout')}  "
                f"ATOK answer={atok['answer_acc']:.3f}  bridge={atok['bridge_acc']:.3f}  "
                f"| early t={early['t']} ans={early['answer_acc']:.3f}")

    _table("F4 mid-scan probes", "f4_probe", f4_row)

    def f5_row(r):
        return f"- **{r['tag']}**  (see JSON for detail)"

    _table("F5 Q/K alignment probes", "f5_probe", f5_row)

    if families["misc"]:
        lines.append(f"## Misc ({len(families['misc'])})\n")
        for r in families["misc"]:
            lines.append(f"- {r['tag']}")

    return "\n".join(lines)


def main():
    families = load_all()
    report = render(families)
    print(report)
    (OUTPUT_DIR / "AGGREGATE.md").write_text(report)


if __name__ == "__main__":
    main()
