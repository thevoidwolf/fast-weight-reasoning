"""Score the H4 curriculum-dispatch rule on fresh seeds.

Rule under test:
  mixed viable iff k1 >= 0.40 at step 1600
  joint viable iff k2 >= 0.28 at step 1200

Ground truth: full-budget k4 (>= 0.85 = PASS) on each curriculum.

Reports:
  - per-seed prediction vs truth
  - overall accuracy
  - per-flip-class accuracy (both-PASS, both-fail, mixed-only, joint-only)
  - combined with training-sample stats (seeds 0-9 from the original run)
"""
from __future__ import annotations
import json, glob, argparse, os
from pathlib import Path

OUTPUTS = Path(__file__).resolve().parent / "outputs"

PASS_K4 = 0.85
MIXED_K1_STEP = 1600
MIXED_K1_THR = 0.40
JOINT_K2_STEP = 1200
JOINT_K2_THR = 0.28


def load_pair(fresh_only: bool) -> dict[int, dict]:
    """seed -> {'mixed': run, 'joint': run}. Missing sides left absent."""
    pairs: dict[int, dict] = {}
    mixed_glob = 'h4_fresh__mixed_s*.json' if fresh_only else 'table1__rev_K1max4_K2_4_s*.json'
    joint_glob = 'h4_fresh__joint_s*.json' if fresh_only else 'table1b__rev_joint_K1max4_K2_4_s*.json'
    for f in sorted(OUTPUTS.glob(mixed_glob)):
        d = json.load(open(f))
        pairs.setdefault(d['seed'], {})['mixed'] = d
    for f in sorted(OUTPUTS.glob(joint_glob)):
        d = json.load(open(f))
        pairs.setdefault(d['seed'], {})['joint'] = d
    return pairs


def sig_at(run, step, key):
    if run is None: return None
    pts = {c['step']: c for c in run['curve']}
    if step not in pts: return None
    if key == 'loss': return pts[step]['loss']
    return pts[step]['acc'].get(key)


def score(pairs: dict) -> tuple[list, dict]:
    """Return per-seed rows + summary stats."""
    rows = []
    for s in sorted(pairs):
        m, j = pairs[s].get('mixed'), pairs[s].get('joint')
        mk4 = m['final_acc']['k4'] if m else None
        jk4 = j['final_acc']['k4'] if j else None
        m_p = mk4 is not None and mk4 >= PASS_K4
        j_p = jk4 is not None and jk4 >= PASS_K4

        m_k1 = sig_at(m, MIXED_K1_STEP, 'k1')
        j_k2 = sig_at(j, JOINT_K2_STEP, 'k2')

        m_viable = m_k1 is not None and m_k1 >= MIXED_K1_THR
        j_viable = j_k2 is not None and j_k2 >= JOINT_K2_THR

        if mk4 is None or jk4 is None:
            truth = 'partial'
        elif m_p and j_p: truth = 'both PASS'
        elif not m_p and not j_p: truth = 'both fail'
        elif m_p: truth = 'MIXED wins'
        else: truth = 'JOINT wins'

        if m_viable and j_viable: pred = 'both PASS'
        elif not m_viable and not j_viable: pred = 'both fail'
        elif m_viable: pred = 'MIXED wins'
        else: pred = 'JOINT wins'

        rows.append({'seed': s, 'mk4': mk4, 'jk4': jk4, 'm_k1@1600': m_k1,
                     'j_k2@1200': j_k2, 'truth': truth, 'pred': pred,
                     'correct': truth == pred})

    complete = [r for r in rows if r['truth'] != 'partial']
    correct = sum(1 for r in complete if r['correct'])
    flip_rows = [r for r in complete if r['truth'] in ('MIXED wins', 'JOINT wins')]
    flip_correct = sum(1 for r in flip_rows if r['correct'])

    summary = {
        'n_seeds': len(complete), 'n_correct': correct,
        'n_flip_seeds': len(flip_rows), 'n_flip_correct': flip_correct,
        'accuracy': correct / len(complete) if complete else None,
    }
    return rows, summary


def print_rows(rows, header):
    print(f'\n=== {header} ===')
    print(f'{"seed":>4}  {"mk4":>6}  {"jk4":>6}   {"k1@1600":>8} {"k2@1200":>8}   '
          f'{"truth":>12}  {"pred":>12}  ok?')
    for r in rows:
        mk = f'{r["mk4"]:.3f}' if r['mk4'] is not None else '  --  '
        jk = f'{r["jk4"]:.3f}' if r['jk4'] is not None else '  --  '
        mk1 = f'{r["m_k1@1600"]:.3f}' if r['m_k1@1600'] is not None else '  --  '
        jk2 = f'{r["j_k2@1200"]:.3f}' if r['j_k2@1200'] is not None else '  --  '
        ok = 'OK' if r['correct'] else 'MISS'
        print(f'{r["seed"]:>4}  {mk:>6}  {jk:>6}   {mk1:>8} {jk2:>8}   '
              f'{r["truth"]:>12}  {r["pred"]:>12}  {ok}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fresh-only', action='store_true',
                    help='Score only the h4_fresh__ runs (seeds 10-19).')
    ap.add_argument('--both', action='store_true',
                    help='Score both the original 0-9 training and the fresh 10-19 sample.')
    args = ap.parse_args()

    if args.both:
        orig, orig_summ = score(load_pair(fresh_only=False))
        fresh, fresh_summ = score(load_pair(fresh_only=True))
        print_rows(orig, 'ORIGINAL training sample (seeds 0-9) — used to derive rule')
        print(f'  training-sample: {orig_summ["n_correct"]}/{orig_summ["n_seeds"]} correct  '
              f'(flip subset: {orig_summ["n_flip_correct"]}/{orig_summ["n_flip_seeds"]})')
        print_rows(fresh, 'FRESH independent sample (seeds 10-19) — VALIDATES the rule')
        print(f'  fresh-sample:    {fresh_summ["n_correct"]}/{fresh_summ["n_seeds"]} correct  '
              f'(flip subset: {fresh_summ["n_flip_correct"]}/{fresh_summ["n_flip_seeds"]})')
    else:
        pairs = load_pair(fresh_only=args.fresh_only)
        rows, summ = score(pairs)
        label = 'fresh seeds 10-19' if args.fresh_only else 'all available'
        print_rows(rows, label)
        print(f'\naccuracy: {summ["n_correct"]}/{summ["n_seeds"]} '
              f'(flip only: {summ["n_flip_correct"]}/{summ["n_flip_seeds"]})')


if __name__ == '__main__':
    main()
