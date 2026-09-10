# Chapter 2: FREKI

*Stop hoping a chain forms during writing. Build the chaining into the read.*

FREKI stands for **Fixed Read via Explicit K-chain Iteration**, which is a fancy
way of saying "read K times on purpose."

This is the chapter write-up. For the big picture and how it connects to
chapter 1 (FENRIR), see the [top-level README](../README.md).

## Where chapter 1 left me

FENRIR could chain two hops inside one forward pass, but only about half the time
(a random-seed coin flip), and only for two hops. Pushing it to three hops did
not work at all. So chapter 2 changes tactics. Instead of nudging the *write* and
hoping a chain assembles itself, I make the *read* do the chaining explicitly:
after all the facts are written into the memory `M`, read from it several times in
a row within the same layer, once per hop.

## Being upfront about prior art

This is not a new mechanism, and I want to say so plainly before any results.
Reading a fast-weight memory several times inside a single layer, with a
normalization step between reads, to do transitive (multi-hop) lookups, is the
core idea of **Fast Weight Memory** (Schlag, Munkhdalai, and Schmidhuber, 2021).
Snapping each intermediate result onto the nearest clean token between hops is the
idea behind **Resonator Networks** (Frady et al., 2020). The subtractive write I
use is the **delta rule** (DeltaNet).

What this chapter actually contributes is narrower and more honest: an
engineering recipe that makes multi-hop chaining train *reliably across seeds* at
a realistic vocabulary size, a breakdown of which ingredients matter and why, and
a clean baseline that proves the chained read is doing the work.

## The recipe, in plain terms

Four ingredients, each fixing a different failure. Removing any one drops the
success rate:

1. **A delta-rule write.** When storing a new memory, first erase whatever was
   filed under that key, then write. This keeps different facts from smearing
   into each other inside the one shared matrix.
2. **A chained read with cleanup.** Read the query out of `M`, snap the result to
   the nearest real token, read *that* out of `M`, snap again, once per hop. The
   cleanup between hops stops noise from compounding.
3. **A small helper signal during training.** A side objective that checks the
   intermediate hop is landing on the right bridge token, so training gets a
   gradient before the full chain works end to end.
4. **An easy-to-hard curriculum.** Start with a trivial version of the task where
   the hops are identities, then ramp up to the real thing, so a randomly
   initialized model has a foothold.

## What I found

The task is a chain-composition lookup, like chapter 1 but pushed to three hops
and larger vocabularies. "Entities" is the size of the pool of possible tokens;
bigger is harder and more realistic. Every number below is the final accuracy on
the deepest link, and every one traces to a committed file under a rig's
`outputs/` folder. Chance (blind guessing) is about 25% here.

| Result | Rig | Seeds | Accuracy (deepest link) | Verdict |
|---|---|---|---|---|
| Two-hop, the original recipe | `rig_003_freki` | 10 | 10 / 10 pass, up to 1.000 | reliable |
| Three-hop, 128 entities, full recipe | `rig_010_arm2_plus_arm3` | 5 | 5 / 5 at 0.988 to 0.996 | reliable |
| Three-hop, **512 entities**, delta-rule write | `rig_013_delta_write` | 3 | 3 / 3 at 0.986, 0.998, 1.000 | reliable |
| **Plain baseline (the control)** | `rig_013_delta_write` | 3 | 3 / 3 at **0.242, 0.246, 0.279** | fails at chance |

**The control is the point of the chapter.** It is a standard delta-rule memory,
three layers stacked, with the chained read switched off (it reads once, not K
times) and the helper signal removed. On the exact same three-hop task at 512
entities, it never rises above the guessing floor: three runs, all at about 25%.
That is what isolates the contribution. The reliable results above are not coming
from "a bigger model" or "more training." They are coming specifically from the
chained-read-with-cleanup stack. Plain stacked memory layers cannot compose three
hops at this scale, full stop.

For the two-hop coin flip from chapter 1, the recipe turns 20% (roughly, the
FENRIR-rev baseline) into 10 out of 10. The three-hop recipe (`rig_010`) reaches
5 out of 5 at 128 entities, and the delta-rule write (`rig_013`) carries that all
the way to 512 entities at 3 out of 3.

## Where it breaks (the honest caveats)

- **At 2048 entities, the standard run fails.** The committed result at 2048
  entities sits at chance (0.248 on the deepest link at the normal training
  budget). A single longer run that I hand-nursed did better, but one run is not
  a result and I do not headline it. The reliable frontier here is 512 entities.
- **Deeper chains get shaky.** At four hops, the recipe passes 3 of 5 seeds at
  128 entities (`rig_011`), and a single-seed run reached 1.000 at 512 entities
  (`rig_013`, four-hop, longer budget). "Single-seed" means exactly that: treat
  it as "possible," not "reliable."
- **It did not transfer to language.** Mixing the chain task into ordinary
  next-token language-model training failed completely: `rig_016` scores 0 of 3
  above chance across every variant I tried (more data density, six times the
  training). The diagnosis is real, not a tuning miss: the memory needs to
  *choose* what is worth writing down based on content, and a plain delta-rule
  write does not have that selectivity. Uniform-random filler tokens actively
  collide with real facts and erase them under the delta rule.

That language-transfer failure is the honest ceiling on this whole line of work,
and it is why the search for a genuinely different memory operator continues
elsewhere.

## Reproduce

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r ../requirements.txt

# Two-hop headline: 10 seeds, ~7 min/seed on an RTX 5090.
python rig_003_freki/run.py --seeds 0 1 2 3 4 5 6 7 8 9 --steps 12000

# Three-hop, 128 entities, the full recipe: 5/5.
python rig_010_arm2_plus_arm3/run.py

# Three-hop, 512 entities, delta-rule write: 3/3 universal.
python rig_013_delta_write/run.py

# The negative control: same task, chained read OFF, helper OFF. Fails at chance.
python rig_013_delta_write/run.py \
  --K-chain 1 --aux-lambda-init 0 --sym-lambda-init 0 \
  --tag-prefix rig013_vanilla_delta_ent512_Kchain1
```

Each rig writes a per-seed JSON and a `*_summary.json` into its own `outputs/`
folder. The summaries already committed here hold the numbers in the table above,
so you can check every claim without a GPU.

## The rig map (including the dead ends)

I kept the failures on purpose. They are half the value of a research log.

| Rig | What it is | Outcome |
|---|---|---|
| `rig_000_baseline_fenrir` | Imports chapter 1's FENRIR mixer, runs it through this harness | Cross-chapter sanity floor |
| `rig_001_matprod` | Pure multiplicative memory | Dead end: higher hops get suppressed |
| `rig_002_neumann` | Learnable multi-hop read weights | Dead end: the model froze the higher-hop terms at zero |
| `rig_003_freki` | The winning two-hop recipe (fixed K-step read) | 10 / 10 |
| `rig_004_rotation` | Rotation-based memory state | Promising but slow; parked, needs a parallel kernel |
| `rig_005_hopstrat` | A separate memory per hop, gated | Dead end: the gates never specialized |
| `rig_006_residual_reads` | Sum every read depth into the output | Dead end: partial paths do not give partial credit |
| `rig_007_aux_only` | Just the helper-signal ingredient | 2 / 5 (one ingredient alone) |
| `rig_008_cleanup_aux` | Helper signal plus between-hop cleanup | 3 / 5 |
| `rig_009_id_homotopy` | Just the easy-to-hard curriculum | 2 / 5 |
| `rig_010_arm2_plus_arm3` | All ingredients stacked, three-hop | 5 / 5 at 128 entities |
| `rig_011_arm23_4hop` | The recipe pushed to four hops | 3 / 5 |
| `rig_012_shared_bus` | A shared-codebook cleanup variant | Partial at 512 entities |
| `rig_013_delta_write` | The delta-rule write, plus the negative control | 3 / 3 at 512 entities; control at chance |
| `rig_014_task_shape` | Does the result depend on how the task is laid out? | No: content-addressed, not position-based |
| `rig_016_lm_mix` | Transfer to language-model training | Failed, 0 / 3 |
| `rig_018_facts_in_noise` | Facts buried in random-token noise | Failed (delta write erases on collisions) |

## Files

```
common/harness.py      training loop, task sampler, evaluation
common/tasks.py        the two-hop and multi-hop chain benchmark
common/tasks_shape.py  task-layout variants (rig_014)
common/tasks_noise.py  facts-in-noise variants (rig_018)
common/tasks_lm_mix.py mixed language-model batches (rig_016)
rig_*/model.py         the memory variant for that experiment
rig_*/run.py           the reproduction entry point
rig_*/outputs/         committed per-seed and summary JSONs
```

The benchmark and harness in `common/` are backbone-agnostic. You can point them
at any recurrent layer, not just the ones here, which makes them a reusable
multi-hop composition test in their own right.
