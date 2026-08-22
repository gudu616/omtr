# RUNS.md — literal run commands

The `--levels` and `--eligible-as` values used for each of the five models are not
recorded in any other artifact (each raw JSON's `meta` block records the model,
dtype, and library versions, but not the CLI invocation that produced it). This
file is the authoritative source for them.

## Hardware

Single consumer GPU: **NVIDIA RTX 4060, 8GB VRAM**. All five models load in
**fp16** (`torch.float16`) via `from_pretrained_no_processing`; the harness falls
back to fp32 on CPU automatically when CUDA is unavailable (`harness/run_pilot.py`).
No multi-GPU or distributed setup was used — 8GB was sufficient for every model
size run here, up to 2.8B parameters. Software versions: see `requirements.txt`.

## Main pilot runs (five models)

Two "flagship" models — **Pythia-1.4B** and **OLMo-2-0425-1B** — ran the full
six-level battery (L0, L0N, L1, L2, L3, L4, L5), because their training corpora
are the two publicly indexed in infini-gram. The three remaining Pythia sizes ran
only the bare-continuation family (L0, L0N, L1) and are pointed at the 1.4B
model's corpus-family tag via `--eligible-as`, since all four Pythia sizes train
on the same corpus (the Pile) and `battery/battery.json` records L0/L0N
eligibility under the name `EleutherAI/pythia-1.4b` for the whole family.

```
python harness/run_pilot.py --model EleutherAI/pythia-1.4b --levels L0,L0N,L1,L2,L3,L4,L5

python harness/run_pilot.py --model allenai/OLMo-2-0425-1B --levels L0,L0N,L1,L2,L3,L4,L5

python harness/run_pilot.py --model EleutherAI/pythia-410m --levels L0,L0N,L1 --eligible-as EleutherAI/pythia-1.4b

python harness/run_pilot.py --model EleutherAI/pythia-1b --levels L0,L0N,L1 --eligible-as EleutherAI/pythia-1.4b

python harness/run_pilot.py --model EleutherAI/pythia-2.8b --levels L0,L0N,L1 --eligible-as EleutherAI/pythia-1.4b
```

Each command writes `results/raw/pilot_<model>.json` plus the
`resid_first_<model>.npz` / `resid_mean_<model>.npz` residual-stream dumps
(`--out` defaults to `results/raw`). `--eligible-as` only changes which battery
items are *selected* (via the `eligible_models` field on each item); it never
changes which model weights load.

## L0N v2.1 rerun + merge (control rebuild)

After the L0N (matched non-memorized control) construction bug was found and
fixed — the content-word boundary rule was not applied to the control the same
way it was applied to L0 (see `docs/PROCESS_LOG.md`, 第十四步) — all five
models' L0N items were rerun into a separate output directory, then merged back
into the main results, replacing the superseded L0N records:

```
python harness/run_pilot.py --model EleutherAI/pythia-1.4b --levels L0N --out results/raw_l0n_v21

python harness/run_pilot.py --model allenai/OLMo-2-0425-1B --levels L0N --out results/raw_l0n_v21

python harness/run_pilot.py --model EleutherAI/pythia-410m --levels L0N --eligible-as EleutherAI/pythia-1.4b --out results/raw_l0n_v21

python harness/run_pilot.py --model EleutherAI/pythia-1b --levels L0N --eligible-as EleutherAI/pythia-1.4b --out results/raw_l0n_v21

python harness/run_pilot.py --model EleutherAI/pythia-2.8b --levels L0N --eligible-as EleutherAI/pythia-1.4b --out results/raw_l0n_v21

python harness/merge_l0n.py
```

`harness/merge_l0n.py` backs up each pre-merge file to
`results/raw/pilot_<model>.json.bak_pre_v21`, drops the superseded L0N records
from the main file, and appends the v2.1 ones (with matching merges of the two
`.npz` residual-stream files). The `.bak_pre_v21` files are physical evidence
that the control rebuild predates the adversarial review of the results — keep
them; do not delete or silently overwrite them. (In the shipped package these
already-generated backup files ship at `archive/raw/`, not `results/raw/` —
they are historical evidence, not a current canonical input; a fresh run of
the command above would still write them to `results/raw/` as described.)

## Reproducibility caveats

- The §5 greedy-vs-sampled side analysis (`harness/check_l5_novelty.py`) draws
  one temperature-0.8 sample per item and is not seeded, so it is not exactly
  reproducible from a rerun; this is a known limitation, not an omission from
  this file.
- `battery/battery.json` (`version: "v2.1"`) is the authoritative frozen battery
  for every command above; the `.bak_pre_v21` files reflect an earlier version
  of the L0N items only, not a different battery version overall.
