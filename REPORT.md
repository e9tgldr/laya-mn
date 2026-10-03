# Laya Mongolian run

## Timeline

| Phase | Started | Finished | Result |
|---|---|---|---|
| Overnight run (4 epochs) | 2026-09-20 23:49 | 2026-09-21 02:17 | exit 0 |
| Continuation chunk 1 (epochs 5-8) | 2026-09-21 19:42 | 2026-09-21 23:14 | exit 0, new best |
| Continuation chunk 2 (epoch 9) | 2026-09-21 23:14 | 2026-09-22 06:28 | stopped on request |

Final artifact: `out_mn/` = **epoch 8**, chosen by best validation accuracy (0.8981).
`best.pt`/`last.pt` are 1.2 GB each and are training checkpoints only - `model.safetensors` is the deliverable.

## Test-split results (MASSIVE mn-MN, 5668 decisions)

| Metric | Shipped multilingual | 4-epoch | **8-epoch (final)** |
|---|---|---|---|
| Overall accuracy | 0.3035 | 0.8918 | **0.8984** |
| Scenario accuracy (18-way) | 0.1513 | 0.8658 | **0.8753** |
| Intent accuracy (conditional) | 0.4714 | 0.9206 | **0.9239** |
| ECE (lower better) | 0.4277 | **0.0478** | 0.0587 |
| Mean NLL (lower better) | 3.6740 | 0.4902 | **0.4836** |

Random baselines: scenario 1/18 = 0.0556, intent 1/4 = 0.25.
Intent accuracy is **conditional on the gold scenario being supplied** and is not end-to-end performance.

## Validation curve

| Epoch | Val acc | Scenario | Intent | Note |
|---|---|---|---|---|
| 1 | 0.8561 | 0.814 | 0.903 | |
| 2 | 0.8639 | 0.830 | 0.902 | |
| 3 | 0.8854 | 0.851 | 0.924 | |
| 4 | 0.8937 | 0.865 | 0.926 | original best |
| 5 | 0.8774 | 0.846 | 0.912 | dip - LR warm restart |
| 6 | 0.8849 | 0.855 | 0.918 | recovering |
| 7 | 0.8917 | 0.862 | 0.924 | |
| **8** | **0.8981** | **0.870** | **0.929** | **exported** |
| 9 | 0.8769 | 0.845 | 0.912 | did not beat best |

The four continuation epochs bought **+0.66 points** overall accuracy. Calibration got slightly
worse (ECE 0.0478 -> 0.0587), so the gain is real but modest and not free.
Fitted choice temperatures: 3.662 (4-epoch) -> 3.433 (8-epoch).

## Verified working

`laya.load(r'out_mn').predict()` on the real test utterance
`энэ долоо хоног намайг таван цагт сэрээгээрэй` (gold scenario `alarm`)
returns `alarm` at **p=0.9856**, confidence 0.9514.

## Method

- Base: `convaiinnovations/laya` multilingual checkpoint (mmBERT-base, 321.9M params).
- Data: MASSIVE mn-MN (CC-BY-4.0), 11514 train rows, 18 scenarios, 60 intents.
- Two questions per utterance: `scenario` (18-way) and `intent` within the gold scenario;
  single-option questions dropped as degenerate (news and weather have one intent each).
- Method: upstream RLCD recipe - GRPO-style REINFORCE on a strictly proper scoring reward
  (log + spherical, + RPS for ordinal) with soft cross-entropy guidance, then LBFGS
  temperature calibration.
- In RLCD the training loss is **not** a progress signal; judge on val accuracy / ECE.
- Chunked continuation gives each chunk a fresh cosine cycle, so epochs 5 and 9 start
  with LR and sigma reset (warm restart) and typically dip before recovering.

## Caveats

- **MASSIVE mn-MN is professionally translated from English.** These numbers are an upper
  bound and do not transfer directly to real Mongolian email, chat or student messages.
- 0.8753 scenario accuracy is on an 18-way forced choice; open-ended real-world inputs will
  be harder and the model never generates text - it only scores options you supply.

## Operational notes

- **CUDA dies across a Windows sleep/resume here.** On 2026-09-22 00:07 the laptop slept for
  5h41m; training froze and the CUDA context became a zombie (GPU pinned at 100%, 5.8 GB
  orphaned, no progress, process missing from `nvidia-smi` compute-apps). The trainer now runs
  a `SetThreadExecutionState` keep-awake thread to block idle sleep. Keep the lid open.
- `logs/continue.log` = current run; `logs/continue_run1.log` = the stalled run's history.
- To stop a run: create a file named `STOP` next to `continue.ps1`. Checked every ~50
  micro-batches (~40 s); it finishes the epoch, calibrates and exports cleanly.
