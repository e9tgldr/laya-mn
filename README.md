# laya-mn

A Mongolian intent and scenario classifier, fine-tuned from the multilingual **laya** checkpoint (mmBERT-base, 321.9M params).

**Test accuracy 0.8984, against 0.3035 for the shipped multilingual checkpoint on the same split.**

## What this model actually does

It is a **scoring model, not a generator**. You give it the utterance and a set of candidate options; it returns a calibrated probability for each. It never produces text.

That constraint is deliberate. "Чатбот бичдэг үү?" — is it a chatbot — is a judgement with a known answer, so the right architecture is a ranking model over supplied options, not free generation. It makes the output auditable: you can see exactly which option won and with what confidence.

## Results

Test split: MASSIVE mn-MN, 5,668 decisions.

| Metric | Shipped multilingual | **This model** |
|---|---|---|
| Overall accuracy | 0.3035 | **0.8984** |
| Scenario accuracy (18-way) | 0.1513 | **0.8753** |
| Intent accuracy (conditional) | 0.4714 | **0.9239** |
| Calibration error (ECE, lower better) | 0.4277 | **0.0587** |
| Mean NLL (lower better) | 3.6740 | **0.4836** |

Random baselines: scenario 1/18 = 0.0556, intent 1/4 = 0.25.

The baseline's ECE of 0.4277 is the important number. The off-the-shelf checkpoint is not just wrong, it is **confidently wrong** — it assigns high probability to incorrect answers. After calibration the model is honest about what it does not know.

**Intent accuracy is conditional on the gold scenario being supplied.** It is not end-to-end performance.

## Method

Two questions per utterance:

- **scenario** — 18-way, which domain the message belongs to
- **intent** — the specific intent *within* the gold scenario

Single-option scenario questions are dropped as degenerate (news and weather have one intent each, so there is nothing to discriminate).

Training follows the upstream RLCD recipe — GRPO-style REINFORCE against a strictly proper scoring reward (log + spherical, plus RPS for ordinal targets), with soft cross-entropy guidance, followed by post-hoc LBFGS temperature calibration.

I ported this from the official 2×T4 recipe down to a **single 6 GB consumer GPU**, which required chunked continuation so each chunk gets a fresh cosine cycle.

One consequence of RLCD worth knowing: **the training loss is not a progress signal.** Judging convergence from loss will mislead you. Validation accuracy and ECE are the only numbers that mean anything.

## The email specialist, and what it taught me

`mn_email.py` builds an email-focused model. It does *not* train on email only — if every utterance were an email, the scenario head would learn to answer "email" for everything and become useless as a filter. So it keeps all email items and adds an equal number of non-email negatives, which keeps the scenario head discriminating.

| Model | On email set | On general set |
|---|---|---|
| Email specialist | 0.8745 | **0.6640** |
| General model | — | 0.8984 |

The specialist is genuinely better at email and much worse everywhere else. More importantly, **the drop-off is the real finding**: MASSIVE mn-MN is professionally translated from English, so scores on real Mongolian email, chat or student messages will be materially lower. **These numbers are an upper bound.**

## Honest caveats

- MASSIVE mn-MN is professionally translated from English. Real Mongolian input will be harder.
- 0.8753 scenario accuracy is on an 18-way forced choice. Open-ended input is harder still.
- The model scores options you supply. It does not classify arbitrary text without a candidate set.

## Operational note

**CUDA dies across a Windows sleep/resume on this machine.** During one run the laptop slept for 5h41m and the CUDA context became a zombie — GPU pinned at 100%, 5.8 GB orphaned, process gone from `nvidia-smi`. The trainer now runs a `SetThreadExecutionState` keep-awake thread. Keep the lid open for any long run.

To stop a run cleanly, create a file named `STOP` next to `continue.ps1`. It is polled about every 40 seconds and finishes the current epoch, calibrates, and exports properly.

## What is and is not in this repo

**Included:** all training, evaluation, calibration and probing scripts, plus the full `reports/` directory with every evaluation JSON.

**Not included:** model weights. `model.safetensors` is the deliverable artifact and is too large for Git. Training checkpoints (`best.pt`, `last.pt`) are 1.2 GB each and are resumable state, not output.

## Scripts

| File | Purpose |
|---|---|
| `mn_data.py` | Build the MASSIVE mn-MN item set |
| `mn_train.py` | RLCD fine-tune on one GPU |
| `mn_eval.py` | Test-split evaluation |
| `mn_calib.py` | Temperature calibration |
| `mn_email.py` | Email-focused item set with negatives |
| `_probe_mn.py` | Probe base-model Mongolian comprehension |
| `_preflight.py` | CUDA / environment check before training |
| `overnight.ps1`, `continue.ps1` | Long-run launchers |

Full run history, the validation curve across all nine epochs, and the export provenance are in [REPORT.md](REPORT.md).