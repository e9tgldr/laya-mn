# Laya -> Mongolian overnight run, deadline-aware (stops in time for 05:30).
# ASCII ONLY: PowerShell 5.1 parses BOM-less UTF-8 scripts as ANSI.
# Stages: preflight -> data prep -> baseline eval -> RLCD train -> final eval -> REPORT.md
# Fully autonomous: no agent supervision needed, everything lands in logs\ and reports\.
$ErrorActionPreference = 'Continue'
$root = "C:\Users\hi\Documents\laya-mn"
Set-Location $root
$py = Join-Path $root "venv\Scripts\python.exe"
$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_DISABLE_TELEMETRY = '1'

$deadline = Get-Date -Hour 5 -Minute 30 -Second 0
if ($deadline -lt (Get-Date)) { $deadline = $deadline.AddDays(1) }
# reserve for calibration + export + final eval + report
$reserveMin = 45
$trainBudgetH = [Math]::Max(0.5, [Math]::Round((($deadline - (Get-Date)).TotalMinutes - $reserveMin) / 60.0, 2))

$log = Join-Path $root "logs\overnight.log"
$status = Join-Path $root "logs\STATUS.txt"
New-Item -ItemType Directory -Force -Path (Join-Path $root "logs") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $root "reports") | Out-Null

function Log($m) {
    $line = "[{0}] {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $m
    Write-Host $line
    Add-Content -Path $log -Value $line -Encoding UTF8
}
function Set-Status($m) { Set-Content -Path $status -Value ("{0}`n{1}" -f (Get-Date -Format 'HH:mm:ss'), $m) -Encoding UTF8 }
function LogTail($f, $n) {
    if (Test-Path $f) {
        Log ("--- tail {0} ---" -f (Split-Path $f -Leaf))
        Get-Content $f -Tail $n | ForEach-Object { Log ("    " + $_) }
    }
}

$items = Join-Path $root "items_192_128.pt"
$modelDir = Join-Path $root "model"
$outDir = Join-Path $root "out_mn"
$t0 = Get-Date
Log "=== overnight run start | deadline $(($deadline).ToString('HH:mm')) | train budget ${trainBudgetH}h ==="
Set-Status "PREFLIGHT"

# ---------------------------------------------------------------- 0. preflight
Log "0/5 preflight: verify shipped weights load and build_sequence works"
$pre = @"
import os, torch, mn_data
from laya.common import build_model
from safetensors.torch import load_file
ml = os.path.join(r'$modelDir', 'multilingual')
cfg = mn_data.load_cfg(ml)
m = build_model(cfg, encoder_dir=os.path.join(ml, 'encoder'))
sd = load_file(os.path.join(ml, 'model.safetensors'))
missing, unexpected = m.load_state_dict(sd, strict=False)
print('PREFLIGHT keys=%d missing=%d unexpected=%d' % (len(sd), len(missing), len(unexpected)))
print('PREFLIGHT missing_sample=%s' % (missing[:5],))
print('PREFLIGHT OK' if len(missing) == 0 else 'PREFLIGHT WARN: untrained params present')
"@
Set-Content -Path (Join-Path $root "_preflight.py") -Value $pre -Encoding UTF8
& $py (Join-Path $root "_preflight.py") 2>&1 | Tee-Object -FilePath (Join-Path $root "logs\preflight.log") | Out-Null
$preExit = $LASTEXITCODE
Log "preflight exit=$preExit"
LogTail (Join-Path $root "logs\preflight.log") 3
if ($preExit -ne 0) { Log "FATAL: preflight failed"; Set-Status "FAILED: preflight"; exit 1 }

# ---------------------------------------------------------------- 1. data prep
if (Test-Path $items) {
    Log "1/5 items already present"
} else {
    Set-Status "DATA PREP"
    Log "1/5 building items from MASSIVE mn-MN ..."
    & $py mn_data.py 2>&1 | Tee-Object -FilePath (Join-Path $root "logs\prep.log") | Out-Null
    Log "prep exit=$LASTEXITCODE"
    LogTail (Join-Path $root "logs\prep.log") 6
}
if (-not (Test-Path $items)) { Log "FATAL: items missing"; Set-Status "FAILED: data prep"; exit 1 }

# ---------------------------------------------------------------- 2. baseline eval
Set-Status "BASELINE EVAL"
Log "2/5 baseline eval (shipped laya-multilingual, Mongolian test split) ..."
& $py mn_eval.py --model (Join-Path $modelDir "multilingual") --tag baseline_mn --items $items 2>&1 |
    Tee-Object -FilePath (Join-Path $root "logs\eval_baseline.log") | Out-Null
Log "baseline eval exit=$LASTEXITCODE"

# ---------------------------------------------------------------- 3. train
Set-Status "TRAINING"
Log "3/5 RLCD training (budget ${trainBudgetH}h, micro-batch 8, 4 epochs, stops early on deadline) ..."
& $py mn_train.py --items $items --epochs 4 --micro-batch 8 --grad-accum 4 --group-size 4 `
    --max-hours $trainBudgetH --out $outDir 2>&1 |
    Tee-Object -FilePath (Join-Path $root "logs\train.log") | Out-Null
$trainExit = $LASTEXITCODE
Log "train exit=$trainExit"
LogTail (Join-Path $root "logs\train.log") 8

# ---------------------------------------------------------------- 4. final eval
if (Test-Path (Join-Path $outDir "model.safetensors")) {
    Set-Status "FINAL EVAL"
    Log "4/5 final eval (fine-tuned mn checkpoint) ..."
    & $py mn_eval.py --model $outDir --tag finetuned_mn --items $items 2>&1 |
        Tee-Object -FilePath (Join-Path $root "logs\eval_finetuned.log") | Out-Null
    Log "final eval exit=$LASTEXITCODE"
} else {
    Log "4/5 SKIPPED: no model.safetensors (training did not reach export)"
}

# ---------------------------------------------------------------- 5. report
Log "5/5 writing REPORT.md"
$rep = Join-Path $root "REPORT.md"
$lines = @()
$lines += "# Laya Mongolian overnight run"
$lines += ""
$lines += ("Started: {0}" -f $t0.ToString('yyyy-MM-dd HH:mm'))
$lines += ("Finished: {0}" -f (Get-Date).ToString('yyyy-MM-dd HH:mm'))
$lines += ("Train exit code: {0}" -f $trainExit)
$lines += ("Train budget: {0}h (deadline {1})" -f $trainBudgetH, $deadline.ToString('HH:mm'))
$lines += ""
foreach ($tag in @('baseline_mn', 'finetuned_mn')) {
    $f = Join-Path $root ("reports\eval_{0}.json" -f $tag)
    if (Test-Path $f) {
        try {
            $j = Get-Content $f -Raw | ConvertFrom-Json
            $lines += ("## {0}" -f $tag)
            $lines += ""
            $lines += ("- overall accuracy: **{0:N4}**" -f $j.accuracy_overall)
            $lines += ("- ECE: {0}" -f $j.ece)
            $lines += ("- mean NLL: {0}" -f $j.mean_nll)
            $lines += ("- decisions: {0}" -f $j.n_decisions)
            foreach ($q in $j.per_question.PSObject.Properties) {
                $lines += ("  - {0}: accuracy {1:N4} (n={2})" -f $q.Name, $q.Value.accuracy, $q.Value.n)
            }
            $lines += ("- random baselines: scenario 1/18 = 0.0556, intent 1/4 = 0.25")
            $lines += ""
        } catch { $lines += ("- failed to parse {0}: {1}" -f $f, $_.Exception.Message) }
    } else {
        $lines += ("## {0} - MISSING" -f $tag)
        $lines += ""
    }
}
$lines += "## Notes"
$lines += ""
$lines += "- Base: convaiinnovations/laya multilingual checkpoint (mmBERT-base, 321.9M)."
$lines += "- Data: MASSIVE mn-MN (CC-BY-4.0), 11514 train rows, 18 scenarios, 60 intents."
$lines += "- Two questions per utterance: scenario (18-way) and intent within the gold scenario."
$lines += "- Intent accuracy is CONDITIONAL on the gold scenario being supplied."
$lines += "- Method: upstream RLCD recipe (GRPO REINFORCE on a strictly proper scoring reward + soft CE)."
Set-Content -Path $rep -Value ($lines -join "`r`n") -Encoding UTF8
Log "report -> $rep"

$mins = [Math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
Log "=== finished in $mins min ==="
Set-Status ("DONE: {0} min, train exit={1}" -f $mins, $trainExit)
