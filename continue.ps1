# ASCII only. Continued Laya-MN training.
# Runs forever in chunks of 4 epochs, resuming from out_mn\last.pt each time.
# To STOP: create a file named STOP in this folder (or just tell the agent).
$ErrorActionPreference = "Continue"
$root = "C:\Users\hi\Documents\laya-mn"
$py   = "$root\venv\Scripts\python.exe"
$log  = "$root\logs\continue.log"
$stop = "$root\STOP"
$out  = "$root\out_mn"

Set-Location $root
New-Item -ItemType Directory -Force -Path "$root\logs" | Out-Null
$env:PYTHONIOENCODING = "utf-8"

function Log($m) {
  $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $m
  Add-Content -Path $log -Value $line
}

Log "=== CONTINUATION START (no deadline; create STOP file to finish) ==="

# CUDA precheck - refuse to burn hours on a CPU fallback
& $py -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 9)" 2>$null
if ($LASTEXITCODE -ne 0) { Log "ABORT: no CUDA GPU visible"; exit 9 }
Log "CUDA OK - starting"

$chunk = 4
$round = 0
while (-not (Test-Path $stop)) {
  $round++
  Log "--- chunk $round : $chunk more epochs ---"
  & $py "$root\mn_train.py" --resume --epochs $chunk --max-hours 24 --out $out --stop-file $stop 2>&1 |
    Out-File -FilePath $log -Append -Encoding utf8
  $code = $LASTEXITCODE
  Log "chunk $round exit=$code"
  if (Test-Path $stop) { Log "STOP present -> ending cleanly"; break }
  if ($code -ne 0) { Log "non-zero exit; retrying in 120s"; Start-Sleep -Seconds 120 }
}
Log "=== CONTINUATION END ==="
