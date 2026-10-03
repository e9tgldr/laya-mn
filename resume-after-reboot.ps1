# ASCII only. Run this once AFTER rebooting: brings DSH web back and restarts Laya-MN training.
# Training itself is a detached process, so it keeps running if you close the browser.
$root = "C:\Users\hi\Documents\laya-mn"

# 1) DSH web GUI (best effort - skip if you start it another way)
Start-Process -FilePath "node" `
  -ArgumentList "C:\Users\hi\deepseek-harness\apps\cli\lib\bin.js","web","--port","3080" `
  -WindowStyle Hidden
Start-Sleep -Seconds 3

# 2) Laya-MN continued training (detached, runs until a STOP file appears)
Start-Process -FilePath "powershell.exe" `
  -ArgumentList "-NoProfile","-ExecutionPolicy","Bypass","-File","$root\continue.ps1" `
  -WindowStyle Hidden

Write-Host ""
Write-Host "DSH web     : http://127.0.0.1:3080"
Write-Host "Train log   : $root\logs\continue.log"
Write-Host "Stop training: New-Item -ItemType File $root\STOP"
Write-Host ""
