# L0P 夜班：五模型依序跑 L0P 條件（預註冊 PREREG_L0P.md 先於本次執行 commit）
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot\..
$py = '.venv\Scripts\python.exe'
New-Item -ItemType Directory -Force 'results\l0p' | Out-Null
$models = @(
  'EleutherAI/pythia-410m',
  'EleutherAI/pythia-1b',
  'EleutherAI/pythia-1.4b',
  'EleutherAI/pythia-2.8b',
  'allenai/OLMo-2-0425-1B'
)
foreach ($m in $models) {
  Write-Host "=== $m ==="
  & $py harness\run_pilot.py --model $m --battery battery\battery_l0p.json `
      --levels L0P --out results\l0p
  if ($LASTEXITCODE -ne 0) { throw "run failed: $m" }
}
Write-Host 'ALL L0P RUNS DONE'
