# NTRVSTA Interview Engine — start the engine + the web app together.
#   pwsh -File run.ps1
# Engine: http://localhost:8000   Web: http://localhost:5173

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

# Prefer the project venv (AI-Interview\myenv) if it's there, else PATH python.
$py = "python"
foreach ($cand in @(
  (Join-Path $root "..\myenv\Scripts\python.exe"),
  (Join-Path $root ".venv\Scripts\python.exe")
)) {
  if (Test-Path $cand) { $py = $cand; break }
}
Write-Host "Using Python: $py" -ForegroundColor DarkGray

Write-Host "Starting the interview engine on :8000 ..." -ForegroundColor Cyan
$engine = Start-Process -PassThru -WorkingDirectory $root `
  -FilePath $py -ArgumentList "-m", "engine.api.main"

Write-Host "Starting the web app on :5173 ..." -ForegroundColor Cyan
$web = Start-Process -PassThru -WorkingDirectory (Join-Path $root "web") `
  -FilePath "npm" -ArgumentList "run", "dev"

Start-Sleep -Seconds 4
Start-Process "http://localhost:5173"

Write-Host ""
Write-Host "Both running. The engine warms its speech models for ~60-90s on first start." -ForegroundColor Green
Write-Host "Wait for 'Application startup complete' before starting an interview." -ForegroundColor Green
Write-Host "Press Ctrl+C here to stop both." -ForegroundColor Green

try {
  Wait-Process -Id $engine.Id, $web.Id
} finally {
  foreach ($p in @($engine, $web)) {
    if ($p -and -not $p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
  }
}
