<#
.SYNOPSIS
  Production server for flashbulljev (Windows): env defaults, file logging, persistent cache.
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts/serve_production.ps1
#>
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

New-Item -ItemType Directory -Force -Path "logs" | Out-Null
New-Item -ItemType Directory -Force -Path "backups" | Out-Null

if (-not $env:FLASHBULLJEV_BACKEND) { $env:FLASHBULLJEV_BACKEND = "ollama" }
if (-not $env:FLASHBULLJEV_MODEL) { $env:FLASHBULLJEV_MODEL = "qwen2.5:1.5b" }
if (-not $env:FLASHBULLJEV_CACHE_DB) { $env:FLASHBULLJEV_CACHE_DB = "flashbulljev.db" }
if (-not $env:FLASHBULLJEV_RPS) { $env:FLASHBULLJEV_RPS = "2" }
if (-not $env:PYTHONPATH) { $env:PYTHONPATH = "src" }

if (-not $env:FLASHBULLJEV_API_KEY) {
  Write-Warning "FLASHBULLJEV_API_KEY not set: API auth is OFF. Set it before exposing this port."
}

try { Invoke-RestMethod http://127.0.0.1:11434/api/tags -TimeoutSec 5 | Out-Null }
catch { Write-Warning "Ollama is not reachable at 127.0.0.1:11434. Start Ollama first." }

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$log = "logs/flashbulljev-$stamp.log"
Write-Output "flashbulljev production server -> http://127.0.0.1:8018 (log: $log)"
python -m uvicorn flashbulljev.api:app --host 127.0.0.1 --port 8018 >> $log 2>&1
