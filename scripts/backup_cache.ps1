<#
.SYNOPSIS
  Timestamped backup of the flashbulljev SQLite cache. Keeps the newest 7 copies.
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts/backup_cache.ps1
#>
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$db = if ($env:FLASHBULLJEV_CACHE_DB) { $env:FLASHBULLJEV_CACHE_DB } else { "flashbulljev.db" }
if (-not (Test-Path -LiteralPath $db)) { Write-Output "no cache db at $db, nothing to back up"; exit 0 }

New-Item -ItemType Directory -Force -Path "backups" | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$dest = "backups/flashbulljev-$stamp.db"
Copy-Item -LiteralPath $db -Destination $dest
Get-ChildItem -LiteralPath "backups" -Filter "flashbulljev-*.db" |
  Sort-Object Name -Descending | Select-Object -Skip 7 | Remove-Item -Force
Write-Output "backup saved: $dest"
