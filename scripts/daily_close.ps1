# Near-kickoff closing-line snapshot. Scheduled to run SEVERAL times across the
# match-day window (see the multi-trigger soccer-close task): odds.py close
# overwrites pending bets with the current best price, so the *last run before a
# game kicks off* sets its true closing line (games already started drop out and
# keep their last pre-kickoff value). ~3 API credits per run; props close ~2/game.
# Appends to one daily log so you can see how the snapshots refined toward kickoff.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot          # ...\soccer_model
Set-Location $root
$logDir = Join-Path $root "logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }
$stamp = Get-Date -Format "yyyy-MM-dd"
$report = Join-Path $logDir "close_$stamp.txt"

"=== close snapshot $(Get-Date -Format 'yyyy-MM-dd HH:mm') ===" | Out-File $report -Append -Encoding utf8
python src/odds.py close 2>&1 | Out-File $report -Append -Encoding utf8
# player props: snapshot closing goalscorer prices for pending rows (CLV vs freeze)
python src/props_track.py close --date $stamp 2>&1 | Out-File $report -Append -Encoding utf8
