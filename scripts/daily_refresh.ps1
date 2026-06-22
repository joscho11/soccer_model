# Morning refresh (no kickoff-timing sensitivity, no extra API cost):
# pull latest results -> grade settled bets -> write the accuracy/CLV report.
# The closing-line snapshot is NOT here -- it must fire near kickoff, so it lives
# in daily_close.ps1 on its own afternoon/evening schedule.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot          # ...\soccer_model
Set-Location $root
$logDir = Join-Path $root "logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }
$stamp = Get-Date -Format "yyyy-MM-dd"
$report = Join-Path $logDir "refresh_$stamp.txt"

"=== daily refresh $(Get-Date -Format 'yyyy-MM-dd HH:mm') ===" | Out-File $report -Encoding utf8
python src/download_data.py --force 2>&1 | Out-File $report -Append -Encoding utf8
# prediction tracker: lock upcoming predictions (pre-kickoff) and grade played games
python src/track.py freeze          2>&1 | Out-File $report -Append -Encoding utf8
python src/track.py grade --live    2>&1 | Out-File $report -Append -Encoding utf8
python src/track.py report          2>&1 | Out-File $report -Append -Encoding utf8
# betting/CLV (optional, secondary)
python src/slate.py grade           2>&1 | Out-File $report -Append -Encoding utf8
python src/slate.py report          2>&1 | Out-File $report -Append -Encoding utf8
