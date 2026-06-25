# Morning refresh: pull latest results -> grade settled bets/predictions -> reports.
# Mostly free; the one paid step is the player-props freeze (pulls goalscorer odds,
# ~2 credits/game) which locks today's OPENING prices so the CLV gap vs the
# near-kickoff close in daily_close.ps1 is measurable. Comment that line out to keep
# the morning run fully free. The match-odds closing snapshot lives in daily_close.ps1.
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
# player props (anytime goalscorer): freeze today's opening prices (~2 credits/game),
# grade who scored from the results data, write the calibration/edge report
python src/props_track.py freeze --date $stamp 2>&1 | Out-File $report -Append -Encoding utf8
python src/props_track.py grade     2>&1 | Out-File $report -Append -Encoding utf8
python src/props_track.py report    2>&1 | Out-File $report -Append -Encoding utf8
# betting/CLV (optional, secondary)
python src/slate.py grade           2>&1 | Out-File $report -Append -Encoding utf8
python src/slate.py report          2>&1 | Out-File $report -Append -Encoding utf8
