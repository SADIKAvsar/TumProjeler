param(
    [string]$Message = "",
    [switch]$NoPush
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

$status = git status --short
if (-not $status) {
    Write-Host "[git_autosync] Degisiklik yok. Cikis." -ForegroundColor Yellow
    exit 0
}

if ([string]::IsNullOrWhiteSpace($Message)) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $Message = "chore: autosync $ts"
}

Write-Host "[git_autosync] Stage: git add -A" -ForegroundColor Cyan
git add -A

Write-Host "[git_autosync] Commit: $Message" -ForegroundColor Cyan
git commit -m $Message

if ($NoPush) {
    Write-Host "[git_autosync] Push atlandi (--NoPush)." -ForegroundColor Yellow
    exit 0
}

$branch = git rev-parse --abbrev-ref HEAD
Write-Host "[git_autosync] Push: origin/$branch" -ForegroundColor Cyan
git push origin $branch

Write-Host "[git_autosync] Tamamlandi." -ForegroundColor Green
