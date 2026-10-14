# Run from within your laptop's existing bengawith/paper Git repository.
# One PowerShell command: & { irm '...' } is NOT needed; instead paste:
# & { . "$PWD\review\READY_FOR_REVIEW\COPY_TO_LAPTOP_OFFLINE.ps1" }
# For the first copy, execute this file from OneDrive, or paste its contents into PowerShell.
$ErrorActionPreference = 'Stop'
$repo = (git rev-parse --show-toplevel 2>$null).Trim()
if ($LASTEXITCODE -ne 0 -or -not $repo) { throw 'Run inside the paper Git repository.' }
$source = Join-Path $env:USERPROFILE 'OneDrive\paper\READY_FOR_REVIEW'
$target = Join-Path $repo 'review\READY_FOR_REVIEW'
$branch = 'robust-v2-viability'
if (-not (Test-Path -LiteralPath (Join-Path $source 'SHA256_MANIFEST.json'))) { throw "OneDrive package not synced: $source" }
$manifest = Get-Content -LiteralPath (Join-Path $source 'SHA256_MANIFEST.json') -Raw | ConvertFrom-Json
foreach ($item in $manifest.files) {
    $file = Join-Path $source $item.path
    if (-not (Test-Path -LiteralPath $file)) { throw "Missing source file: $($item.path)" }
    if ((Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash.ToLowerInvariant() -ne $item.sha256.ToLowerInvariant()) { throw "Source hash mismatch: $($item.path)" }
}
# Fail closed on dirty working tree; do not reset, clean or stash untracked/ignored work.
$dirty = @(git -C $repo status --porcelain --untracked-files=all)
if ($dirty.Count -gt 0) { throw "Local repository contains changes. Preserve them manually first. No files were changed. $($dirty.Count) status entries." }
git -C $repo fetch origin $branch
if ($LASTEXITCODE -ne 0) { throw 'Fetch failed.' }
$remote = (git -C $repo rev-parse "origin/$branch").Trim()
if ($remote -ne $manifest.source_git_commit) { throw "Branch ($remote) differs from package source commit ($($manifest.source_git_commit)). Obtain a matching package; do not mix revisions." }
git -C $repo switch -C $branch "origin/$branch"
if ($LASTEXITCODE -ne 0) { throw 'Switch failed.' }
if (Test-Path -LiteralPath $target) { throw "Destination exists; refusing to overwrite: $target" }
New-Item -ItemType Directory -Path $target -Force | Out-Null
& robocopy.exe $source $target /E /R:2 /W:1 /NFL /NDL /NP
if ($LASTEXITCODE -ge 8) { throw 'Copy failed.' }
foreach ($item in $manifest.files) {
    $file = Join-Path $target $item.path
    if (-not (Test-Path -LiteralPath $file)) { throw "Missing copied file: $($item.path)" }
    if ((Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash.ToLowerInvariant() -ne $item.sha256.ToLowerInvariant()) { throw "Copied hash mismatch: $($item.path)" }
    if ((Get-Item -LiteralPath $file).Length -ne $item.bytes) { throw "Copied size mismatch: $($item.path)" }
}
Write-Host "SUCCESS: Offline review package verified: $target" -ForegroundColor Green
Write-Host "Current branch: $branch; commit: $remote"
Write-Host "Start: $target\READ_THIS_FIRST.md"
