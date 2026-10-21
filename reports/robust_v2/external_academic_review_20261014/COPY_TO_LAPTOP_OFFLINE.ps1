[CmdletBinding()]
param(
    [string]$Source = (Join-Path $env:USERPROFILE 'OneDrive\paper\READY_FOR_REVIEW'),
    [string]$RepositoryPath = (Join-Path $env:USERPROFILE 'workspaces\github.com\bengawith\paper'),
    [string]$Destination = (Join-Path ([Environment]::GetFolderPath('Desktop')) 'paper-review-offline')
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Run-Git([string[]]$GitArguments) {
    $answer = & git -C $RepositoryPath @GitArguments 2>&1
    if ($LASTEXITCODE -ne 0) { throw "Git check failed: $($GitArguments -join ' ')`n$answer" }
    return (($answer | Out-String).Trim())
}
function Safe-File([string]$Base, [string]$Relative) {
    if ([IO.Path]::IsPathRooted($Relative) -or $Relative.Contains('\') -or (($Relative -split '/') -contains '..')) {
        throw "Unsafe manifest path: $Relative"
    }
    $prefix = [IO.Path]::GetFullPath($Base).TrimEnd('\') + '\'
    $path = [IO.Path]::GetFullPath((Join-Path $Base $Relative))
    if (-not $path.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) { throw "Path escapes package: $Relative" }
    return $path
}
function Check-Files([string]$Base, $Entries) {
    foreach ($entry in $Entries) {
        $path = Safe-File -Base $Base -Relative $entry.path
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing package file: $($entry.path)" }
        $item = Get-Item -LiteralPath $path -Force
        if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { throw "Reparse-point files are not admitted: $($entry.path)" }
        if ($item.Length -ne $entry.size) { throw "Size mismatch: $($entry.path)" }
        if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.sha256) { throw "Hash mismatch: $($entry.path)" }
    }
    $actual = @(Get-ChildItem -LiteralPath $Base -File -Recurse -Force)
    if ($actual.Count -ne (@($Entries).Count + 1)) { throw 'Unmanifested or missing files in package' }
}

if (Test-Path -LiteralPath $Destination) { throw 'Destination already exists; it will not be overwritten or deleted' }
$Source = [IO.Path]::GetFullPath($Source)
$RepositoryPath = [IO.Path]::GetFullPath($RepositoryPath)
$Destination = [IO.Path]::GetFullPath($Destination)
foreach ($base in @($Source, $RepositoryPath)) {
    if ($Destination.StartsWith($base.TrimEnd('\')+'\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Destination cannot be inside the source package or repository' }
}
$manifestPath = Join-Path $Source 'SHA256_MANIFEST.json'
$manifestHash = (Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($manifest.schema -ne 'academic-review-release-v2') { throw 'Unsupported or historical package manifest; use the current release' }
if ($manifest.source_commit_sha -notmatch '^[0-9a-f]{40}$' -or $manifest.source_remote_sha -ne $manifest.source_commit_sha) { throw 'Invalid or unpushed package commit identity' }
if ((Run-Git -GitArguments @('status','--porcelain=v1','--untracked-files=all')).Length -ne 0) { throw 'Repository has tracked or untracked local changes; nothing will be reset' }
if ((Run-Git -GitArguments @('branch','--show-current')) -ne $manifest.source_branch) { throw 'Current branch differs from the reviewed package branch' }
if ((Run-Git -GitArguments @('rev-parse','HEAD')) -ne $manifest.source_commit_sha) { throw 'Local commit differs from the reviewed package commit; unpublished work will not be discarded' }
$null = Run-Git -GitArguments @('fetch','--quiet','origin',$manifest.source_branch)
if ((Run-Git -GitArguments @('rev-parse',('origin/'+$manifest.source_branch))) -ne $manifest.source_commit_sha) { throw 'Remote branch differs from the reviewed package commit' }
Check-Files -Base $Source -Entries $manifest.files
$null = New-Item -ItemType Directory -Path $Destination
foreach ($entry in $manifest.files) {
    $from = Safe-File -Base $Source -Relative $entry.path
    $to = Safe-File -Base $Destination -Relative $entry.path
    $parent = Split-Path -Parent $to
    if (-not (Test-Path -LiteralPath $parent)) { $null = New-Item -ItemType Directory -Path $parent -Force }
    Copy-Item -LiteralPath $from -Destination $to
}
Copy-Item -LiteralPath $manifestPath -Destination (Join-Path $Destination 'SHA256_MANIFEST.json')
Check-Files -Base $Destination -Entries $manifest.files
if ((Get-FileHash -LiteralPath (Join-Path $Destination 'SHA256_MANIFEST.json') -Algorithm SHA256).Hash -ne $manifestHash) { throw 'Source manifest changed during copying; retain the partial copy for inspection' }
if ((Run-Git -GitArguments @('rev-parse','HEAD')) -ne $manifest.source_commit_sha -or (Run-Git -GitArguments @('branch','--show-current')) -ne $manifest.source_branch -or (Run-Git -GitArguments @('status','--porcelain=v1','--untracked-files=all')).Length -ne 0) { throw 'Repository changed during copying; review before using the copy' }
[PSCustomObject]@{status='VERIFIED_OFFLINE_REVIEW_COPY'; commit=$manifest.source_commit_sha; destination=$Destination; verified_files=@($manifest.files).Count; repository_modified=$false} | ConvertTo-Json
