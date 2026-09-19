$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$runRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot 'runs/wavelet_nafnet_20260919')).Path
$verification = Get-Content -LiteralPath (Join-Path $runRoot 'verification.json') -Raw | ConvertFrom-Json
if ($verification.public_images -ne 460 -or $verification.test_images -ne 20 -or -not $verification.same_backend_repeat_sha256_match) {
    throw 'Replacement outputs have not passed verification.'
}
$oldFolder = Join-Path $PSScriptRoot 'runs/nafnet/public_rerun'
$manifestPath = Join-Path $runRoot 'replaced_outputs.json'
if (-not (Test-Path -LiteralPath $oldFolder)) { exit 0 }
$oldRoot = (Resolve-Path -LiteralPath $oldFolder).Path
if (-not $oldRoot.StartsWith($repoRoot + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
    throw 'Old output folder escapes the repository.'
}
$deletions = @()
foreach ($file in Get-ChildItem -LiteralPath $oldRoot -File) {
    if ($file.Name -notmatch '^\d{3}\.png$') { throw 'Unexpected file in old output folder.' }
    $resolved = (Resolve-Path -LiteralPath $file.FullName).Path
    if ([System.IO.Path]::GetDirectoryName($resolved) -ne $oldRoot) { throw 'Unsafe output path.' }
    $replacement = Join-Path (Join-Path $runRoot 'public') $file.Name
    if (-not (Test-Path -LiteralPath $replacement)) { throw 'Replacement image is missing.' }
    $deletions += [pscustomobject]@{path=$resolved; sha256=(Get-FileHash -LiteralPath $resolved -Algorithm SHA256).Hash; bytes=$file.Length; replacement=$replacement}
}
$deletions | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
foreach ($entry in $deletions) { Remove-Item -LiteralPath $entry.path }
if (@(Get-ChildItem -LiteralPath $oldRoot -Force).Count -eq 0) { Remove-Item -LiteralPath $oldRoot }
Write-Output ('Replaced {0} old prediction images; originals and model checkpoints retained.' -f $deletions.Count)
$trialFiles = @('smoke_model.pth', 'smoke_model.onnx', 'smoke_cpu/461.png', 'smoke_gpu/461.png')
foreach ($relative in $trialFiles) {
    $candidate = Join-Path $runRoot $relative
    if (Test-Path -LiteralPath $candidate) {
        $resolved = (Resolve-Path -LiteralPath $candidate).Path
        if (-not $resolved.StartsWith($runRoot + '\', [System.StringComparison]::OrdinalIgnoreCase)) { throw 'Unsafe trial path.' }
        Remove-Item -LiteralPath $resolved
    }
}
foreach ($name in @('smoke_cpu', 'smoke_gpu')) {
    $candidate = Join-Path $runRoot $name
    if ((Test-Path -LiteralPath $candidate) -and @(Get-ChildItem -LiteralPath $candidate -Force).Count -eq 0) {
        Remove-Item -LiteralPath $candidate
    }
}
