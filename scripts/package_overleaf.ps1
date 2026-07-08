$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$sourceDir = Join-Path $repoRoot "paper\dvcl"
$outputZip = Join-Path $repoRoot "paper\dvcl.zip"

if (-not (Test-Path $sourceDir)) {
    throw "Missing source directory: $sourceDir"
}

if (Test-Path $outputZip) {
    Remove-Item $outputZip -Force
}

Compress-Archive -Path (Join-Path $sourceDir "*") -DestinationPath $outputZip -Force
Write-Host "Created $outputZip"
