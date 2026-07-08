$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$sourceDir = Join-Path $repoRoot "paper\dvcl"
$exportRoot = Join-Path $repoRoot "paper\_exports"
$exportDir = Join-Path $exportRoot "dvcl"
$outputZip = Join-Path $repoRoot "paper\dvcl.zip"

if (-not (Test-Path $sourceDir)) {
    throw "Missing source directory: $sourceDir"
}

if (Test-Path $exportDir) {
    Remove-Item $exportDir -Recurse -Force
}

New-Item -ItemType Directory -Force $exportDir | Out-Null
Copy-Item -Path (Join-Path $sourceDir "main.tex") -Destination $exportDir
Copy-Item -Path (Join-Path $sourceDir "references.bib") -Destination $exportDir
Copy-Item -Path (Join-Path $sourceDir "sections") -Destination (Join-Path $exportDir "sections") -Recurse

if (Test-Path $outputZip) {
    Remove-Item $outputZip -Force
}

Compress-Archive -Path (Join-Path $exportDir "*") -DestinationPath $outputZip -Force
Write-Host "Created $outputZip"
