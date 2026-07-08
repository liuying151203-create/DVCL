$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$readingDir = Join-Path $repoRoot "paper\dvcl_reading"
$englishDir = Join-Path $repoRoot "paper\dvcl"
$exportRoot = Join-Path $repoRoot "paper\_exports"
$exportDir = Join-Path $exportRoot "dvcl_reading"
$outputZip = Join-Path $repoRoot "paper\dvcl_reading.zip"

if (-not (Test-Path $readingDir)) {
    throw "Missing reading directory: $readingDir"
}

if (-not (Test-Path $englishDir)) {
    throw "Missing English source directory: $englishDir"
}

if (Test-Path $exportDir) {
    Remove-Item $exportDir -Recurse -Force
}

New-Item -ItemType Directory -Force $exportDir | Out-Null
Copy-Item -Path (Join-Path $readingDir "*") -Destination $exportDir -Recurse
Copy-Item -Path (Join-Path $englishDir "sections") -Destination (Join-Path $exportDir "sections") -Recurse
Copy-Item -Path (Join-Path $englishDir "references.bib") -Destination (Join-Path $exportDir "references.bib")

$mainZh = Join-Path $exportDir "main_zh.tex"
$mainBilingual = Join-Path $exportDir "main_bilingual.tex"

(Get-Content $mainZh -Raw -Encoding UTF8).Replace("../dvcl/references", "references") |
    Set-Content $mainZh -Encoding UTF8

(Get-Content $mainBilingual -Raw -Encoding UTF8).
    Replace("../dvcl/sections", "sections").
    Replace("../dvcl/references", "references") |
    Set-Content $mainBilingual -Encoding UTF8

if (Test-Path $outputZip) {
    Remove-Item $outputZip -Force
}

Compress-Archive -Path (Join-Path $exportDir "*") -DestinationPath $outputZip -Force
Write-Host "Created $outputZip"
