# docker/build.ps1 — Build the TENSA container image from the repo root.
#
# Usage:
#   .\docker\build.ps1                                  # default tag, full features
#   .\docker\build.ps1 -Tag myreg/tensa:v0.79.2
#   .\docker\build.ps1 -Features "server,mcp"
#   .\docker\build.ps1 -Push

param(
    [string]$Tag = "tensa:latest",
    [string]$Features = "server,studio-chat,inference,web-ingest,docparse,generation,adversarial,gemini,bedrock,mcp,rocksdb,disinfo",
    [switch]$Push
)

$ErrorActionPreference = 'Stop'
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = (Split-Path -Parent $ScriptDir)

Write-Host "  Building $Tag"
Write-Host "  Features: $Features"
Write-Host "  Context:  $RootDir"
Write-Host ""

Push-Location $RootDir
try {
    $env:DOCKER_BUILDKIT = "1"
    docker build `
        --file docker/Dockerfile `
        --build-arg "TENSA_FEATURES=$Features" `
        --tag $Tag `
        .
    if ($LASTEXITCODE -ne 0) { throw "docker build failed (exit $LASTEXITCODE)" }
} finally {
    Pop-Location
}

Write-Host ""
Write-Host "  Built: $Tag" -ForegroundColor Green

if ($Push) {
    Write-Host ""
    Write-Host "  Pushing $Tag..."
    docker push $Tag
}
