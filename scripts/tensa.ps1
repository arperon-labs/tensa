# scripts/tensa.ps1 — Start/stop/status for the TENSA API server
# Usage: .\scripts\tensa.ps1 [status|start|stop|restart|build|rebuild|logs]
#
# This is the public/community-facing launcher. It manages a single service —
# the TENSA REST API server — and writes pidfiles + logs to .pids/ and .logs/
# under the repo root.

param(
    [string]$Command = "status"
)

$ScriptDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$PidDir = Join-Path $ScriptDir ".pids"
$LogDir = Join-Path $ScriptDir ".logs"
New-Item -ItemType Directory -Force -Path $PidDir | Out-Null
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$ServiceName = "api"
$ServicePort = "4350"
$Features = "server,studio-chat,embedding,inference,web-ingest,docparse,generation,adversarial,gemini,bedrock,mcp"

# --- Helpers ---

function Test-ServiceRunning {
    $pidFile = Join-Path $PidDir "$ServiceName.pid"
    if (Test-Path $pidFile) {
        $svcPid = [int](Get-Content $pidFile -Raw).Trim()
        try {
            $proc = Get-Process -Id $svcPid -ErrorAction Stop
            if (-not $proc.HasExited) { return $true }
        } catch {}
        Remove-Item $pidFile -Force
    }
    return $false
}

function Get-ServicePid {
    $pidFile = Join-Path $PidDir "$ServiceName.pid"
    if (Test-Path $pidFile) { return (Get-Content $pidFile -Raw).Trim() }
    return $null
}

# --- Service control ---

function Start-Api {
    if (Test-ServiceRunning) {
        Write-Host "  API already running (pid $(Get-ServicePid))" -ForegroundColor Yellow
        return
    }
    Write-Host "  Starting TENSA API on :$ServicePort..." -ForegroundColor Cyan

    $logFile = (Join-Path $LogDir "api.log") -replace '\\','/'
    $errFile = (Join-Path $LogDir "api.err.log") -replace '\\','/'
    $proc = Start-Process -FilePath "cmd.exe" `
        -ArgumentList "/c","set TENSA_ADDR=0.0.0.0:$ServicePort&& cargo run --release --bin tensa-server --features $Features >$logFile 2>$errFile" `
        -WorkingDirectory $ScriptDir `
        -PassThru -WindowStyle Hidden

    $proc.Id | Out-File -FilePath (Join-Path $PidDir "$ServiceName.pid") -NoNewline

    Start-Sleep -Seconds 3
    try {
        $check = Get-Process -Id $proc.Id -ErrorAction Stop
        if (-not $check.HasExited) {
            Write-Host "  API started (pid $($proc.Id)) -> http://localhost:$ServicePort" -ForegroundColor Green
            return
        }
    } catch {}
    Write-Host "  API failed to start - check .logs/api.log and .logs/api.err.log" -ForegroundColor Red
    Remove-Item (Join-Path $PidDir "$ServiceName.pid") -Force -ErrorAction SilentlyContinue
}

function Stop-Api {
    $hadPid = $false
    if (Test-ServiceRunning) {
        $hadPid = $true
        $svcPid = [int](Get-ServicePid)
        Write-Host "  Stopping API (pid $svcPid)..." -ForegroundColor Cyan
        try {
            taskkill /F /T /PID $svcPid 2>&1 | Out-Null
        } catch {
            try { Stop-Process -Id $svcPid -Force -ErrorAction Stop } catch {}
        }
        Remove-Item (Join-Path $PidDir "$ServiceName.pid") -Force -ErrorAction SilentlyContinue
    }

    # Sweep any orphaned tensa-server.exe processes (cargo's grandchild can outlive cmd.exe).
    $procs = Get-Process -Name "tensa-server" -ErrorAction SilentlyContinue
    if ($procs) {
        Write-Host "  Sweeping stray tensa-server.exe ($($procs.Count) proc(s))" -ForegroundColor DarkYellow
        taskkill /F /IM "tensa-server.exe" 2>&1 | Out-Null
        $hadPid = $true
    }

    if ($hadPid) { Write-Host "  API stopped" -ForegroundColor Green }
    else         { Write-Host "  API not running" -ForegroundColor Yellow }
}

function Show-Status {
    Write-Host ""
    Write-Host "  TENSA API"
    Write-Host "  ------------------------------------"
    if (Test-ServiceRunning) {
        Write-Host "  [running] pid=$(Get-ServicePid)  http://localhost:$ServicePort" -ForegroundColor Green
        Write-Host ""
        Write-Host "  Logs: .logs\api.log" -ForegroundColor Cyan
        Write-Host "  Tail: Get-Content .logs\api.log -Wait" -ForegroundColor Cyan
    } else {
        Write-Host "  [stopped]" -ForegroundColor Red
    }

    Write-Host ""
    Write-Host "  Features"
    Write-Host "  ------------------------------------"
    Write-Host "  $Features" -ForegroundColor DarkGray
    Write-Host ""
    Write-Host "  Commands"
    Write-Host "  ------------------------------------"
    Write-Host "  .\scripts\tensa.ps1 status"
    Write-Host "  .\scripts\tensa.ps1 start | stop | restart"
    Write-Host "  .\scripts\tensa.ps1 build | rebuild"
    Write-Host "  .\scripts\tensa.ps1 logs"
    Write-Host ""
}

# --- Build ---

function Build-Api {
    Write-Host ""
    Write-Host "  Building TENSA (release, full feature set)..." -ForegroundColor Cyan
    Push-Location $ScriptDir
    & cargo build --release --bin tensa-server --features $Features
    $exitCode = $LASTEXITCODE
    Pop-Location
    if ($exitCode -eq 0) {
        Write-Host "  Build succeeded" -ForegroundColor Green
    } else {
        Write-Host "  Build failed" -ForegroundColor Red
    }
    return $exitCode
}

# --- Main ---

switch ($Command) {
    { $_ -in "status", "s" } { Show-Status }
    "start"    { Write-Host ""; Start-Api; Write-Host ""; Show-Status }
    "stop"     { Write-Host ""; Stop-Api;  Write-Host ""; Show-Status }
    "restart"  { Write-Host ""; Stop-Api;  Start-Api; Write-Host ""; Show-Status }
    "build"    { Build-Api; Write-Host "" }
    "rebuild"  {
        Write-Host ""; Stop-Api
        if ((Build-Api) -eq 0) { Start-Api }
        Write-Host ""; Show-Status
    }
    "logs"     { Get-Content (Join-Path $LogDir "api.log") -Wait }
    default    {
        Write-Host ""
        Write-Host "  Usage: .\scripts\tensa.ps1 [command]"
        Write-Host ""
        Write-Host "  Commands:"
        Write-Host "    status   Show service status (default)"
        Write-Host "    start    Build (if needed) and start the API"
        Write-Host "    stop     Stop the API"
        Write-Host "    restart  Stop + start"
        Write-Host "    build    cargo build --release with the full feature set"
        Write-Host "    rebuild  Stop + build + start (after code changes)"
        Write-Host "    logs     Tail the API log"
        Write-Host ""
    }
}
