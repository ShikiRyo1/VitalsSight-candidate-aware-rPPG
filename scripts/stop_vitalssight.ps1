$ErrorActionPreference = "Stop"
$Project = Split-Path -Parent $PSScriptRoot
$PidFile = Join-Path $Project "runtime\vitalsight_services.json"
if (-not (Test-Path -LiteralPath $PidFile)) {
    Write-Host "No VitalsSight service pid file was found."
    exit 0
}
$Services = Get-Content -Raw -LiteralPath $PidFile | ConvertFrom-Json
foreach ($PidValue in @($Services.api_pid, $Services.ui_pid)) {
    $Process = Get-Process -Id ([int]$PidValue) -ErrorAction SilentlyContinue
    if ($Process) {
        Stop-Process -Id $Process.Id
        Write-Host "Stopped process $($Process.Id)"
    }
}
$Ports = @($Services.api_port, $Services.ui_port) | Where-Object { $_ }
foreach ($PortValue in $Ports) {
    $Listener = Get-NetTCPConnection -State Listen -LocalPort ([int]$PortValue) -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($Listener) {
        Stop-Process -Id ([int]$Listener.OwningProcess) -Force -ErrorAction SilentlyContinue
        Write-Host "Stopped listener on port $PortValue"
    }
}
Remove-Item -LiteralPath $PidFile -Force
