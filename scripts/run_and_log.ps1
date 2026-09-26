function Run-Logged {
    param(
        [Parameter(Mandatory=$true)][string]$Command,
        [Parameter(Mandatory=$true)][string]$LogFile
    )
    $logDir = Split-Path $LogFile -Parent
    if ($logDir -and -not (Test-Path $logDir)) {
        New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    }
    $timestamp = (Get-Date).ToString("s")
    $header = "PS $PWD> $Command"
    "[$timestamp] $header" | Out-File -FilePath $LogFile -Append -Encoding utf8
    Write-Host $header -ForegroundColor Cyan
    Invoke-Expression $Command 2>&1 | Tee-Object -FilePath $LogFile -Append
    $exitTimestamp = (Get-Date).ToString("s")
    "[$exitTimestamp] (exit code: $LASTEXITCODE)" | Out-File -FilePath $LogFile -Append -Encoding utf8
}
