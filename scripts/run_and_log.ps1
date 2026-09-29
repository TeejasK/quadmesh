# Set UTF-8 encoding across PowerShell and Python so Rich Unicode symbols (like ✓) work cleanly
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

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
    Write-Host $header -ForegroundColor Cyan

    $env:PYTHONIOENCODING = "utf-8"
    $env:PYTHONUTF8 = "1"

    # Run command, stream each line to console, collect all lines
    $lines = [System.Collections.Generic.List[string]]::new()
    $lines.Add("[$timestamp] $header")
    Invoke-Expression $Command 2>&1 | ForEach-Object {
        $lineStr = $_.ToString()
        Write-Host $lineStr
        $lines.Add($lineStr)
    }

    $exitTimestamp = (Get-Date).ToString("s")
    $lines.Add("[$exitTimestamp] (exit code: $LASTEXITCODE)")

    # Write ALL output to file AFTER command finishes, in plain UTF-8 (no BOM)
    $fullPath = Join-Path $PWD $LogFile
    [System.IO.File]::WriteAllLines(
        $fullPath,
        $lines,
        [System.Text.UTF8Encoding]::new($false)
    )
    Write-Host "[LOG] Saved to $LogFile" -ForegroundColor Green
}
