# Run this in a SECOND PowerShell window while your training command runs in the first one
# (foreground, no --detach needed). Polls the real metrics file the trainer writes on the
# volume every log_every steps, and appends any new steps to a local CSV - the same numbers
# the notebook's polling cells show, just saved to disk for your documentation.
#
# Usage:
#   .\scripts\poll_loss.ps1 -Tier 3B -Phase pretrain
#   .\scripts\poll_loss.ps1 -Tier 3B -Phase sft -IntervalSec 60

param(
    [string]$Tier = "3B",
    [ValidateSet("pretrain", "sft")]
    [string]$Phase = "pretrain",
    [int]$IntervalSec = 120,
    [string]$OutCsv = ""
)

if ($OutCsv -eq "") { $OutCsv = "loss_log_${Phase}_${Tier}.csv" }
$remote = "/Quadmesh-$Tier/live/metrics_$Phase.jsonl"
$tmp = "_metrics_${Phase}_${Tier}.jsonl"

if (-not (Test-Path $OutCsv)) {
    "step,loss,grad_norm,lr,tok_per_s,polled_at" | Out-File -FilePath $OutCsv -Encoding utf8
}

$seen = @{}
if (Test-Path $OutCsv) {
    Import-Csv $OutCsv | ForEach-Object { $seen[$_.step] = $true }
}

Write-Host "Polling metrics_$Phase.jsonl for Quadmesh-$Tier every $IntervalSec s -> $OutCsv"
Write-Host "Press Ctrl+C to stop (the CSV keeps everything already polled)."

while ($true) {
    try {
        modal volume get quadmesh-ckpts $remote $tmp --force 2>$null | Out-Null
        if (Test-Path $tmp) {
            $newCount = 0
            Get-Content $tmp | ForEach-Object {
                if ($_.Trim() -eq "") { return }
                try { $r = $_ | ConvertFrom-Json } catch { return }
                if ($null -ne $r.step -and -not $seen.ContainsKey([string]$r.step)) {
                    $seen[[string]$r.step] = $true
                    $now = (Get-Date).ToString("o")
                    "$($r.step),$($r.loss),$($r.grad_norm),$($r.lr),$($r.tok_per_s),$now" | Add-Content $OutCsv
                    $newCount++
                }
            }
            Write-Host "$(Get-Date -Format T)  +$newCount new steps  ($($seen.Count) total logged)"
        }
    } catch {
        Write-Host "$(Get-Date -Format T)  poll failed (file may not exist yet): $_"
    }
    Start-Sleep -Seconds $IntervalSec
}
