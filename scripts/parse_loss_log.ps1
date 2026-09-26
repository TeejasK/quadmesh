# Turns lines like:
#   [shared/pretrain] {"step": 20, "loss": 9.5089, "grad_norm": 1.832, "lr": 4e-05, "tok_per_s": 2856}
# into a clean CSV and a Markdown table, ready to paste into documentation.
#
# Usage (after training, from a saved console log):
#   .\scripts\parse_loss_log.ps1 -InFile pretrain_console.log -OutPrefix pretrain_3B
#
# Usage (live, while training runs — captures AND parses at once, one terminal, no --detach needed):
#   modal run modal_app.py::shared_pretrain --tier 3B --budget chinchilla --mb 16 --accum 16 --grad_ckpt 0 --workers 12 |
#     .\scripts\parse_loss_log.ps1 -OutPrefix pretrain_3B -Live

param(
    [string]$InFile = "",
    [string]$OutPrefix = "loss_log",
    [switch]$Live
)

$csvPath = "$OutPrefix.csv"
$mdPath = "$OutPrefix.md"

"step,loss,grad_norm,lr,tok_per_s" | Out-File -FilePath $csvPath -Encoding utf8
$rows = New-Object System.Collections.Generic.List[object]

function Process-Line($line) {
    # Pull out the {...} JSON regardless of what prefix ([shared/pretrain], [shared/sft], etc.) comes before it.
    if ($line -match '(\{.*"step".*\})') {
        try {
            $r = $matches[1] | ConvertFrom-Json
            "$($r.step),$($r.loss),$($r.grad_norm),$($r.lr),$($r.tok_per_s)" | Add-Content $csvPath
            $rows.Add($r) | Out-Null
        } catch { }
    }
}

if ($Live) {
    Write-Host "Reading live pipeline input, writing $csvPath as steps arrive..."
    $input | ForEach-Object {
        Write-Host $_          # still show the normal console output
        Process-Line $_
    }
} else {
    if ($InFile -eq "" -or -not (Test-Path $InFile)) {
        Write-Host "Pass -InFile <path to saved console log> or use -Live in a pipeline. See script header for usage."
        exit 1
    }
    Get-Content $InFile | ForEach-Object { Process-Line $_ }
}

if ($rows.Count -eq 0) {
    Write-Host "No step lines found."
    exit 0
}

# Markdown table for documentation
$md = New-Object System.Text.StringBuilder
[void]$md.AppendLine("| Step | Loss | Grad Norm | LR | Tok/s |")
[void]$md.AppendLine("|---|---|---|---|---|")
foreach ($r in $rows) {
    [void]$md.AppendLine("| $($r.step) | $($r.loss) | $($r.grad_norm) | $($r.lr) | $($r.tok_per_s) |")
}
$md.ToString() | Out-File -FilePath $mdPath -Encoding utf8

Write-Host "$($rows.Count) steps parsed."
Write-Host "CSV:      $csvPath"
Write-Host "Markdown: $mdPath"
