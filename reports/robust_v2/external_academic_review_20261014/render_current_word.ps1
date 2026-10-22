$ErrorActionPreference = 'Stop'
$Root = 'C:\Users\Ben\workspaces\github.com\bengawith\paper'
$Review = Join-Path $Root 'reports\robust_v2\external_academic_review_20261014'
$Output = Join-Path $Review 'RENDERED'
$null = New-Item -ItemType Directory -Path $Output -Force
$Word = New-Object -ComObject Word.Application
if ($Word.Documents.Count -ne 0) { throw 'The acquired Word instance already has documents; refusing to interfere' }
$Word.Visible = $false
$Word.DisplayAlerts = 0
$Rows = @()
try {
    foreach ($Spec in @(@{input=(Join-Path $Root 'paper\LATEST_MANUSCRIPT_FOR_AVA.docx'); name='manuscript'}, @{input=(Join-Path $Review 'SCIENTIFIC_REVIEW_REPORT.docx'); name='referee_report'})) {
        $Document = $Word.Documents.Open($Spec.input, $false, $true)
        try {
            $PDF = Join-Path $Output ($Spec.name + '.pdf')
            $Document.ExportAsFixedFormat($PDF, 17)
            $Rows += @{document=$Spec.input; pdf=$PDF; pages=$Document.ComputeStatistics(2); renderer='Microsoft Word native PDF export'}
            Write-Output ($Spec.name + ' rendered, pages=' + $Document.ComputeStatistics(2))
        } finally { $Document.Close(0); [void][Runtime.InteropServices.Marshal]::ReleaseComObject($Document) }
    }
} finally { $Word.Quit(); [void][Runtime.InteropServices.Marshal]::ReleaseComObject($Word) }
$Rows | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $Output 'WORD_RENDER.json') -Encoding UTF8
