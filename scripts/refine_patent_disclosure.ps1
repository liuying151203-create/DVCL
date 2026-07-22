param(
    [Parameter(Mandatory = $true)]
    [string]$DocPath,
    [string]$PdfPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Replace-DocumentText {
    param(
        [Parameter(Mandatory = $true)]$Document,
        [Parameter(Mandatory = $true)][string]$OldText,
        [Parameter(Mandatory = $true)][string]$NewText
    )

    $find = $Document.Content.Find
    $find.ClearFormatting()
    $find.Replacement.ClearFormatting()
    $null = $find.Execute($OldText, $false, $false, $false, $false, $false, $true, 1, $false, $NewText, 2)
}

function Replace-HeaderFooterText {
    param(
        [Parameter(Mandatory = $true)]$Document,
        [Parameter(Mandatory = $true)][string]$OldText,
        [Parameter(Mandatory = $true)][string]$NewText
    )

    foreach ($section in $Document.Sections) {
        foreach ($headerFooter in @(
            $section.Headers.Item(1),
            $section.Headers.Item(2),
            $section.Headers.Item(3),
            $section.Footers.Item(1),
            $section.Footers.Item(2),
            $section.Footers.Item(3)
        )) {
            $find = $headerFooter.Range.Find
            $find.ClearFormatting()
            $find.Replacement.ClearFormatting()
            $null = $find.Execute($OldText, $false, $false, $false, $false, $false, $true, 1, $false, $NewText, 2)
        }
    }
}

function Convert-FormulaParagraph {
    param(
        [Parameter(Mandatory = $true)]$Document,
        [Parameter(Mandatory = $true)][int]$Number,
        [Parameter(Mandatory = $true)][string]$LinearFormula
    )

    $numberToken = "{0}{1}{2}" -f [char]0xFF08, $Number, [char]0xFF09
    for ($i = 1; $i -le $Document.Paragraphs.Count; $i++) {
        $paragraph = $Document.Paragraphs.Item($i)
        $range = $paragraph.Range
        $text = $range.Text
        if ($text.IndexOf($numberToken, [System.StringComparison]::Ordinal) -lt 0) {
            continue
        }
        if ($range.OMaths.Count -gt 0) {
            return $false
        }

        $formulaEnd = $range.Start + $text.IndexOf($numberToken, [System.StringComparison]::Ordinal)
        while ($formulaEnd -gt $range.Start -and [char]::IsWhiteSpace($Document.Range($formulaEnd - 1, $formulaEnd).Text[0])) {
            $formulaEnd--
        }
        $formulaRange = $Document.Range($range.Start, $formulaEnd)
        $formulaRange.Text = $LinearFormula
        $mathRange = $Document.OMaths.Add($formulaRange)
        $mathRange.OMaths.BuildUp()

        $paragraph.Range.ParagraphFormat.Alignment = 1
        $paragraph.Range.ParagraphFormat.LeftIndent = 0
        $paragraph.Range.ParagraphFormat.FirstLineIndent = 0
        return $true
    }
    throw "Formula ($Number) was not found."
}

function Fix-ProcessStepIndent {
    param([Parameter(Mandatory = $true)]$Document)

    $indent = $Document.Application.CentimetersToPoints(1.45)
    for ($i = 1; $i -le $Document.Paragraphs.Count; $i++) {
        $paragraph = $Document.Paragraphs.Item($i)
        $range = $paragraph.Range
        $text = $range.Text.TrimEnd([char]13, [char]7)
        if ($text -notmatch '^(S[1-8]\.)(\s+)(.+)$') {
            continue
        }

        $marker = $matches[1]
        $content = $matches[3]
        $range.Text = "$marker`t$content`r"
        $format = $paragraph.Range.ParagraphFormat
        $format.LeftIndent = $indent
        $format.FirstLineIndent = -$indent
        $format.TabStops.ClearAll()
        $null = $format.TabStops.Add($indent, 0, 0)
    }
}

$oldTitle = '一种基于双视图对比学习的鲁棒异构图节点分类方法、系统、设备及存储介质'
$newTitle = '一种基于双视图对比学习的鲁棒异构图节点分类方法'
$effectsHeading = '相对现有技术的优点和效果'
$systemSectionSuffix = '系统、设备及存储介质方案'

$formulas = [ordered]@{
    9  = '\tilde{x}_i=\frac{x_i}{\lVert x_i\rVert_2}'
    10 = '\operatorname{sim}(i,j)=\tilde{x}_i^{\mathsf{T}}\tilde{x}_j'
    11 = '\mathcal{N}_k(i)=\operatorname{TopK}_{j\ne i}\operatorname{sim}(i,j)'
    12 = 'E_{\mathrm{feat}}=\{(i,j)\mid j\in\mathcal{N}_k(i)\},\quad\mathcal{G}_{\mathrm{feat}}=(\mathcal{V}_t,E_{\mathrm{feat}})'
    15 = 'z_i=z_i^{\mathrm{topo}}\Vert z_i^{\mathrm{feat}}'
    16 = 'o_i=W_cz_i+b_c'
    17 = '\mathcal{L}_{\mathrm{cls}}=-\frac{1}{|\mathcal{V}_{\mathrm{tr}}|}\sum_{i\in\mathcal{V}_{\mathrm{tr}}}\log\frac{\exp(o_{i,y_i})}{\sum_{c=1}^{C}\exp(o_{i,c})}'
    18 = '\mathcal{L}_{t\rightarrow f}=-\frac{1}{|\mathcal{V}_{\mathrm{tr}}|}\sum_i\log\frac{\exp((\tilde{z}_i^{\mathrm{topo}})^{\mathsf{T}}\tilde{z}_i^{\mathrm{feat}}/\tau)}{\sum_j\exp((\tilde{z}_i^{\mathrm{topo}})^{\mathsf{T}}\tilde{z}_j^{\mathrm{feat}}/\tau)}'
    19 = '\mathcal{L}_{\mathrm{cl}}=\frac{1}{2}(\mathcal{L}_{t\rightarrow f}+\mathcal{L}_{f\rightarrow t})'
    20 = '\mathcal{L}=\lambda_{\mathrm{HAN}}\mathcal{L}_{\mathrm{HAN}}+\mathcal{L}_{\mathrm{cls}}+\lambda_{\mathrm{DVCL}}\mathcal{L}_{\mathrm{cl}}'
}

$word = $null
$document = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $word.AutomationSecurity = 3
    $document = $word.Documents.Open((Resolve-Path $DocPath).Path, $false, $false, $false)

    Replace-DocumentText $document $oldTitle $newTitle
    Replace-DocumentText $document '方法、系统、电子设备及计算机可读存储介质' '方法'
    Replace-DocumentText $document '实施例三：训练服务与在线推理部署' '实施例三：训练与推理过程'
    Replace-DocumentText $document '系统可部署为离线训练服务和在线或批量推理服务。' '本实施例包括离线训练和在线或批量推理两个阶段。'
    Replace-DocumentText $document '离线训练服务读取' '离线训练阶段读取'
    Replace-DocumentText $document '推理服务加载' '推理阶段加载'
    Replace-DocumentText $document '与上述方法对应的系统模块、电子设备和计算机可读存储介质，以及' '上述方法的'
    Replace-DocumentText $document '另设置系统、电子设备和存储介质独立权利要求。' '从属权利要求可进一步限定相应替换方案。'

    $sectionStart = $null
    $sectionEnd = $null
    for ($i = 1; $i -le $document.Paragraphs.Count; $i++) {
        $paragraph = $document.Paragraphs.Item($i)
        $text = $paragraph.Range.Text.TrimEnd([char]13, [char]7)
        if ($sectionStart -eq $null -and $text.StartsWith('8. ') -and $text.EndsWith($systemSectionSuffix)) {
            $sectionStart = $paragraph.Range.Start
            continue
        }
        if ($sectionStart -ne $null -and $text -eq $effectsHeading) {
            $sectionEnd = $paragraph.Range.Start
            break
        }
    }
    if ($sectionStart -ne $null -and $sectionEnd -ne $null) {
        $document.Range($sectionStart, $sectionEnd).Delete()
    }

    foreach ($entry in $formulas.GetEnumerator()) {
        $null = Convert-FormulaParagraph -Document $document -Number $entry.Key -LinearFormula $entry.Value
    }
    Fix-ProcessStepIndent -Document $document

    Replace-HeaderFooterText $document '鲁棒异构图节点分类方法及系统' '鲁棒异构图节点分类方法'

    $document.Save()
    if ($PdfPath) {
        $document.ExportAsFixedFormat((Resolve-Path (Split-Path $PdfPath -Parent)).Path + '\' + (Split-Path $PdfPath -Leaf), 17)
    }
    $document.Close($true)
    $word.Quit()
} catch {
    [Console]::Error.WriteLine("Patent refinement failed at: $($_.InvocationInfo.PositionMessage)")
    [Console]::Error.WriteLine($_.Exception.ToString())
    throw
} finally {
    if ($document -ne $null) { try { $document.Close($false) } catch {} }
    if ($word -ne $null) { try { $word.Quit() } catch {} }
}
