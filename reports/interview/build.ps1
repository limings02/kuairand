$ErrorActionPreference = "Stop"

$rootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $rootDir

$outDir = Join-Path $rootDir "output"
$qmdFile = Join-Path $rootDir "interview_deck.qmd"
$htmlOut = Join-Path $outDir "kuairand_interview.html"
$pdfOut = Join-Path $outDir "kuairand_interview.pdf"

New-Item -ItemType Directory -Path $outDir -Force | Out-Null

function Get-Tool {
    param([string]$Name)
    Get-Command $Name -ErrorAction SilentlyContinue | Select-Object -First 1
}

function Print-Status {
    param(
        [string]$Label,
        $Tool
    )
    if ($null -eq $Tool) {
        Write-Host ("[check] {0,-7}: NOT FOUND" -f $Label)
        return
    }

    $versionLine = ""
    try {
        $versionLine = (& $Tool.Source --version 2>$null | Select-Object -First 1)
    } catch {
        $versionLine = ""
    }

    if ([string]::IsNullOrWhiteSpace($versionLine)) {
        Write-Host ("[check] {0,-7}: {1}" -f $Label, $Tool.Source)
    } else {
        Write-Host ("[check] {0,-7}: {1}" -f $Label, $versionLine)
    }
}

$quarto = Get-Tool -Name "quarto"
$pandoc = Get-Tool -Name "pandoc"
$xelatex = Get-Tool -Name "xelatex"

Write-Host "[check] project dir: $rootDir"
Write-Host "[check] output dir : $outDir"
Print-Status -Label "quarto" -Tool $quarto
Print-Status -Label "pandoc" -Tool $pandoc
Print-Status -Label "xelatex" -Tool $xelatex

if ($null -ne $quarto) {
    Write-Host "[build] render revealjs HTML..."
    & $quarto.Source render $qmdFile --to revealjs --output-dir $outDir --output kuairand_interview.html

    Write-Host "[build] render beamer PDF..."
    & $quarto.Source render $qmdFile --to beamer --output-dir $outDir --output kuairand_interview.pdf

    Write-Host "[done] HTML: $htmlOut"
    Write-Host "[done] PDF : $pdfOut"
    exit 0
}

Write-Host ""
Write-Host "[warn] quarto not found, using fallback mode."
Write-Host "[warn] revealjs + beamer same-source output requires quarto."

if ($null -ne $pandoc) {
    Write-Host "[fallback] export HTML with pandoc..."
    & $pandoc.Source @(
        "-f", "markdown", $qmdFile, "-s",
        "-o", $htmlOut,
        "--toc",
        "-c", "assets/style.css",
        "-V", "lang=zh"
    )
    Write-Host "[fallback] HTML: $htmlOut"
} else {
    Write-Host "[error] pandoc not found, cannot export HTML."
}

if ($null -ne $pandoc -and $null -ne $xelatex) {
    Write-Host "[fallback] export PDF with pandoc + xelatex..."
    & $pandoc.Source @(
        "-f", "markdown", $qmdFile, "-s",
        "-o", $pdfOut,
        "--pdf-engine=$($xelatex.Source)",
        "-H", "assets/latex-header.tex"
    )
    Write-Host "[fallback] PDF : $pdfOut"
} else {
    Write-Host "[error] missing pandoc or xelatex, cannot export PDF."
}

Write-Host ""
Write-Host "[hint] if you install quarto later:"
Write-Host "       quarto render interview_deck.qmd --to revealjs --output-dir output --output kuairand_interview.html"
Write-Host "       quarto render interview_deck.qmd --to beamer  --output-dir output --output kuairand_interview.pdf"
