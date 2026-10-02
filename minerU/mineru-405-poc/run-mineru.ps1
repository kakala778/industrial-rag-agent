$ErrorActionPreference = 'Stop'

$mineruPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $mineruPython -PathType Leaf)) {
    throw "MinerU 4.0.5 Python environment not found: $mineruPython"
}

$env:MINERU_HOME = Join-Path $PSScriptRoot '.mineru-home'
& $mineruPython -m mineru.kit.main @args
exit $LASTEXITCODE
