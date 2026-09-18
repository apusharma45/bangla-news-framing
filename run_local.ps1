$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ProjectRoot ".venv312\Scripts\python.exe"
$AppPath = Join-Path $ProjectRoot "app.py"

if (-not (Test-Path -LiteralPath $VenvPython)) {
    throw "Local environment not found. Run .\setup_local.ps1 first."
}

$VenvVersion = & $VenvPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($LASTEXITCODE -ne 0 -or $VenvVersion.Trim() -ne "3.12") {
    throw "The local app environment must use Python 3.12. Rerun .\setup_local.ps1."
}

if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot "artifacts"))) {
    throw "The artifacts folder is missing from $ProjectRoot."
}

Push-Location $ProjectRoot
try {
    & $VenvPython $AppPath
}
finally {
    Pop-Location
}
