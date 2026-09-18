$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDirectory = Join-Path $ProjectRoot ".venv312"
$VenvPython = Join-Path $VenvDirectory "Scripts\python.exe"
$Requirements = Join-Path $ProjectRoot "requirements-local.txt"

Push-Location $ProjectRoot
try {
    if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
        throw "Python Launcher (py.exe) was not found. Install 64-bit Python 3.12 first."
    }

    & py -3.12 -c "import sys; assert sys.version_info[:2] == (3, 12)" 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "Python 3.12 is not installed. Install 64-bit Python 3.12, then rerun this script."
    }

    if (Test-Path -LiteralPath $VenvPython) {
        $VenvVersion = & $VenvPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
        if ($LASTEXITCODE -ne 0 -or $VenvVersion.Trim() -ne "3.12") {
            throw "The .venv312 environment is not using Python 3.12. Rename or remove .venv312, then rerun this script."
        }
    }
    else {
        Write-Host "Creating Python 3.12 virtual environment..."
        & py -3.12 -m venv $VenvDirectory
    }

    Write-Host "Updating pip..."
    & $VenvPython -m pip install --upgrade pip

    Write-Host "Installing local UI dependencies..."
    & $VenvPython -m pip install -r $Requirements

    Write-Host ""
    Write-Host "Setup finished. Start the interface with:"
    Write-Host "  .\run_local.ps1"
}
finally {
    Pop-Location
}
