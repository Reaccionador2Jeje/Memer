$ErrorActionPreference = "Stop"

$Repository = "https://github.com/Reaccionador2Jeje/Memer/archive/refs/heads/main.zip"

Write-Host "Memer CLI installer" -ForegroundColor Cyan

$PythonLauncher = Get-Command py -ErrorAction SilentlyContinue
if ($PythonLauncher) {
    $VersionCheck = & py -3 -c "import sys; print(sys.version_info >= (3, 10))"
    if ($LASTEXITCODE -ne 0 -or $VersionCheck -ne "True") {
        throw "Python 3.10 or newer is required."
    }
    & py -3 -m pip install --upgrade $Repository
    if ($LASTEXITCODE -ne 0) {
        throw "Memer installation failed."
    }
} else {
    $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if (-not $PythonCommand) {
        throw "Python 3.10+ is required. Install Python and run this installer again."
    }
    $VersionCheck = & python -c "import sys; print(sys.version_info >= (3, 10))"
    if ($LASTEXITCODE -ne 0 -or $VersionCheck -ne "True") {
        throw "Python 3.10 or newer is required."
    }
    & python -m pip install --upgrade $Repository
    if ($LASTEXITCODE -ne 0) {
        throw "Memer installation failed."
    }
}

Write-Host ""
Write-Host "Installation finished. Try: memer --help" -ForegroundColor Green
Write-Host "If the command is not found, activate your virtual environment or add Python's Scripts directory to PATH."
