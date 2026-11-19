$ErrorActionPreference = "Stop"

$Repository = "https://github.com/Reaccionador2Jeje/Memer/archive/refs/heads/main.zip"

Write-Host "Memer CLI installer" -ForegroundColor Cyan

$Python = Get-Command py -ErrorAction SilentlyContinue
if ($Python) {
    & py -3 -c "import sys; assert sys.version_info >= (3, 10), 'Python 3.10+ is required'"
    & py -3 -m pip install --upgrade $Repository
} else {
    $Python = Get-Command python -ErrorAction SilentlyContinue
    if (-not $Python) {
        throw "Python 3.10+ is required. Install Python and run this installer again."
    }
    & python -c "import sys; assert sys.version_info >= (3, 10), 'Python 3.10+ is required'"
    & python -m pip install --upgrade $Repository
}

Write-Host ""
Write-Host "Installation finished. Try: memer --help" -ForegroundColor Green
Write-Host "If the command is not found, activate your virtual environment or add Python's Scripts directory to PATH."
