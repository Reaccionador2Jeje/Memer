#!/usr/bin/env sh
set -eu

REPOSITORY="git+https://github.com/Reaccionador2Jeje/Memer.git"

echo "Memer CLI installer"
echo "Checking Python..."
if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3.10+ is required. Install Python and run this installer again."
  exit 1
fi

PYTHON_VERSION="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
echo "Found Python $PYTHON_VERSION"

if command -v pipx >/dev/null 2>&1; then
  echo "Installing isolated CLI with pipx..."
  pipx install --force "$REPOSITORY"
else
  echo "pipx was not found; installing into the current Python environment."
  python3 -m pip install --upgrade "$REPOSITORY"
fi

echo
echo "Installation finished. Try: memer --help"
echo "If the command is not found, activate your virtual environment or add Python's scripts directory to PATH."
