#!/bin/sh
# Build dist/VesselMetrics (single-file macOS executable). Run on a Mac:
#     chmod +x build_mac.sh && ./build_mac.sh
# Needs a python 3 with tkinter (python.org installer, or brew install python-tk).
# The executable targets the architecture of the Mac it is built on
# (Apple silicon or Intel).
set -e
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
if [ ! -x .venv/bin/python ]; then
    echo "Creating virtual environment .venv ..."
    "$PYTHON" -m venv .venv
fi
.venv/bin/python -c "import tkinter" || {
    echo "This python has no tkinter. Install python from python.org or: brew install python-tk"
    exit 1
}
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt pyinstaller
.venv/bin/python -m PyInstaller --noconfirm --clean vm_UI.spec

echo
echo "Done: dist/VesselMetrics"
echo "Double-click it in Finder (it opens in Terminal) or run ./dist/VesselMetrics"
echo "If macOS blocks it: right-click > Open, or run"
echo "    xattr -dr com.apple.quarantine dist/VesselMetrics"
