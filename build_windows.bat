@echo off
rem Build dist\VesselMetrics.exe (single file). Run from this folder.
setlocal
cd /d "%~dp0"

if not exist .venv\Scripts\python.exe (
    echo Creating virtual environment .venv ...
    python -m venv .venv || goto :error
)
.venv\Scripts\python -m pip install --upgrade pip || goto :error
.venv\Scripts\python -m pip install -r requirements.txt pyinstaller || goto :error
.venv\Scripts\python -m PyInstaller --noconfirm --clean vm_UI.spec || goto :error

echo.
echo Done: dist\VesselMetrics.exe
exit /b 0

:error
echo Build failed.
exit /b 1
