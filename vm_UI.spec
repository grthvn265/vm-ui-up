# -*- mode: python ; coding: utf-8 -*-
# PyInstaller recipe for a single-file Vessel Metrics executable.
#
#   Windows:  build_windows.bat      ->  dist\VesselMetrics.exe
#   macOS:    ./build_mac.sh         ->  dist/VesselMetrics
#
# PyInstaller cannot cross-compile: build the macOS executable on a Mac (or
# with the GitHub Actions workflow in .github/workflows/build.yml).
#
# The program prints its progress to the terminal, so it is built as a console
# application: a terminal window opens next to the graphical interface.
from PyInstaller.utils.hooks import collect_submodules

hiddenimports = (
    # .czi support is imported only when a .czi file is opened
    ["czifile", "tifffile"]
    + collect_submodules("imagecodecs")
)

a = Analysis(
    ["vm_UI.py"],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={
        # figures are only saved to disk, the interactive backends are not needed
        "matplotlib": {"backends": ["Agg"]},
    },
    runtime_hooks=[],
    excludes=[
        "PyQt5", "PyQt6", "PySide2", "PySide6", "IPython", "jupyter",
        "notebook", "pytest", "sphinx", "pandas", "dask", "numba",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="VesselMetrics",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
