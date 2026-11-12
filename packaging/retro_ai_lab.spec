# PyInstaller spec for the Retro AI Lab desktop build.
#
# Build (from the repository root):
#     pyinstaller packaging/retro_ai_lab.spec --noconfirm
# The result is a one-folder app in dist/RetroAILab/ (RetroAILab.exe on Windows).
# `--exit-after-ms` still works in the built app, which is how CI smoke-tests it.

import importlib.util
import os
import sys

from PyInstaller.utils.hooks import collect_submodules

block_cipher = None

hidden = collect_submodules("retro_ai_lab")

a = Analysis(
    ["../src/retro_ai_lab/__main__.py"],
    pathex=["../src"],
    binaries=[],
    datas=[
        # The BizHawk bridge script ships next to the adapter package.
        ("../src/retro_ai_lab/emulator/bizhawk/retro_ai_lab.lua", "retro_ai_lab/emulator/bizhawk"),
        # EmuHawk has no socket library: the loader and the luasocket folder go with the script.
        ("../src/retro_ai_lab/emulator/bizhawk/socket.lua", "retro_ai_lab/emulator/bizhawk"),
        ("../src/retro_ai_lab/emulator/bizhawk/luasocket", "retro_ai_lab/emulator/bizhawk/luasocket"),
        ("../src/retro_ai_lab/emulator/mgba/retro_ai_lab.lua", "retro_ai_lab/emulator/mgba"),
    ],
    hiddenimports=hidden,
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        # Qt modules the app does not use; keeps the build small.
        "PySide6.QtWebEngineCore",
        "PySide6.QtWebEngineWidgets",
        "PySide6.QtQml",
        "PySide6.QtQuick",
        "PySide6.Qt3DCore",
        "PySide6.QtMultimedia",
        "PySide6.QtCharts",
        "PySide6.QtDataVisualization",
        "PySide6.QtPdf",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# The "starting" window (T-1002). The bootloader draws it before Python is loaded, so a
# first start that spends a minute reading new files has something on the screen; the app
# writes the step it is on under the picture and closes it (retro_ai_lab/core/starting.py).
# It is drawn with the build Python's Tcl/Tk, which is not supported on macOS and is
# missing from some Linux Pythons: there the app is built without it and starts as before.
# The Windows build must have it, and the smoke test in CI checks that it does.
TEXT_POS = (28, 168)  # the lower left corner of the step's text, inside splash.png
splash = None
if sys.platform != "darwin" and importlib.util.find_spec("tkinter") is not None:
    splash = Splash(
        os.path.join(SPECPATH, "splash.png"),
        binaries=a.binaries,
        datas=a.datas,
        text_pos=TEXT_POS,
        text_size=11,
        text_color="#eaeefa",
        text_default="Starting",
        always_on_top=False,  # a minute is long: other windows may be put in front of it
    )
elif sys.platform == "win32":
    raise SystemExit("The Windows build needs tkinter for the starting window.")

exe = EXE(
    pyz,
    a.scripts,
    *([splash] if splash is not None else []),
    [],
    exclude_binaries=True,
    name="RetroAILab",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # a windowed app: no console window on Windows
    disable_windowed_traceback=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    *([splash.binaries] if splash is not None else []),
    strip=False,
    upx=False,
    name="RetroAILab",
)
