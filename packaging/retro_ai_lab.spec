# PyInstaller spec for the Retro AI Lab desktop build.
#
# Build (from the repository root):
#     pyinstaller packaging/retro_ai_lab.spec --noconfirm
# The result is a one-folder app in dist/RetroAILab/ (RetroAILab.exe on Windows).
# `--exit-after-ms` still works in the built app, which is how CI smoke-tests it.

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

exe = EXE(
    pyz,
    a.scripts,
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
    strip=False,
    upx=False,
    name="RetroAILab",
)
