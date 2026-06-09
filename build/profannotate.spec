# build/profannotate.spec
import re
import sys
from pathlib import Path as _Path
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

block_cipher = None
ROOT = _Path(SPECPATH).parent

_SYSTEM_LIB_PATTERNS = [
    r"libglib", r"libgthread", r"libgcc_s", r"libstdc\+\+",
    r"libc\.so", r"libm\.so", r"libdl\.so", r"libpthread",
    r"librt\.so", r"libresolv", r"libnss_", r"libutil\.so",
    r"libz\.so", r"libGL", r"libGLdispatch", r"libGLX", r"libEGL",
    r"libdrm", r"libxcb", r"libX11", r"libXcb", r"libXext",
    r"libXrender", r"libXfixes", r"libfontconfig", r"libfreetype",
    r"libexpat", r"libuuid", r"libdbus", r"libgobject", r"libgio",
    r"libffi", r"libpcre", r"libwayland", r"libwl", r"libva",
    r"libvulkan", r"libxkbcommon", r"libxshmfence", r"libxdamage",
    r"libxfixes", r"libgomp", r"libopencv",
]

def _is_system_lib(name: str) -> bool:
    fname = _Path(name).name
    return any(re.search(p, fname) for p in _SYSTEM_LIB_PATTERNS)

ort_libs = collect_dynamic_libs("onnxruntime")
pil_libs = collect_dynamic_libs("PIL")

a = Analysis(
    [str(ROOT / "main.py")],
    pathex=[str(ROOT)],
    binaries=[*ort_libs, *pil_libs],
    datas=[
        (str(ROOT / "assets"), "assets"),
        (str(ROOT / "models"), "models"),
        *collect_data_files("onnxruntime"),
        *collect_data_files("cv2"),
        *collect_data_files("PIL"),
    ],
    hiddenimports=[
        "onnxruntime",
        "onnxruntime.capi._pybind_state",
        "cv2",
        "PIL",
        "PIL.Image",
        "PIL.ImageFile",
        "PIL.ImageOps",
        "PIL.ImageFilter",
        "PIL._imaging",
        "PIL.JpegImagePlugin",
        "PIL.Jpeg2KImagePlugin",
        "PIL.PngImagePlugin",
        "PIL.BmpImagePlugin",
        "PIL.WebPImagePlugin",
        "PIL.TiffImagePlugin",
        "PIL.GifImagePlugin",
        "PIL.PpmImagePlugin",
        "PIL.IcoImagePlugin",
        *collect_submodules("PIL"),
        "yaml",
        "git",
        "numpy",
        "PySide6.QtCore",
        "PySide6.QtGui",
        "PySide6.QtWidgets",
        "PySide6.QtOpenGL",
        "PySide6.QtOpenGLWidgets",
        "src.config.constants",
        "src.config.shortcuts",
        "src.config.skeleton",
        "src.core.annotation.models",
        "src.core.annotation.parser",
        "src.core.annotation.writer",
        "src.core.annotation.undo",
        "src.core.dataset.loader",
        "src.core.dataset.validator",
        "src.core.dataset.merger",
        "src.core.dataset.splitter",
        "src.core.dataset.yaml_handler",
        "src.core.inference.engine",
        "src.core.inference.postprocess",
        "src.core.inference.filter",
        "src.core.git.reader",
        "src.core.recovery.autosave",
        "src.ui.main_window",
        "src.ui.tutorial",
        "src.ui.prof_annotate",
        "src.ui.prof_watcher",
        "src.utils.color",
        "src.utils.filesystem",
        "src.utils.image",
        "src.utils.logger",
        "src.utils.prefs",
        "src.utils.ui_scaling",
    ],
    hookspath=[str(ROOT / "build" / "hooks")],
    excludes=["tkinter", "matplotlib", "scipy", "IPython", "jupyter"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

a.binaries = [(name, src, kind)
              for name, src, kind in a.binaries
              if not _is_system_lib(name)]

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="profannotate",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    windowed=True,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="profannotate",
)
