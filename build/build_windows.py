#!/usr/bin/env python3
"""
build/build_windows.py — Windows .exe build for ProfAnnotate.

Cross-platform counterpart to build/build_appimage.sh (which is Linux-only:
bash, patchelf, soname patching, AppDir/appimagetool). This script runs Nuitka
on Windows to produce a single portable .exe. The Nuitka arg list mirrors the
one in build_appimage.sh:176-232 minus the Linux-only post-processing.

Usage:
    python build/build_windows.py [cpu|gpu-cuda12]

  cpu        : CPU-only onnxruntime (universal, runs on any Windows x64 machine).
  gpu-cuda12 : onnxruntime-gpu for CUDA 12 / cuDNN 9. Falls back to CPU at runtime
               when the host CUDA libraries are absent.

Requires an activated Python env with the package installed plus the build
extras (nuitka, ordered-set, zstandard). Nuitka downloads its own C toolchain
(MinGW64) on first run thanks to --assume-yes-for-downloads.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist_nuitka_win"

# onnxruntime wheel per variant — mirrors build_appimage.sh:120-124. The heavy CUDA
# runtime (cudnn/cublas/cudart) is NOT bundled; onnxruntime-gpu falls back to CPU
# automatically when the host CUDA libraries are missing.
ORT_SPECS = {
    "cpu": "onnxruntime>=1.17.0",
    "gpu-cuda12": "onnxruntime-gpu>=1.19,<2",
}


def log(msg: str) -> None:
    print(f"==> {msg}", flush=True)


def ensure_onnxruntime(variant: str) -> None:
    """Install exactly the onnxruntime wheel for this variant, wiping both first so
    repeated builds with different variants in the same env don't clash."""
    spec = ORT_SPECS[variant]
    log(f"Ensuring onnxruntime wheel for '{variant}': {spec}")
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "onnxruntime", "onnxruntime-gpu"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run([sys.executable, "-m", "pip", "install", spec], check=True)
    device = subprocess.run(
        [sys.executable, "-c", "import onnxruntime; print(onnxruntime.get_device())"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    print(f"  onnxruntime device tag: {device or 'unknown'}")


def generate_icon(ico_path: Path) -> None:
    """Render the app icon to a multi-size .ico — mirrors build_appimage.sh:356-366."""
    log("Generating icon")
    from PIL import Image, ImageDraw

    img = Image.new("RGBA", (256, 256), (74, 30, 85, 255))
    d = ImageDraw.Draw(img)
    d.ellipse([40, 40, 216, 216], fill=(30, 15, 50, 255))
    d.ellipse([80, 80, 176, 176], fill=(212, 175, 55, 255))
    d.polygon([(128, 70), (148, 120), (100, 90), (156, 90), (108, 120)], fill=(255, 255, 255, 200))
    ico_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(ico_path, sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    print(f"  Icon saved: {ico_path}")


def build(variant: str) -> Path:
    out_name = f"ProfAnnotate-{variant}-windows-x64"
    ico_path = DIST / "profannotate.ico"

    if DIST.exists():
        shutil.rmtree(DIST)
    DIST.mkdir(parents=True)

    generate_icon(ico_path)

    # Portable subset of the Linux Nuitka args (build_appimage.sh:176-232). No
    # patchelf / soname patching / AppDir — those are Linux-only.
    nuitka_args = [
        "--standalone",
        "--onefile",
        f"--output-dir={DIST}",
        f"--output-filename={out_name}",
        "--enable-plugin=pyside6",
        "--enable-plugin=numpy",
        "--include-package=profannotate",
        "--include-package=PIL",
        "--include-package-data=PIL",
        "--include-package=PIL.Image",
        "--include-package=PIL.ImageFile",
        "--include-package=PIL.ImageOps",
        "--include-package=PIL.ImageFilter",
        "--include-package=PIL.JpegImagePlugin",
        "--include-package=PIL.Jpeg2KImagePlugin",
        "--include-package=PIL.PngImagePlugin",
        "--include-package=PIL.BmpImagePlugin",
        "--include-package=PIL.WebPImagePlugin",
        "--include-package=PIL.TiffImagePlugin",
        "--include-package=PIL.GifImagePlugin",
        "--include-package=PIL.PpmImagePlugin",
        "--include-package=PIL.IcoImagePlugin",
        "--include-package=PIL.TgaImagePlugin",
        "--include-package=PIL.MpoImagePlugin",
        "--include-package=cv2",
        "--include-package-data=cv2",
        "--include-package=onnxruntime",
        "--include-package-data=onnxruntime",
        "--include-package=numpy",
        "--include-package=yaml",
        "--include-package=git",
        "--include-package=gitdb",
        "--include-package=smmap",
        f"--include-data-dir={ROOT / 'assets'}=assets",
        f"--include-data-dir={ROOT / 'models'}=models",
        "--nofollow-import-to=tkinter",
        "--nofollow-import-to=matplotlib",
        "--nofollow-import-to=scipy",
        "--nofollow-import-to=IPython",
        "--nofollow-import-to=jupyter",
        "--nofollow-import-to=notebook",
        "--nofollow-import-to=test",
        "--nofollow-import-to=unittest",
        # Windows-specific: hide the console window for a GUI app, set the exe icon.
        "--windows-console-mode=disable",
        f"--windows-icon-from-ico={ico_path}",
        "--assume-yes-for-downloads",
        "--show-progress",
        f"--jobs={os.cpu_count() or 1}",
    ]

    log(f"Nuitka compile ({variant})")
    subprocess.run(
        [sys.executable, "-m", "nuitka", *nuitka_args, str(ROOT / "main.py")],
        check=True,
        cwd=ROOT,
    )

    built = DIST / f"{out_name}.exe"
    if not built.is_file():
        sys.exit(f"ERROR: Nuitka output not found at {built}")

    # Copy to repo root, matching where the AppImages land (and where the CI
    # smoke test / artifact upload expect it).
    final = ROOT / f"{out_name}.exe"
    shutil.copy2(built, final)
    size_mb = final.stat().st_size / (1024 * 1024)
    print()
    print("==============================================")
    print(f"  Done: {final}")
    print(f"  Size: {size_mb:.0f} MB")
    print(f"  Variant: {variant}")
    print("==============================================")
    return final


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the ProfAnnotate Windows .exe")
    parser.add_argument(
        "variant",
        nargs="?",
        default="cpu",
        choices=sorted(ORT_SPECS),
        help="compute variant (default: cpu)",
    )
    args = parser.parse_args()

    ensure_onnxruntime(args.variant)
    build(args.variant)
    return 0


if __name__ == "__main__":
    sys.exit(main())
