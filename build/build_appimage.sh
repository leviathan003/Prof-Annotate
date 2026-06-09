#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
DIST="$ROOT/dist_nuitka"
APPIMAGE_OUT="$ROOT/ProfAnnotate-x86_64.AppImage"
APPDIR="$DIST/ProfAnnotate.AppDir"

# ── Python from active conda env ──────────────────────────────────────────────
if [ -z "${CONDA_PREFIX:-}" ]; then
    echo "ERROR: conda env not activated. Run: conda activate <your_env>"
    exit 1
fi
PYTHON="$CONDA_PREFIX/bin/python"
echo "Using Python: $PYTHON ($("$PYTHON" --version 2>&1))"

# ── Dependencies ──────────────────────────────────────────────────────────────
_install_pkg() {
    local pkg="$1"
    if command -v pacman &>/dev/null;  then sudo pacman -S --noconfirm "$pkg"
    elif command -v apt-get &>/dev/null; then sudo apt-get install -y "$pkg"
    elif command -v dnf &>/dev/null;   then sudo dnf install -y "$pkg"
    else echo "ERROR: install $pkg manually"; exit 1; fi
}

echo "==> Checking system dependencies"
command -v patchelf >/dev/null 2>&1 || _install_pkg patchelf
command -v objdump  >/dev/null 2>&1 || _install_pkg binutils
command -v wget     >/dev/null 2>&1 || _install_pkg wget

echo "==> Checking Python dependencies"
"$PYTHON" -c "import nuitka"      2>/dev/null || { echo "ERROR: run: pip install nuitka ordered-set zstandard"; exit 1; }
"$PYTHON" -c "import PIL.Image"   2>/dev/null || { echo "ERROR: run: pip install Pillow"; exit 1; }
"$PYTHON" -c "import PySide6"     2>/dev/null || { echo "ERROR: run: pip install PySide6"; exit 1; }
"$PYTHON" -c "import cv2"         2>/dev/null || { echo "ERROR: run: pip install opencv-python-headless"; exit 1; }
"$PYTHON" -c "import onnxruntime" 2>/dev/null || { echo "ERROR: run: pip install onnxruntime"; exit 1; }
"$PYTHON" -c "import numpy"       2>/dev/null || { echo "ERROR: run: pip install numpy"; exit 1; }
"$PYTHON" -c "import yaml"        2>/dev/null || { echo "ERROR: run: pip install pyyaml"; exit 1; }
"$PYTHON" -c "import git"         2>/dev/null || { echo "ERROR: run: pip install gitpython"; exit 1; }

# patchelf 0.18.0 is buggy and rejected by Nuitka — use 0.17.2
PATCHELF_BIN=""
PATCHELF_VERSION=$(patchelf --version 2>/dev/null | awk '{print $2}' || echo "none")
if [ "$PATCHELF_VERSION" = "0.18.0" ] || [ "$PATCHELF_VERSION" = "none" ]; then
    echo "patchelf $PATCHELF_VERSION is unusable — downloading 0.17.2"
    PATCHELF_LOCAL="$ROOT/build/patchelf-0.17.2"
    if [ ! -f "$PATCHELF_LOCAL" ]; then
        wget -q \
            "https://github.com/NixOS/patchelf/releases/download/0.17.2/patchelf-0.17.2-x86_64.tar.gz" \
            -O /tmp/patchelf.tar.gz
            TMP_EXTRACT="$(mktemp -d)"
            tar --no-same-owner --no-same-permissions -xzf /tmp/patchelf.tar.gz -C "$TMP_EXTRACT"

            PATCHELF_SRC="$(find "$TMP_EXTRACT" -type f -name patchelf | head -n 1)"
            if [ -z "$PATCHELF_SRC" ]; then
                echo "ERROR: patchelf binary not found after extract"
                exit 1
            fi

            cp "$PATCHELF_SRC" "$PATCHELF_LOCAL"
            chmod +x "$PATCHELF_LOCAL"
            rm -rf "$TMP_EXTRACT"
        chmod +x "$PATCHELF_LOCAL"
    fi
    export PATH="$(dirname "$PATCHELF_LOCAL"):$PATH"
    # Symlink with expected name so Nuitka finds it
    ln -sf "$PATCHELF_LOCAL" "$(dirname "$PATCHELF_LOCAL")/patchelf"
    PATCHELF_BIN="$PATCHELF_LOCAL"
    echo "  Using patchelf: $("$PATCHELF_BIN" --version)"
else
    echo "  patchelf $PATCHELF_VERSION OK"
fi

echo "==> [1/5] Nuitka compile"
cd "$ROOT"
rm -rf "$DIST"
mkdir -p "$DIST"
export PATH="$ROOT/build:$PATH"

"$PYTHON" -m nuitka \
    --standalone \
    --onefile \
    --output-dir="$DIST" \
    --output-filename="profannotate" \
    \
    --enable-plugin=pyside6 \
    --enable-plugin=numpy \
    \
    --include-package=src \
    \
    --include-package=PIL \
    --include-package-data=PIL \
    --include-package=PIL.Image \
    --include-package=PIL.ImageFile \
    --include-package=PIL.ImageOps \
    --include-package=PIL.ImageFilter \
    --include-package=PIL.JpegImagePlugin \
    --include-package=PIL.Jpeg2KImagePlugin \
    --include-package=PIL.PngImagePlugin \
    --include-package=PIL.BmpImagePlugin \
    --include-package=PIL.WebPImagePlugin \
    --include-package=PIL.TiffImagePlugin \
    --include-package=PIL.GifImagePlugin \
    --include-package=PIL.PpmImagePlugin \
    --include-package=PIL.IcoImagePlugin \
    --include-package=PIL.TgaImagePlugin \
    --include-package=PIL.MpoImagePlugin \
    \
    --include-package=cv2 \
    --include-package-data=cv2 \
    \
    --include-package=onnxruntime \
    --include-package-data=onnxruntime \
    \
    --include-package=numpy \
    --include-package=yaml \
    --include-package=git \
    --include-package=gitdb \
    --include-package=smmap \
    \
    --include-data-dir="$ROOT/assets=assets" \
    --include-data-dir="$ROOT/models=models" \
    \
    --nofollow-import-to=tkinter \
    --nofollow-import-to=matplotlib \
    --nofollow-import-to=scipy \
    --nofollow-import-to=IPython \
    --nofollow-import-to=jupyter \
    --nofollow-import-to=notebook \
    --nofollow-import-to=test \
    --nofollow-import-to=unittest \
    \
    --assume-yes-for-downloads \
    --show-progress \
    --jobs="$(nproc)" \
    main.py

ONEFILE_BIN="$DIST/profannotate"
[ -f "$ONEFILE_BIN" ] || { echo "ERROR: Nuitka output not found at $ONEFILE_BIN"; exit 1; }
echo "  Binary size: $(du -sh "$ONEFILE_BIN" | cut -f1)"

echo "==> [2/5] Patching hashed sonames"
NUITKA_DIST="$DIST/profannotate.dist"

_find_canonical_soname() {
    local search="$1"
    local fallback="$2"
    local path=""
    path=$(ldconfig -p 2>/dev/null | grep -E "^\s+${search}" | awk '{print $NF}' | head -1)
    if [ -z "$path" ]; then
        for d in /usr/lib /usr/lib64 /usr/lib/x86_64-linux-gnu /lib /lib64 /lib/x86_64-linux-gnu; do
            [ -d "$d" ] || continue
            local f
            f=$(find "$d" -maxdepth 2 -name "${search}" -type f 2>/dev/null | head -1)
            [ -n "$f" ] && path="$f" && break
        done
    fi
    [ -z "$path" ] && echo "$fallback" && return
    local real soname
    real=$(readlink -f "$path")
    soname=$(objdump -p "$real" 2>/dev/null | awk '/SONAME/{print $2}' | head -1)
    echo "${soname:-$fallback}"
}

XCB_SONAME=$(_find_canonical_soname "libxcb.so.1" "libxcb.so.1")
DRM_SONAME=$(_find_canonical_soname "libdrm.so.2" "libdrm.so.2")
echo "  libxcb canonical: $XCB_SONAME"
echo "  libdrm canonical: $DRM_SONAME"

if [ -d "$NUITKA_DIST" ]; then
    find "$NUITKA_DIST" -name "*.so*" -type f | while read -r lib; do
        needed=$(objdump -p "$lib" 2>/dev/null | awk '/NEEDED/{print $2}') || continue
        for n in $needed; do
            case "$n" in
                libxcb-*.so.*)
                    patchelf --replace-needed "$n" "$XCB_SONAME" "$lib" 2>/dev/null || true
                    echo "  patched $(basename "$lib"): $n -> $XCB_SONAME"
                    ;;
                libdrm-*.so.*)
                    patchelf --replace-needed "$n" "$DRM_SONAME" "$lib" 2>/dev/null || true
                    echo "  patched $(basename "$lib"): $n -> $DRM_SONAME"
                    ;;
            esac
        done
    done
fi

echo "==> [3/5] Assembling AppDir"
rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/bin"
mkdir -p "$APPDIR/usr/share/icons/hicolor/256x256/apps"
mkdir -p "$APPDIR/usr/share/applications"

cp "$ONEFILE_BIN" "$APPDIR/usr/bin/profannotate"
chmod +x "$APPDIR/usr/bin/profannotate"
ln -sf "usr/bin/profannotate" "$APPDIR/profannotate"

cat > "$APPDIR/AppRun" << 'APPRUN'
#!/bin/bash
HERE="$(dirname "$(readlink -f "$0")")"
BIN="$HERE/usr/bin"

export QT_AUTO_SCREEN_SCALE_FACTOR=1
export FONTCONFIG_PATH="/etc/fonts"
unset LD_PRELOAD

if [ -z "${QT_QPA_PLATFORM:-}" ]; then
    if [ -n "${WAYLAND_DISPLAY:-}" ]; then
        export QT_QPA_PLATFORM=wayland
    else
        export QT_QPA_PLATFORM=xcb
    fi
fi

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/usr/local/cuda/lib64"

LOG_DIR="${HOME}/.profannotate"
mkdir -p "$LOG_DIR"

if [ ! -t 1 ]; then
    exec "$BIN/profannotate" "$@" >> "$LOG_DIR/launch.log" 2>&1
else
    exec "$BIN/profannotate" "$@"
fi
APPRUN
chmod +x "$APPDIR/AppRun"

cat > "$APPDIR/ProfAnnotate.desktop" << 'DESKTOP'
[Desktop Entry]
Name=Prof Annotate
Exec=profannotate
Icon=profannotate
Type=Application
Categories=Graphics;Science;
Comment=Terminal-style annotation tool for YOLO training data
Terminal=false
StartupNotify=true
DESKTOP

cp "$APPDIR/ProfAnnotate.desktop" "$APPDIR/usr/share/applications/ProfAnnotate.desktop"

echo "==> Generating icon"
"$PYTHON" - << PYEOF
from PIL import Image, ImageDraw
img = Image.new('RGBA', (256, 256), (74, 30, 85, 255))
d = ImageDraw.Draw(img)
d.ellipse([40, 40, 216, 216], fill=(30, 15, 50, 255))
d.ellipse([80, 80, 176, 176], fill=(212, 175, 55, 255))
d.polygon([(128,70),(148,120),(100,90),(156,90),(108,120)], fill=(255,255,255,200))
img.save("$APPDIR/profannotate.png")
img.save("$APPDIR/usr/share/icons/hicolor/256x256/apps/profannotate.png")
print("Icon saved")
PYEOF

echo "==> [4/5] Fetching appimagetool"
APPIMAGETOOL="$DIST/appimagetool-x86_64.AppImage"
if [ ! -f "$APPIMAGETOOL" ]; then
    wget -q \
        "https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage" \
        -O "$APPIMAGETOOL"
    chmod +x "$APPIMAGETOOL"
fi

if [ "${CI:-false}" = "true" ]; then
    TOOL_TMP="$DIST/_appimagetool_extracted"
    rm -rf "$TOOL_TMP"
    cd "$DIST"
    "$APPIMAGETOOL" --appimage-extract
    mv "$DIST/squashfs-root" "$TOOL_TMP"
    APPIMAGETOOL_BIN="$TOOL_TMP/AppRun"
    cd "$ROOT"
else
    APPIMAGETOOL_BIN="$APPIMAGETOOL"
fi

echo "==> [5/5] Building AppImage -> $APPIMAGE_OUT"
ARCH=x86_64 "$APPIMAGETOOL_BIN" "$APPDIR" "$APPIMAGE_OUT"
chmod +x "$APPIMAGE_OUT"

echo ""
echo "=============================================="
echo "  Done: $APPIMAGE_OUT"
echo "  Size: $(du -sh "$APPIMAGE_OUT" | cut -f1)"
echo "=============================================="
echo ""
echo "  Double-click to run, or from terminal:"
echo "    $APPIMAGE_OUT"
echo ""
echo "  To install as system command:"
echo "    sudo cp $APPIMAGE_OUT /usr/local/bin/profannotate"
echo "    sudo chmod +x /usr/local/bin/profannotate"
