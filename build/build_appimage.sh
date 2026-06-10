#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
DIST="$ROOT/dist_nuitka"

# ── Logging (tee'd to build/logs) + error context ─────────────────────────────
# shellcheck source=build/_log.sh
source "$SCRIPT_DIR/_log.sh"
log_init "$ROOT/build"

_on_err() {
    local rc=$? line=$1
    err "build_appimage.sh failed at line $line (exit $rc)"
    printf "%s\n" \
        "The step above did not complete. Things to check:" \
        "  • read the failure in the full log: ${BUILD_LOG:-<none>}" \
        "  • missing system lib? re-run after installing it (see the manual printed above)" \
        "  • Nuitka/C-compiler errors? ensure gcc + the Qt/X11 dev libs are present" \
        "  • inconsistent env? delete dist_nuitka/ and the venv, then rebuild" | manual "Build failed"
}
trap '_on_err $LINENO' ERR

# ── Build variant ─────────────────────────────────────────────────────────────
# cpu        : CPU-only onnxruntime (universal, runs anywhere).
# gpu-cuda12 : onnxruntime-gpu for CUDA 12 / cuDNN 9 (modern NVIDIA, driver R525+).
# gpu-cuda11 : onnxruntime-gpu for CUDA 11.8 / cuDNN 8 (legacy NVIDIA).
# CUDA runtime libraries are NOT bundled — they are provided by the host. Both GPU
# variants fall back to CPU automatically when host CUDA libs are absent.
# Default is auto: gpu-cuda12 if an NVIDIA GPU is detected, else cpu.
VARIANT=auto
for arg in "$@"; do
    case "$arg" in
        --cpu)        VARIANT=cpu ;;
        --gpu)        VARIANT=gpu-cuda12 ;;   # alias for the modern GPU build
        --gpu-cuda12) VARIANT=gpu-cuda12 ;;
        --gpu-cuda11) VARIANT=gpu-cuda11 ;;
        *) echo "Unknown argument: $arg"; echo "Usage: bash build/build_appimage.sh [--cpu|--gpu|--gpu-cuda12|--gpu-cuda11]"; exit 1 ;;
    esac
done

detect_nvidia_gpu() {
    command -v nvidia-smi &>/dev/null || return 1
    nvidia-smi -L 2>/dev/null | grep -q "GPU "
}

if [ "$VARIANT" = auto ]; then
    if detect_nvidia_gpu; then VARIANT=gpu-cuda12; else VARIANT=cpu; fi
    echo "==> Auto-selected variant: $VARIANT"
fi

case "$VARIANT" in cpu|gpu-cuda12|gpu-cuda11) ;; *) echo "ERROR: bad variant '$VARIANT'"; exit 1 ;; esac
[ "$VARIANT" = cpu ] && IS_GPU=false || IS_GPU=true

APPIMAGE_OUT="$ROOT/ProfAnnotate-${VARIANT}-x86_64.AppImage"
APPDIR="$DIST/ProfAnnotate-${VARIANT}.AppDir"

# ── Python from active env (conda or venv) ────────────────────────────────────
if [ -z "${CONDA_PREFIX:-}" ]; then
    manual "No active build environment" <<'EOF'
build_appimage.sh expects an activated Python env (its Python is $CONDA_PREFIX/bin/python).

  Easiest — use the project bootstrap, which creates and uses a venv:
    bash scripts/setup.sh
    source .venv/bin/activate
    export CONDA_PREFIX="$PWD/.venv"   # so this script finds the venv Python

  Or, for the portable cross-distro build, skip all of this and run:
    bash build/build_in_container.sh --all
EOF
    die "no active environment (CONDA_PREFIX unset)"
fi
PYTHON="$CONDA_PREFIX/bin/python"
[ -x "$PYTHON" ] || die "Python not found at $PYTHON — is the environment activated?"
ok "Python: $PYTHON ($("$PYTHON" --version 2>&1))"

# ── System dependencies ────────────────────────────────────────────────────────
# Try to auto-install via the system package manager; if that fails (no sudo,
# unknown distro), print a manual instead of dying silently.
_install_pkg() {
    local pkg="$1"
    if command -v pacman &>/dev/null;    then sudo pacman -S --noconfirm "$pkg"
    elif command -v apt-get &>/dev/null; then sudo apt-get install -y "$pkg"
    elif command -v dnf &>/dev/null;     then sudo dnf install -y "$pkg"
    elif command -v zypper &>/dev/null;  then sudo zypper install -y "$pkg"
    else return 1; fi
}
_need_tool() {
    local cmd="$1" pkg="$2"
    command -v "$cmd" >/dev/null 2>&1 && { ok "$cmd"; return; }
    warn "$cmd missing — attempting to install '$pkg'"
    if ! _install_pkg "$pkg" >/dev/null 2>&1; then
        pkg_manual "$pkg"
        die "could not install '$pkg' automatically; install it and re-run"
    fi
    command -v "$cmd" >/dev/null 2>&1 || die "'$cmd' still not found after installing '$pkg'"
    ok "$cmd (installed)"
}

step "Checking system dependencies"
_need_tool patchelf patchelf
_need_tool objdump  binutils
_need_tool wget     wget

step "Checking Python dependencies"
need_pymod "$PYTHON" nuitka      "nuitka"
need_pymod "$PYTHON" PIL.Image   "Pillow"
need_pymod "$PYTHON" PySide6     "PySide6>=6.6.0"
need_pymod "$PYTHON" cv2         "opencv-python-headless>=4.9.0"
need_pymod "$PYTHON" onnxruntime "onnxruntime>=1.17.0"
need_pymod "$PYTHON" numpy       "numpy>=1.26.0"
need_pymod "$PYTHON" yaml        "PyYAML>=6.0.1"
need_pymod "$PYTHON" git         "gitpython>=3.1.40"

# ── Ensure the onnxruntime wheel matching the variant is installed ─────────────
# Nuitka bundles whatever `import onnxruntime` resolves to, so we swap the wheel
# itself. We deliberately do NOT install the nvidia-*-cu1x runtime wheels — the
# heavy CUDA libraries (cudnn/cublas/cudart) are provided by the host at runtime.
case "$VARIANT" in
    cpu)        ORT_SPEC="onnxruntime>=1.17.0" ;;
    gpu-cuda12) ORT_SPEC="onnxruntime-gpu>=1.19,<2" ;;   # CUDA 12 / cuDNN 9
    gpu-cuda11) ORT_SPEC="onnxruntime-gpu==1.18.1" ;;    # CUDA 11.8 / cuDNN 8
esac
echo "==> Ensuring onnxruntime wheel for '$VARIANT': $ORT_SPEC"

# Wipe both onnxruntime distributions, then install exactly the one we want, so
# repeated builds with different variants in the same env don't clash.
"$PYTHON" -m pip uninstall -y onnxruntime onnxruntime-gpu >/dev/null 2>&1 || true
"$PYTHON" -m pip install "$ORT_SPEC"
echo "  onnxruntime device tag: $("$PYTHON" -c "import onnxruntime; print(onnxruntime.get_device())" 2>/dev/null || echo unknown)"

# Locate the onnxruntime capi dir so we can guarantee the GPU provider stubs are
# bundled (the package-data glob alone has been unreliable for these).
ORT_CAPI="$("$PYTHON" -c "import os, onnxruntime; print(os.path.join(os.path.dirname(onnxruntime.__file__), 'capi'))" 2>/dev/null || echo "")"

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

echo "==> [1/5] Nuitka compile ($VARIANT)"
cd "$ROOT"
rm -rf "$DIST"
mkdir -p "$DIST"
export PATH="$ROOT/build:$PATH"

NUITKA_ARGS=(
    --standalone
    --onefile
    --output-dir="$DIST"
    --output-filename="profannotate"

    --enable-plugin=pyside6
    --enable-plugin=numpy

    --include-package=profannotate

    --include-package=PIL
    --include-package-data=PIL
    --include-package=PIL.Image
    --include-package=PIL.ImageFile
    --include-package=PIL.ImageOps
    --include-package=PIL.ImageFilter
    --include-package=PIL.JpegImagePlugin
    --include-package=PIL.Jpeg2KImagePlugin
    --include-package=PIL.PngImagePlugin
    --include-package=PIL.BmpImagePlugin
    --include-package=PIL.WebPImagePlugin
    --include-package=PIL.TiffImagePlugin
    --include-package=PIL.GifImagePlugin
    --include-package=PIL.PpmImagePlugin
    --include-package=PIL.IcoImagePlugin
    --include-package=PIL.TgaImagePlugin
    --include-package=PIL.MpoImagePlugin

    --include-package=cv2
    --include-package-data=cv2

    --include-package=onnxruntime
    --include-package-data=onnxruntime

    --include-package=numpy
    --include-package=yaml
    --include-package=git
    --include-package=gitdb
    --include-package=smmap

    --include-data-dir="$ROOT/assets=assets"
    --include-data-dir="$ROOT/models=models"

    --nofollow-import-to=tkinter
    --nofollow-import-to=matplotlib
    --nofollow-import-to=scipy
    --nofollow-import-to=IPython
    --nofollow-import-to=jupyter
    --nofollow-import-to=notebook
    --nofollow-import-to=test
    --nofollow-import-to=unittest

    --assume-yes-for-downloads
    --show-progress
    --jobs="$(nproc)"
)

# Note: the onnxruntime CUDA provider libs (libonnxruntime_providers_cuda.so,
# _shared.so) are bundled automatically by Nuitka's dll-files plugin together with
# --include-package-data=onnxruntime, so no explicit --include-data-files is needed
# (adding one conflicts with the auto-detected extension). The external CUDA runtime
# (cudnn/cublas/cudart) is NOT bundled — it is dlopen'd from the host at runtime.

"$PYTHON" -m nuitka "${NUITKA_ARGS[@]}" main.py

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

# Host-provided CUDA: prepend common CUDA toolkit dirs so onnxruntime-gpu can
# dlopen the host's libcudart/libcublas/libcudnn. System library dirs are already
# covered by ldconfig; the NVIDIA driver (libcuda.so.1) is always present on
# NVIDIA machines. Absent CUDA -> onnxruntime falls back to CPU automatically.
_CUDA_DIRS=""
for d in /usr/local/cuda/lib64 /usr/local/cuda-12*/lib64 /usr/local/cuda-11*/lib64 \
         /usr/local/cuda/lib /opt/cuda/lib64; do
    [ -d "$d" ] && _CUDA_DIRS="${_CUDA_DIRS:+$_CUDA_DIRS:}$d"
done
export LD_LIBRARY_PATH="${_CUDA_DIRS:+$_CUDA_DIRS:}${LD_LIBRARY_PATH:-}"

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
echo "  Variant: $VARIANT"
if [ "$IS_GPU" = true ]; then
    echo "  Compute: GPU ($ORT_SPEC) — CUDA libs provided by host, CPU fallback if absent"
else
    echo "  Compute: CPU ($ORT_SPEC)"
fi
echo "=============================================="
echo ""
echo "  Double-click to run, or from terminal:"
echo "    $APPIMAGE_OUT"
echo ""
echo "  To install as system command:"
echo "    sudo cp $APPIMAGE_OUT /usr/local/bin/profannotate"
echo "    sudo chmod +x /usr/local/bin/profannotate"
