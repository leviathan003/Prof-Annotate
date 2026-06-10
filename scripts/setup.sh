#!/usr/bin/env bash
# scripts/setup.sh
# One-shot setup for Prof Annotate.
# Usage:
#   bash scripts/setup.sh                # auto: gpu-cuda12 if an NVIDIA GPU is found, else cpu
#   bash scripts/setup.sh --cpu          # force CPU-only onnxruntime (universal)
#   bash scripts/setup.sh --gpu          # modern NVIDIA: onnxruntime-gpu, CUDA 12 / cuDNN 9
#   bash scripts/setup.sh --gpu-cuda12   # same as --gpu
#   bash scripts/setup.sh --gpu-cuda11   # legacy NVIDIA: onnxruntime-gpu 1.18, CUDA 11.8 / cuDNN 8
#   bash scripts/setup.sh --dev          # + dev tools (pytest, black, ruff)
#   bash scripts/setup.sh --gpu --dev
#
# CUDA runtime libraries are provided by the host (not installed here). The app
# falls back to CPU automatically if the matching CUDA/cuDNN libs are absent.

set -euo pipefail

# ── Resolve project root (works regardless of where the script is called from)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$ROOT/.venv"

VARIANT=auto   # auto | cpu | gpu-cuda12 | gpu-cuda11
DEV=false

for arg in "$@"; do
    case "$arg" in
        --cpu)        VARIANT=cpu ;;
        --gpu)        VARIANT=gpu-cuda12 ;;
        --gpu-cuda12) VARIANT=gpu-cuda12 ;;
        --gpu-cuda11) VARIANT=gpu-cuda11 ;;
        --dev) DEV=true ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: bash scripts/setup.sh [--cpu|--gpu|--gpu-cuda12|--gpu-cuda11] [--dev]"
            exit 1
            ;;
    esac
done

# ── Colour helpers
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()    { echo -e "${GREEN}[setup]${NC} $*"; }
warn()    { echo -e "${YELLOW}[setup]${NC} $*"; }
error()   { echo -e "${RED}[setup] ERROR:${NC} $*" >&2; }

# ── GPU detection
# Returns 0 if a usable NVIDIA GPU is present (nvidia-smi lists at least one).
detect_nvidia_gpu() {
    command -v nvidia-smi &>/dev/null || return 1
    nvidia-smi -L 2>/dev/null | grep -q "GPU "
}

# ── Resolve variant (auto -> cpu/gpu-cuda12 based on hardware)
if [ "$VARIANT" = auto ]; then
    info "Detecting GPU..."
    if detect_nvidia_gpu; then
        VARIANT=gpu-cuda12
        info "NVIDIA GPU detected — using onnxruntime-gpu (CUDA 12)."
    else
        VARIANT=cpu
        info "No NVIDIA GPU detected — using CPU onnxruntime."
    fi
fi

# Map the variant to a GPU flag + the onnxruntime wheel to install afterwards.
case "$VARIANT" in
    cpu)        GPU=false; ORT_GPU_SPEC="" ;;
    gpu-cuda12) GPU=true;  ORT_GPU_SPEC="onnxruntime-gpu>=1.19,<2" ;;
    gpu-cuda11) GPU=true;  ORT_GPU_SPEC="onnxruntime-gpu==1.18.1" ;;
esac
info "Variant: $VARIANT"

# ── Python version check
info "Checking Python version..."
PYTHON=""
for candidate in python3.12 python3.11 python3.10 python3 python; do
    if command -v "$candidate" &>/dev/null; then
        ver=$("$candidate" -c "import sys; print(sys.version_info[:2])" 2>/dev/null || echo "")
        if [[ "$ver" == "(3, 1"[0-9]")"* ]] || \
           [[ "$ver" == "(3, 10)" ]] || \
           [[ "$ver" == "(3, 11)" ]] || \
           [[ "$ver" == "(3, 12)" ]] || \
           [[ "$ver" == "(3, 13)" ]]; then
            PYTHON="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    error "Python 3.10 or higher is required but was not found."
    error "Install it from https://www.python.org/downloads/ and re-run this script."
    exit 1
fi

PY_VERSION=$("$PYTHON" --version 2>&1)
info "Using $PY_VERSION at $(command -v "$PYTHON")"

# ── Create virtual environment
if [ -d "$VENV_DIR" ]; then
    warn "Virtual environment already exists at $VENV_DIR — skipping creation."
    warn "Delete it and re-run to start fresh: rm -rf $VENV_DIR"
else
    info "Creating virtual environment at $VENV_DIR..."
    "$PYTHON" -m venv "$VENV_DIR"
    info "Virtual environment created."
fi

# ── Activate
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
info "Activated venv: $(which python) ($(python --version 2>&1))"

# ── Upgrade pip + setuptools
info "Upgrading pip and setuptools..."
pip install --quiet --upgrade pip setuptools wheel

# ── Install package
if [ "$GPU" = true ] && [ "$DEV" = true ]; then
    info "Installing package with dev extras (GPU mode)..."
    pip install --quiet -e "$ROOT[dev]"
elif [ "$DEV" = true ]; then
    info "Installing package with dev extras (CPU mode)..."
    pip install --quiet -e "$ROOT[dev]"
elif [ "$GPU" = true ]; then
    info "Installing package (CPU mode first, then swapping to GPU onnxruntime)..."
    pip install --quiet -e "$ROOT"
else
    info "Installing package (CPU mode)..."
    pip install --quiet -e "$ROOT"
fi

# ── Swap onnxruntime for the requested GPU variant
# Only the onnxruntime-gpu wheel is installed; the heavy CUDA runtime libraries
# (cudnn/cublas/cudart) are expected on the host. Missing -> CPU fallback at runtime.
if [ "$GPU" = true ]; then
    info "Switching to onnxruntime-gpu ($ORT_GPU_SPEC)..."
    pip uninstall -y onnxruntime onnxruntime-gpu 2>/dev/null || true
    pip install --quiet "$ORT_GPU_SPEC"
    info "onnxruntime-gpu installed."
    if [ "$VARIANT" = gpu-cuda12 ]; then
        warn "GPU acceleration needs host CUDA 12 + cuDNN 9 and NVIDIA driver >= R525."
    else
        warn "GPU acceleration needs host CUDA 11.8 + cuDNN 8."
    fi
    warn "Without them, Prof Annotate still runs — it just uses the CPU."
fi

# ── Model check
MODEL_PATH="$ROOT/models/yolo11n_segpose.onnx"
if [ -f "$MODEL_PATH" ]; then
    info "ONNX model found: $MODEL_PATH"
else
    warn "ONNX model NOT found at $MODEL_PATH"
    warn "Auto-annotation will be disabled until you place the model there."
    warn "Expected file: models/yolo11n_segpose.onnx"
fi

# ── Asset check
ASSETS_DIR="$ROOT/assets"
if [ -d "$ASSETS_DIR" ]; then
    info "Assets directory found: $ASSETS_DIR"
else
    warn "Assets directory missing at $ASSETS_DIR — UI may not render correctly."
fi

# ── Summary
echo ""
echo -e "${GREEN}══════════════════════════════════════════${NC}"
echo -e "${GREEN}  Prof Annotate setup complete.${NC}"
echo -e "${GREEN}══════════════════════════════════════════${NC}"
echo ""
echo "  Variant: $VARIANT$([ "$GPU" = true ] && echo '  (CUDA libs provided by host; CPU fallback if absent)')"
echo "  Dev    : $([ "$DEV" = true ] && echo 'yes (pytest, black, ruff)' || echo 'no')"
echo "  Venv   : $VENV_DIR"
echo ""
echo "  To activate the environment:"
echo "    source $VENV_DIR/bin/activate"
echo ""
echo "  To launch Prof Annotate:"
echo "    python $ROOT/main.py"
