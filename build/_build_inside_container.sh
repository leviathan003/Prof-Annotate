#!/usr/bin/env bash
# build/_build_inside_container.sh
# Runs INSIDE a low-glibc build container (default: manylinux_2_28_x86_64).
# Installs Qt/X11 build deps, creates a venv, installs the package + Nuitka, then
# invokes build_appimage.sh for each requested variant. Shared by the local
# wrapper (build_in_container.sh) and the GitHub Actions workflow.
#
# Usage (inside container):
#   bash build/_build_inside_container.sh [cpu|gpu-cuda12|gpu-cuda11|all ...]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
# shellcheck source=build/_log.sh
source "$SCRIPT_DIR/_log.sh"
log_init "$ROOT/build"

# ── Variants (positional). None or "all" -> all three. ────────────────────────
VARIANTS=("$@")
if [ "${#VARIANTS[@]}" -eq 0 ] || [ "${VARIANTS[0]}" = all ]; then
    VARIANTS=(cpu gpu-cuda12 gpu-cuda11)
fi
step "Container build for variants: ${VARIANTS[*]}"
log "glibc: $(ldd --version 2>/dev/null | head -1 || echo unknown)"

# ── System libraries so Nuitka can detect + bundle the Qt/X11 runtime deps ─────
QT_LIBS_DNF="libxcb libX11 libXext libXrender libXi libSM libICE libxkbcommon libxkbcommon-x11 mesa-libGL mesa-libEGL fontconfig freetype wget patchelf binutils which"
QT_LIBS_APT="libxcb1 libx11-6 libxext6 libxrender1 libxi6 libsm6 libice6 libxkbcommon0 libxkbcommon-x11-0 libgl1 libegl1 libfontconfig1 wget patchelf binutils"
step "Installing Qt/X11 runtime + build tools"
if command -v dnf &>/dev/null; then
    # Robust opts: time out stalled mirrors (don't hang forever), fail fast on
    # dead-slow transfers so dnf fails over to another mirror, skip docs/weak deps.
    DNF_OPTS="--setopt=timeout=45 --setopt=retries=10 --setopt=minrate=1k --setopt=fastestmirror=True --setopt=install_weak_deps=False --nodocs"
    # Retry the whole transaction a few times to ride out transient network stalls.
    _dnf() {
        local tries=0
        while [ "$tries" -lt 4 ]; do
            tries=$((tries + 1))
            log "dnf attempt $tries: install $*"
            # shellcheck disable=SC2086
            if dnf install -y $DNF_OPTS "$@"; then return 0; fi
            warn "dnf attempt $tries failed/stalled — cleaning metadata and retrying"
            dnf clean all >/dev/null 2>&1 || true
            sleep 4
        done
        return 1
    }
    # shellcheck disable=SC2086
    if ! _dnf $QT_LIBS_DNF; then
        pkg_manual "$QT_LIBS_DNF"
        die "could not install system build dependencies (dnf) after retries"
    fi
    ok "dnf packages installed"
    # ccache speeds up repeat Nuitka compiles (CI cache); best-effort, never fatal.
    command -v ccache >/dev/null || _dnf ccache >/dev/null 2>&1 || true
elif command -v apt-get &>/dev/null; then
    apt-get update -qq || true
    # shellcheck disable=SC2086
    if ! apt-get install -y -qq $QT_LIBS_APT >/dev/null; then
        pkg_manual "$QT_LIBS_APT"
        die "could not install system build dependencies (apt)"
    fi
    ok "apt packages installed"
else
    warn "no dnf/apt found — assuming Qt/X11 libs are already present in the image"
fi

# manylinux ships Python WITHOUT a shared libpython; Nuitka needs the static
# libpython, provided as a tarball that must be unpacked once. Harmless elsewhere.
if [ -f /opt/_internal/static-libs-for-embedding-only.tar.xz ]; then
    step "Unpacking manylinux static libpython (required by Nuitka)"
    ( cd /opt/_internal && tar xf static-libs-for-embedding-only.tar.xz ) \
        && ok "static libpython ready" \
        || warn "could not unpack static libpython — Nuitka may fail"
fi

# ── Python 3.10+ (manylinux ships several under /opt/python) ───────────────────
PYBIN=""
for cand in /opt/python/cp312-cp312/bin/python /opt/python/cp311-cp311/bin/python \
            /opt/python/cp310-cp310/bin/python python3.12 python3.11 python3.10 python3; do
    if command -v "$cand" &>/dev/null; then PYBIN="$(command -v "$cand")"; break; fi
done
if [ -z "$PYBIN" ]; then
    manual "No Python 3.10+ in the build image" <<EOF
The default image (manylinux_2_28_x86_64) ships CPython under /opt/python.
If you passed a custom --image, ensure it has Python >= 3.10, e.g.:
    --image quay.io/pypa/manylinux_2_28_x86_64
EOF
    die "no Python 3.10+ found in container"
fi
ok "Python: $PYBIN ($("$PYBIN" --version 2>&1))"

# ── Build venv (outside the bind-mounted repo so we don't pollute it) ──────────
VENV=/tmp/profannotate-buildvenv
rm -rf "$VENV"
step "Creating build venv + installing deps"
"$PYBIN" -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
run "upgrade pip"        pip install --quiet --upgrade pip setuptools wheel || die "pip upgrade failed"
run "install package"    pip install --quiet -e "$ROOT" || die "editable install failed (check pyproject.toml)"
run "install nuitka"     pip install --quiet nuitka ordered-set zstandard || die "nuitka install failed"
ok "build environment ready"

export CONDA_PREFIX="$VENV"   # build_appimage.sh keys on this for its Python
export CI=true                # forces the FUSE-free appimagetool path

# ── Build each variant ─────────────────────────────────────────────────────────
for v in "${VARIANTS[@]}"; do
    case "$v" in
        cpu|gpu-cuda12|gpu-cuda11) ;;
        *) die "unknown variant '$v' (expected cpu|gpu-cuda12|gpu-cuda11|all)" ;;
    esac
    step "Building variant: $v"
    bash "$SCRIPT_DIR/build_appimage.sh" "--$v" || die "build failed for variant '$v'"
done

# ── Hand build artifacts back to the host user (container runs as root) ─────────
if [ -n "${HOST_UID:-}" ] && [ -n "${HOST_GID:-}" ]; then
    chown -R "$HOST_UID:$HOST_GID" \
        "$ROOT"/ProfAnnotate-*-x86_64.AppImage \
        "$ROOT/dist_nuitka" \
        "$ROOT"/*.egg-info \
        "$ROOT/build/logs" \
        "$ROOT/build/patchelf-0.17.2" 2>/dev/null || true
fi

step "Container build complete"
ls -lh "$ROOT"/ProfAnnotate-*-x86_64.AppImage 2>/dev/null || warn "no artifacts found"
