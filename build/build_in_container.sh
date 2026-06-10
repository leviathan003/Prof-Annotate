#!/usr/bin/env bash
# build/build_in_container.sh
# Build portable AppImages inside a low-glibc container (default:
# manylinux_2_28_x86_64, glibc 2.28) so the result runs on a wide range of
# distros — Debian 10+, Ubuntu 18.04→24.04+, Mint, Fedora, RHEL 8+, Arch/CachyOS,
# and other rolling releases.
#
# Usage:
#   bash build/build_in_container.sh                 # all three variants (default)
#   bash build/build_in_container.sh --all
#   bash build/build_in_container.sh --cpu
#   bash build/build_in_container.sh --gpu-cuda12
#   bash build/build_in_container.sh --gpu-cuda11
#   bash build/build_in_container.sh --image <docker-image>
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
# shellcheck source=build/_log.sh
source "$SCRIPT_DIR/_log.sh"
log_init "$ROOT/build"

IMAGE="quay.io/pypa/manylinux_2_28_x86_64"
VARIANTS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --cpu)              VARIANTS+=(cpu) ;;
        --gpu|--gpu-cuda12) VARIANTS+=(gpu-cuda12) ;;
        --gpu-cuda11)       VARIANTS+=(gpu-cuda11) ;;
        --all)              VARIANTS=(cpu gpu-cuda12 gpu-cuda11) ;;
        --image)            shift; IMAGE="${1:-}"; [ -n "$IMAGE" ] || die "--image needs an argument" ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown argument: $1 (try --help)" ;;
    esac
    shift
done
[ "${#VARIANTS[@]}" -eq 0 ] && VARIANTS=(cpu gpu-cuda12 gpu-cuda11)

step "Containerized build"
log "image    : $IMAGE"
log "variants : ${VARIANTS[*]}"
log "repo     : $ROOT"

# ── Docker availability ────────────────────────────────────────────────────────
if ! command -v docker >/dev/null 2>&1; then
    err "docker not found"
    manual "Install Docker (needed for the portable, low-glibc build)" <<'EOF'
  Debian / Ubuntu / Mint:
    sudo apt-get update && sudo apt-get install -y docker.io
    sudo systemctl enable --now docker
    sudo usermod -aG docker "$USER"   # then log out/in so 'docker' works without sudo

  Fedora / RHEL / Alma:
    sudo dnf install -y docker && sudo systemctl enable --now docker

  Arch / CachyOS / Omarchy:
    sudo pacman -S --needed docker && sudo systemctl enable --now docker

Alternative without containers (uses host glibc — less portable):
    bash scripts/setup.sh   # or activate an env, then: bash build/build_appimage.sh
EOF
    die "docker is required for build_in_container.sh"
fi

if ! docker info >/dev/null 2>&1; then
    err "docker is installed but the daemon is not reachable (or needs sudo)"
    manual "Start Docker / fix permissions" <<'EOF'
  Start the daemon:
    sudo systemctl start docker

  Run without sudo (recommended): add yourself to the 'docker' group, then
  log out and back in:
    sudo usermod -aG docker "$USER"

  Or run this script with sudo (artifacts are chowned back to you):
    sudo bash build/build_in_container.sh ...
EOF
    die "docker daemon not available"
fi
ok "docker: $(docker --version 2>/dev/null)"

# ── Run the shared inner build inside the container ────────────────────────────
step "Launching build inside $IMAGE"
if ! run "docker run ($IMAGE)" docker run --rm \
        -v "$ROOT":/work \
        -w /work \
        -e PIP_ROOT_USER_ACTION=ignore \
        -e HOST_UID="$(id -u)" \
        -e HOST_GID="$(id -g)" \
        -e BUILD_LOG="/work/build/logs/container-build.log" \
        "$IMAGE" \
        bash /work/build/_build_inside_container.sh "${VARIANTS[@]}"; then
    manual "The containerized build failed — common causes" <<EOF
1. Image pull failed (no network / proxy): pre-pull with
     docker pull $IMAGE
2. Disk space: Nuitka + 3 variants needs several GB. Check 'df -h' and
     docker system prune
3. Inspect the detailed log written inside the repo:
     build/logs/container-build.log
     ${BUILD_LOG:-}
EOF
    die "containerized build failed"
fi

step "Done"
ls -lh "$ROOT"/ProfAnnotate-*-x86_64.AppImage 2>/dev/null || warn "no AppImage artifacts found"
