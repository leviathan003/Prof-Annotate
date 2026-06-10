#!/usr/bin/env bash
# build/_log.sh — shared logging + remediation helpers for Prof Annotate builds.
#
# Source this AFTER defining $ROOT (or pass a dir to log_init). It provides:
#   log_init <dir>      mirror all output to <dir>/logs/build-<ts>.log
#   step / log / ok / warn / err / die / hr
#   run "<label>" cmd…  run a command, stream + log it, die with context on failure
#   need_cmd <cmd> [pkg]            ensure a CLI tool exists, else print install manual
#   need_pymod <py> <module> <pip>  ensure a Python module imports, else manual
#   pkg_manual <pkg…>   print distro-specific install commands
#   manual <title>      wrap stdin in a boxed remediation block
#
# Honors NO_COLOR. Safe to source more than once.

if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
    C_G=$'\033[0;32m'; C_Y=$'\033[1;33m'; C_R=$'\033[0;31m'
    C_B=$'\033[0;36m'; C_D=$'\033[2m'; C_N=$'\033[0m'
else
    C_G=''; C_Y=''; C_R=''; C_B=''; C_D=''; C_N=''
fi

_ts() { date '+%H:%M:%S' 2>/dev/null || echo "--:--:--"; }

log_init() {
    local base="${1:-.}"
    if [ -z "${BUILD_LOG:-}" ]; then
        if mkdir -p "$base/logs" 2>/dev/null; then
            BUILD_LOG="$base/logs/build-$(date +%Y%m%d-%H%M%S 2>/dev/null || echo run).log"
        else
            BUILD_LOG=/dev/null
        fi
    fi
    export BUILD_LOG
    # Only the outermost script redirects; nested children inherit the tee'd
    # stream and reuse the same BUILD_LOG (avoids double-logging).
    if [ -n "${_LOG_REDIRECTED:-}" ]; then
        return 0
    fi
    if [ "$BUILD_LOG" != /dev/null ]; then
        : >> "$BUILD_LOG" 2>/dev/null || { BUILD_LOG=/dev/null; export BUILD_LOG; return 0; }
        # Mirror everything (stdout+stderr) to console AND the logfile.
        exec > >(tee -a "$BUILD_LOG") 2>&1
        export _LOG_REDIRECTED=1
        printf "${C_D}[log] full build log -> %s${C_N}\n" "$BUILD_LOG"
    fi
}

step() { printf "\n${C_B}==> %s${C_N}\n" "$*"; }
log()  { printf "${C_D}[%s]${C_N} %s\n" "$(_ts)" "$*"; }
ok()   { printf "${C_G}  ✓ %s${C_N}\n" "$*"; }
warn() { printf "${C_Y}  ! %s${C_N}\n" "$*"; }
err()  { printf "${C_R}  ✗ %s${C_N}\n" "$*" >&2; }
hr()   { printf "${C_D}%s${C_N}\n" "------------------------------------------------------------"; }

# manual <title>: wrap stdin in a boxed remediation block (for setup guidance).
manual() {
    printf "\n${C_Y}┌─ %s ${C_N}\n" "$1"
    while IFS= read -r line; do printf "${C_Y}│${C_N} %s\n" "$line"; done
    printf "${C_Y}└──────────────────────────────────────────────────────────${C_N}\n\n"
}

die() {
    err "$*"
    [ -n "${BUILD_LOG:-}" ] && [ "$BUILD_LOG" != /dev/null ] && printf "${C_D}    full log: %s${C_N}\n" "$BUILD_LOG"
    exit 1
}

# run "<label>" cmd args…  — log + execute; on failure print label, exit code, log path.
run() {
    local label="$1"; shift
    log "$label"
    log "\$ $*"
    local rc=0
    "$@" || rc=$?
    if [ "$rc" -ne 0 ]; then
        err "step failed (exit $rc): $label"
        printf "${C_D}    command: %s${C_N}\n" "$*"
        [ -n "${BUILD_LOG:-}" ] && [ "$BUILD_LOG" != /dev/null ] && printf "${C_D}    full log: %s${C_N}\n" "$BUILD_LOG"
        return "$rc"
    fi
}

# pkg_manual <pkg…> — print distro-specific install commands for system packages.
pkg_manual() {
    local pkgs="$*"
    manual "Install missing system package(s): $pkgs" <<EOF
Pick the line for your distribution:

  Debian / Ubuntu / Mint / Pop!_OS:
    sudo apt-get update && sudo apt-get install -y $pkgs

  Fedora / RHEL / AlmaLinux / Rocky:
    sudo dnf install -y $pkgs

  Arch / CachyOS / Omarchy / EndeavourOS:
    sudo pacman -S --needed $pkgs

  openSUSE:
    sudo zypper install -y $pkgs

Then re-run this build.
EOF
}

# need_cmd <cmd> [pkg] — ensure a CLI tool is on PATH, else print a manual and die.
need_cmd() {
    local cmd="$1" pkg="${2:-$1}"
    if ! command -v "$cmd" >/dev/null 2>&1; then
        err "required command not found: $cmd"
        pkg_manual "$pkg"
        die "install '$cmd' and re-run"
    fi
    ok "$cmd: $(command -v "$cmd")"
}

# need_pymod <python> <module> <pip-spec> — ensure a Python module imports.
need_pymod() {
    local py="$1" mod="$2" spec="$3"
    if ! "$py" -c "import $mod" >/dev/null 2>&1; then
        err "Python module '$mod' not importable with $py"
        manual "Install the missing Python dependency" <<EOF
Activate the build environment, then:
    $py -m pip install "$spec"

If you are bootstrapping from scratch, install everything at once:
    $py -m pip install -e .            # runtime deps
    $py -m pip install nuitka ordered-set zstandard
EOF
        die "missing Python module: $mod"
    fi
    ok "python module: $mod"
}
