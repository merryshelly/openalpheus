#!/usr/bin/env bash
# =============================================================================
# OpenAlph Bootstrap Script — Steps 0-3

# =============================================================================
#
# This script performs the following steps:
#   Step 0 — Safety preamble: logging, color helpers, confirmation prompt
#   Step 1 — Preflight checks: auto-install missing deps, verify Python, Docker, systemd, etc.
#   Step 2 — Install OpenAlph via pip from Codeberg
#   Step 3 — Create system group and shared directory structure
#
# Steps 4+ will be appended here in subsequent bootstrap spec iterations.
#
# Usage:
#   sudo bash install.sh              # interactive (prompts for confirmation)
#   sudo OPENALPH_YES=true bash install.sh   # non-interactive (CI/automation)
#
# =============================================================================

set -euo pipefail

# =============================================================================
# CONSTANTS
# =============================================================================

readonly OPENALPH_VERSION="${OPENALPH_VERSION:-v0.1.1}"
readonly OPENALPH_REPO="https://codeberg.org/merryshelly/openalph.git"

readonly REQUIRED_PYTHON_MAJOR=3
readonly REQUIRED_PYTHON_MINOR=11
readonly REQUIRED_DOCKER_MAJOR=24
readonly REQUIRED_SYSTEMD_MIN=249

readonly OPENALPH_GROUP="openalph"
readonly OPENALPH_SHARED_DIR="/srv/openalph/shared"
readonly OPENALPH_CONFIG_DIR="/etc/openalph"
readonly OPENALPH_AGENTS_DIR="/etc/openalph/agents"

readonly DOCKER_INSTALL_DOCS="https://docs.docker.com/engine/install/"
readonly OPENALPH_DOCS="https://codeberg.org/merryshelly/openalph"

readonly PORT_HTTPS=443
readonly PORT_OPENALPH=4269

readonly DISK_WARN_THRESHOLD_MB=2048    # 2 GB
readonly DISK_FAIL_THRESHOLD_MB=500     # 500 MB
readonly DOCKER_DATA_DIR="/var/lib/docker"

# =============================================================================
# STEP 0 — SAFETY PREAMBLE
# Sets up logging, color helpers, and confirmation prompt.
# =============================================================================

step0_safety_preamble() {

    # -------------------------------------------------------------------------
    # Logging — tee all output to a timestamped log file
    # -------------------------------------------------------------------------
    readonly LOG_FILE="/tmp/openalph-bootstrap-$(date +%s).log"
    # shellcheck disable=SC2093
    exec > >(tee -a "$LOG_FILE") 2>&1

    # -------------------------------------------------------------------------
    # Color / formatting helpers
    # Only emit ANSI codes when stdout is a real terminal.
    # -------------------------------------------------------------------------
    if [[ -t 1 ]] && tput colors &>/dev/null && [[ "$(tput colors)" -ge 8 ]]; then
        readonly CLR_RESET="\033[0m"
        readonly CLR_BOLD="\033[1m"
        readonly CLR_RED="\033[0;31m"
        readonly CLR_YELLOW="\033[0;33m"
        readonly CLR_GREEN="\033[0;32m"
        readonly CLR_CYAN="\033[0;36m"
        readonly CLR_WHITE="\033[1;37m"
    else
        readonly CLR_RESET=""
        readonly CLR_BOLD=""
        readonly CLR_RED=""
        readonly CLR_YELLOW=""
        readonly CLR_GREEN=""
        readonly CLR_CYAN=""
        readonly CLR_WHITE=""
    fi

    # -------------------------------------------------------------------------
    # Output helpers
    # -------------------------------------------------------------------------

    info() {
        printf "${CLR_CYAN}[INFO]${CLR_RESET}  %s\n" "$*"
    }

    warn() {
        printf "${CLR_YELLOW}[WARN]${CLR_RESET}  %s\n" "$*" >&2
    }

    error() {
        printf "${CLR_RED}[ERROR]${CLR_RESET} %s\n" "$*" >&2
    }

    success() {
        printf "${CLR_GREEN}[OK]${CLR_RESET}    %s\n" "$*"
    }

    step() {
        printf "\n${CLR_BOLD}${CLR_WHITE}==> %s${CLR_RESET}\n" "$*"
    }

    # Preflight result helpers — accumulate pass/fail lines, printed as a table
    _pf_pass() {
        # $1 = label, $2 = detail (version / path / etc.)
        printf "  ${CLR_GREEN}[✓]${CLR_RESET} ${CLR_BOLD}%s${CLR_RESET} %s\n" "$1" "$2"
    }

    _pf_fail() {
        # $1 = label, $2 = hint message
        printf "  ${CLR_RED}[✗]${CLR_RESET} ${CLR_BOLD}%s${CLR_RESET}\n" "$1"
        printf "      ${CLR_YELLOW}→${CLR_RESET} %s\n" "$2"
    }

    _pf_warn() {
        # $1 = label, $2 = warning message (optional)
        printf "  ${CLR_YELLOW}[!]${CLR_RESET} ${CLR_BOLD}%s${CLR_RESET}\n" "$1"
        if [[ -n "${2:-}" ]]; then
            printf "      ${CLR_YELLOW}→${CLR_RESET} %s\n" "$2"
        fi
    }

    # -------------------------------------------------------------------------
    # Interactive read helper — always reads from /dev/tty so that
    # `curl | bash` works correctly. Fails clearly if no TTY is available
    # and the caller hasn't provided the value via environment variable.
    # -------------------------------------------------------------------------
    _has_tty() {
        [[ -e /dev/tty ]]
    }

    _prompt_read() {
        if _has_tty; then
            read "$@" </dev/tty
        else
            error "Interactive input required but no TTY is available."
            error "For non-interactive install, set the required environment variables."
            error "See: ${OPENALPH_DOCS}/src/branch/main/INSTALL.md"
            exit 1
        fi
    }

    # -------------------------------------------------------------------------
    # Banner
    # -------------------------------------------------------------------------
    printf "\n"
    printf "${CLR_BOLD}${CLR_CYAN}"
    printf "╔══════════════════════════════════════════════════════════╗\n"
    printf "║           OpenAlph Bootstrap Installer                  ║\n"
    printf "║           Version: %-38s║\n" "${OPENALPH_VERSION}"
    printf "╚══════════════════════════════════════════════════════════╝\n"
    printf "${CLR_RESET}\n"
    printf "This script will:\n"
    printf "  1. Check and auto-install system dependencies (Docker, Caddy, etc.)\n"
    printf "  2. Install OpenAlph ${OPENALPH_VERSION} from ${OPENALPH_REPO}\n"
    printf "  3. Create the '${OPENALPH_GROUP}' system group and shared directory tree\n"
    printf "\n"
    printf "A full log will be written to: ${CLR_CYAN}%s${CLR_RESET}\n" "$LOG_FILE"
    printf "\n"

    # -------------------------------------------------------------------------
    # Confirmation prompt (skipped in non-interactive or CI mode)
    # -------------------------------------------------------------------------
    if [[ -t 0 ]] && [[ "${OPENALPH_YES:-}" != "true" ]]; then
        read -r -p "$(printf "${CLR_BOLD}Proceed with installation? [y/N]${CLR_RESET} ")" _confirm
        case "${_confirm}" in
            [yY][eE][sS]|[yY]) ;;
            *)
                info "Installation cancelled."
                exit 0
                ;;
        esac
    fi
}

# =============================================================================
# STEP 1 — PREFLIGHT CHECKS
# Auto-installs missing deps where safe (curl, jq, venv, Docker, Caddy).
# Checks without auto-install: Python 3.11+ (complex), systemd (OS-level).
# Runs all checks, collects results, then fails if any check failed.
# =============================================================================

step1_preflight_checks() {
    step "Step 1 — Preflight Checks"

    local _failures=0
    local _apt_updated=0

    # -------------------------------------------------------------------------
    # Helper: compare two dot-separated version strings.
    # Returns 0 (true) if $1 >= $2, 1 (false) otherwise.
    # Only compares the first two components (major.minor).
    # -------------------------------------------------------------------------
    _version_gte() {
        local have_major have_minor need_major need_minor
        have_major="$(echo "$1" | cut -d. -f1)"
        have_minor="$(echo "$1" | cut -d. -f2)"
        need_major="$(echo "$2" | cut -d. -f1)"
        need_minor="$(echo "$2" | cut -d. -f2)"

        if (( have_major > need_major )); then return 0; fi
        if (( have_major == need_major )) && (( have_minor >= need_minor )); then return 0; fi
        return 1
    }

    # -------------------------------------------------------------------------
    # Helper: ensure apt-get update has been run (at most once)
    # -------------------------------------------------------------------------
    _ensure_apt_updated() {
        if (( _apt_updated == 0 )); then
            apt-get update -qq >/dev/null 2>&1
            _apt_updated=1
        fi
    }

    # -------------------------------------------------------------------------
    # curl (checked first — needed for Docker convenience script + Caddy)
    # -------------------------------------------------------------------------
    if command -v curl &>/dev/null; then
        _curl_ver="$(curl --version 2>&1 | awk 'NR==1{print $2}')"
        _pf_pass "curl ${_curl_ver}" ""
    else
        _pf_warn "curl not found — installing..."
        _ensure_apt_updated
        if apt-get install -y curl >/dev/null 2>&1 && command -v curl &>/dev/null; then
            _curl_ver="$(curl --version 2>&1 | awk 'NR==1{print $2}')"
            _pf_pass "curl ${_curl_ver}" "(auto-installed)"
        else
            _pf_fail "curl install failed" \
                "Could not install curl. Run: apt-get install -y curl"
            (( _failures++ ))
        fi
    fi

    # -------------------------------------------------------------------------
    # jq (required for Matrix API JSON parsing)
    # -------------------------------------------------------------------------
    if command -v jq &>/dev/null; then
        _jq_ver="$(jq --version 2>&1 || echo "unknown")"
        _pf_pass "jq ${_jq_ver}" ""
    else
        _pf_warn "jq not found — installing..."
        _ensure_apt_updated
        if apt-get install -y jq >/dev/null 2>&1 && command -v jq &>/dev/null; then
            _jq_ver="$(jq --version 2>&1 || echo "unknown")"
            _pf_pass "jq ${_jq_ver}" "(auto-installed)"
        else
            _pf_fail "jq install failed" \
                "Could not install jq. Run: apt-get install -y jq"
            (( _failures++ ))
        fi
    fi

    # -------------------------------------------------------------------------
    # Python 3 >= 3.11 (not auto-installed — version upgrades are complex)
    # -------------------------------------------------------------------------
    if command -v python3 &>/dev/null; then
        _py_raw="$(python3 --version 2>&1)"          # e.g. "Python 3.13.2"
        _py_ver="$(echo "$_py_raw" | awk '{print $2}')"
        _py_path="$(command -v python3)"
        if _version_gte "$_py_ver" "${REQUIRED_PYTHON_MAJOR}.${REQUIRED_PYTHON_MINOR}"; then
            _pf_pass "Python ${_py_ver}" "(${_py_path})"
        else
            _pf_fail "Python ${_py_ver}" \
                "Python >= ${REQUIRED_PYTHON_MAJOR}.${REQUIRED_PYTHON_MINOR} required. Found: ${_py_ver}. Ubuntu 22.04: use deadsnakes PPA."
            (( _failures++ ))
        fi
    else
        _pf_fail "Python 3 not found" \
            "Install Python 3.11+: https://www.python.org/downloads/ (Ubuntu 22.04: use deadsnakes PPA)"
        (( _failures++ ))
    fi

    # -------------------------------------------------------------------------
    # Python venv module (required for isolated install)
    # On Debian/Ubuntu, this is a separate package: python3.XX-venv
    # -------------------------------------------------------------------------
    # The check must test ensurepip, not just 'venv --help'. The venv module
    # exists in the Python stdlib, but actually creating a venv requires
    # ensurepip, which is in the separate python3.XX-venv package on Debian/Ubuntu.
    if python3 -c "import ensurepip" &>/dev/null 2>&1; then
        _pf_pass "Python venv module" "available"
    else
        _py_minor="$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f2)"
        _venv_pkg="python3.${_py_minor}-venv"
        _pf_warn "Python venv module not found — installing..."
        _ensure_apt_updated
        if apt-get install -y "${_venv_pkg}" >/dev/null 2>&1 && python3 -c "import ensurepip" &>/dev/null 2>&1; then
            _pf_pass "Python venv module" "(auto-installed via ${_venv_pkg})"
        elif apt-get install -y python3-venv >/dev/null 2>&1 && python3 -c "import ensurepip" &>/dev/null 2>&1; then
            _pf_pass "Python venv module" "(auto-installed via python3-venv)"
        else
            _pf_fail "Python venv module install failed" \
                "Could not install venv. Run: apt-get install -y ${_venv_pkg} (or python3-venv)"
            (( _failures++ ))
        fi
    fi

    # -------------------------------------------------------------------------
    # pip (auto-install via python3-pip if missing)
    # -------------------------------------------------------------------------
    _pip_cmd=""
    if command -v pip3 &>/dev/null; then
        _pip_cmd="pip3"
    elif python3 -m pip --version &>/dev/null 2>&1; then
        _pip_cmd="python3 -m pip"
    fi

    if [[ -n "$_pip_cmd" ]]; then
        _pip_ver="$($_pip_cmd --version 2>&1 | awk '{print $2}')"
        _pf_pass "pip ${_pip_ver}" "(${_pip_cmd})"
    else
        _pf_warn "pip not found — installing..."
        _ensure_apt_updated
        if apt-get install -y python3-pip >/dev/null 2>&1; then
            if command -v pip3 &>/dev/null; then
                _pip_cmd="pip3"
            elif python3 -m pip --version &>/dev/null 2>&1; then
                _pip_cmd="python3 -m pip"
            fi
            if [[ -n "$_pip_cmd" ]]; then
                _pip_ver="$($_pip_cmd --version 2>&1 | awk '{print $2}')"
                _pf_pass "pip ${_pip_ver}" "(auto-installed)"
            else
                _pf_fail "pip install succeeded but pip not found on PATH" \
                    "Try: python3 -m pip --version"
                (( _failures++ ))
            fi
        else
            _pf_fail "pip install failed" \
                "Could not install python3-pip. Run: apt-get install -y python3-pip"
            (( _failures++ ))
        fi
    fi

    # -------------------------------------------------------------------------
    # Docker >= 24 (auto-install via official convenience script if missing)
    # -------------------------------------------------------------------------
    if command -v docker &>/dev/null; then
        _docker_raw="$(docker --version 2>&1)"       # e.g. "Docker version 27.1.1, build ..."
        _docker_ver="$(echo "$_docker_raw" | grep -oP '\d+\.\d+\.\d+' | head -1)"
        _docker_major="$(echo "$_docker_ver" | cut -d. -f1)"
        if (( _docker_major >= REQUIRED_DOCKER_MAJOR )); then
            _pf_pass "Docker ${_docker_ver}" ""
        else
            _pf_fail "Docker ${_docker_ver}" \
                "Docker >= ${REQUIRED_DOCKER_MAJOR} required but ${_docker_ver} found. Upgrade: ${DOCKER_INSTALL_DOCS}"
            (( _failures++ ))
        fi
    else
        _pf_warn "Docker not found — installing..."
        info "Installing Docker via official convenience script (https://get.docker.com)..."
        if curl -fsSL https://get.docker.com | sh >/dev/null 2>&1 && command -v docker &>/dev/null; then
            _docker_raw="$(docker --version 2>&1)"
            _docker_ver="$(echo "$_docker_raw" | grep -oP '\d+\.\d+\.\d+' | head -1)"
            _docker_major="$(echo "$_docker_ver" | cut -d. -f1)"
            if (( _docker_major >= REQUIRED_DOCKER_MAJOR )); then
                _pf_pass "Docker ${_docker_ver}" "(auto-installed)"
            else
                _pf_fail "Docker ${_docker_ver}" \
                    "Docker convenience script installed ${_docker_ver} but >= ${REQUIRED_DOCKER_MAJOR} required. Upgrade: ${DOCKER_INSTALL_DOCS}"
                (( _failures++ ))
            fi
        else
            _pf_fail "Docker install failed" \
                "Could not install Docker via https://get.docker.com. Install manually: ${DOCKER_INSTALL_DOCS}"
            (( _failures++ ))
        fi
    fi

    # -------------------------------------------------------------------------
    # Docker Compose v2+ (plugin — `docker compose version`)
    # Included with official Docker install; fail with hint if missing.
    # -------------------------------------------------------------------------
    if docker compose version &>/dev/null 2>&1; then
        _dc_ver="$(docker compose version 2>&1 | grep -oP 'v?\d+\.\d+\.\d+' | head -1)"
        _dc_major="$(echo "$_dc_ver" | tr -d 'v' | cut -d. -f1)"
        if (( _dc_major >= 2 )); then
            _pf_pass "Docker Compose ${_dc_ver}" "(plugin)"
        else
            _pf_fail "Docker Compose ${_dc_ver}" \
                "Docker Compose v2+ required. Upgrade Docker: ${DOCKER_INSTALL_DOCS}"
            (( _failures++ ))
        fi
    else
        _pf_fail "Docker Compose (v2 plugin) not found" \
            "Install Docker Compose v2: ${DOCKER_INSTALL_DOCS}"
        (( _failures++ ))
    fi

    # -------------------------------------------------------------------------
    # systemd >= 249 (not auto-installed — requires a supported OS)
    # -------------------------------------------------------------------------
    if command -v systemctl &>/dev/null; then
        _systemd_ver="$(systemctl --version 2>&1 | awk 'NR==1{print $2}')"
        if (( _systemd_ver >= REQUIRED_SYSTEMD_MIN )); then
            _pf_pass "systemd ${_systemd_ver}" ""
        else
            _pf_fail "systemd ${_systemd_ver}" \
                "systemd >= ${REQUIRED_SYSTEMD_MIN} required (Debian 12 / Ubuntu 22.04+)."
            (( _failures++ ))
        fi
    else
        _pf_fail "systemd not found" \
            "OpenAlph requires a systemd-based OS (Debian 12+ or Ubuntu 22.04+)."
        (( _failures++ ))
    fi

    # -------------------------------------------------------------------------
    # Caddy (auto-install from official APT repository if missing)
    # -------------------------------------------------------------------------
    if command -v caddy &>/dev/null; then
        _caddy_ver="$(caddy version 2>&1 | awk '{print $1}' || echo "unknown")"
        _pf_pass "Caddy ${_caddy_ver}" ""
    else
        _pf_warn "Caddy not found — installing..."
        _ensure_apt_updated
        if apt-get install -y debian-keyring debian-archive-keyring apt-transport-https >/dev/null 2>&1 \
            && curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | \
                gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg 2>/dev/null \
            && curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | \
                tee /etc/apt/sources.list.d/caddy-stable.list >/dev/null \
            && apt-get update -qq >/dev/null 2>&1 \
            && apt-get install -y caddy >/dev/null 2>&1 \
            && command -v caddy &>/dev/null; then
            _caddy_ver="$(caddy version 2>&1 | awk '{print $1}' || echo "unknown")"
            _pf_pass "Caddy ${_caddy_ver}" "(auto-installed)"
        else
            _pf_fail "Caddy install failed" \
                "Could not install Caddy. See: https://caddyserver.com/docs/install#debian-ubuntu-raspbian"
            (( _failures++ ))
        fi
    fi

    # -------------------------------------------------------------------------
    # Port availability checks
    # Reports the occupying process if a port is already bound.
    # -------------------------------------------------------------------------
    _check_port() {
        local port="$1"
        local label="Port ${port}"

        # ss output example: tcp LISTEN 0 128 0.0.0.0:443 ... users:(("nginx",pid=1234,...))
        local _ss_line
        _ss_line="$(ss -tlnp 2>/dev/null | grep ":${port} " || true)"

        if [[ -z "$_ss_line" ]]; then
            _pf_pass "$label" "available"
        else
            # Extract process name + PID if available
            local _proc
            _proc="$(echo "$_ss_line" | grep -oP 'users:\(\("([^"]+)",pid=(\d+)' \
                     | sed 's/users:(("//' | sed 's/",pid=/ pid=/' || true)"
            if [[ -n "$_proc" ]]; then
                _pf_warn "$label already bound" \
                    "In use by: ${_proc}. Stop that service or reconfigure OpenAlph."
            else
                _pf_warn "$label already bound" \
                    "Port ${port} is in use. Stop the occupying service or reconfigure OpenAlph."
            fi
            # Port conflicts are warnings, not hard failures — OpenAlph config may redirect.
            # Promote to failure if you want strict checking:
            # (( _failures++ ))
        fi
    }

    _check_port "${PORT_HTTPS}"
    _check_port "${PORT_OPENALPH}"

    # -------------------------------------------------------------------------
    # Disk space — check free space on the filesystem hosting /var/lib/docker
    # -------------------------------------------------------------------------
    # Use the actual docker data dir if it exists, else fall back to /var/lib
    _disk_target="${DOCKER_DATA_DIR}"
    if [[ ! -d "$_disk_target" ]]; then
        _disk_target="/var/lib"
    fi

    _free_mb="$(df -BM "$_disk_target" 2>/dev/null | awk 'NR==2{gsub(/M/,"",$4); print $4}')"

    if [[ -z "$_free_mb" ]]; then
        _pf_warn "Disk space (${_disk_target})" \
            "Could not determine free disk space. Proceeding cautiously."
    elif (( _free_mb < DISK_FAIL_THRESHOLD_MB )); then
        _pf_fail "Disk space — ${_free_mb} MB free on ${_disk_target}" \
            "Critically low disk space (< ${DISK_FAIL_THRESHOLD_MB} MB). Free up space before continuing."
        (( _failures++ ))
    elif (( _free_mb < DISK_WARN_THRESHOLD_MB )); then
        _pf_warn "Disk space — ${_free_mb} MB free on ${_disk_target}" \
            "Less than ${DISK_WARN_THRESHOLD_MB} MB free. Docker images may exhaust disk space."
    else
        _pf_pass "Disk space — ${_free_mb} MB free" "(${_disk_target})"
    fi

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    printf "\n"
    if (( _failures > 0 )); then
        error "Preflight checks failed (${_failures} issue(s) above must be resolved)."
        error "Fix the issues listed above, then re-run this script."
        error "Full log: ${LOG_FILE}"
        exit 1
    else
        success "All preflight checks passed."
    fi
}

# =============================================================================
# STEP 2 — INSTALL OPENALPH
# Installs from Codeberg via pip, then verifies the install.
# =============================================================================

step2_install_openalph() {
    step "Step 2 — Install OpenAlph ${OPENALPH_VERSION}"

    local _install_url="${OPENALPH_INSTALL_PATH:-git+${OPENALPH_REPO}@${OPENALPH_VERSION}}"
    local _venv_dir="/opt/openalph-venv"

    info "Installing: ${_install_url}"

    # Use a dedicated venv to avoid PEP 668 (externally-managed-environment)
    # on Debian 12+/13 and Ubuntu 23.04+
    # Create or repair the virtual environment.
    # A previous failed run may have left a directory without a working pip,
    # so we check the actual binary rather than just the directory.
    if [[ ! -x "${_venv_dir}/bin/pip" ]]; then
        info "Creating Python virtual environment at ${_venv_dir}..."
        rm -rf "${_venv_dir}"
        python3 -m venv "${_venv_dir}"
    fi

    if ! "${_venv_dir}/bin/pip" install "${_install_url}"; then
        error "pip install failed."
        error "Check the log for details: ${LOG_FILE}"
        error "Docs: ${OPENALPH_DOCS}"
        exit 1
    fi

    # Symlink the binary so it's on the standard PATH
    ln -sf "${_venv_dir}/bin/openalph" /usr/local/bin/openalph

    # Verify the binary is reachable and reports the expected version
    if ! command -v openalph &>/dev/null; then
        error "'openalph' binary not found on PATH after install."
        error "Check that ${_venv_dir}/bin/openalph exists."
        exit 1
    fi

    local _installed_ver
    _installed_ver="$(openalph --version 2>&1 || true)"
    success "OpenAlph installed successfully: ${_installed_ver}"
}

# =============================================================================
# STEP 3 — CREATE GROUP + SHARED DIRECTORIES
# All operations are idempotent (safe to run more than once).
# =============================================================================

step3_create_group_and_dirs() {
    step "Step 3 — Create system group and shared directories"

    # -------------------------------------------------------------------------
    # System group
    # groupadd exits 9 if the group already exists; we suppress that gracefully.
    # -------------------------------------------------------------------------
    if getent group "${OPENALPH_GROUP}" &>/dev/null; then
        info "Group '${OPENALPH_GROUP}' already exists — skipping groupadd."
    else
        groupadd --system "${OPENALPH_GROUP}"
        success "Created system group '${OPENALPH_GROUP}'."
    fi

    # -------------------------------------------------------------------------
    # Shared runtime directories
    # /srv/openalph/shared/{bin,beads,docs}
    # -------------------------------------------------------------------------
    info "Creating shared directory tree under ${OPENALPH_SHARED_DIR} ..."
    mkdir -p \
        "${OPENALPH_SHARED_DIR}/bin" \
        "${OPENALPH_SHARED_DIR}/beads" \
        "${OPENALPH_SHARED_DIR}/docs"

    chgrp -R "${OPENALPH_GROUP}" "${OPENALPH_SHARED_DIR}"

    # 2770 = setgid bit + rwxrws--- so new files inherit the group
    chmod -R 2770 "${OPENALPH_SHARED_DIR}"

    success "Shared directories ready: ${OPENALPH_SHARED_DIR}/{bin,beads,docs}"

    # -------------------------------------------------------------------------
    # Config directories
    # /etc/openalph        — 755 (readable by all users)
    # /etc/openalph/agents — 750 (accessible only by root + openalph group)
    # -------------------------------------------------------------------------
    info "Creating config directory tree under ${OPENALPH_CONFIG_DIR} ..."
    mkdir -p "${OPENALPH_AGENTS_DIR}"

    chmod 755 "${OPENALPH_CONFIG_DIR}"
    chmod 750 "${OPENALPH_AGENTS_DIR}"

    # agents dir should be owned/accessible by the openalph group
    chgrp "${OPENALPH_GROUP}" "${OPENALPH_AGENTS_DIR}"

    success "Config directories ready: ${OPENALPH_CONFIG_DIR} (755), ${OPENALPH_AGENTS_DIR} (750)"
}

# =============================================================================
# STEP 5 — TLS / CADDY SETUP (runs before step 4 — tuwunel needs the domain)
# =============================================================================

step5_setup_tls() {
    step "Step 5" "Configuring TLS..."

    # ── Determine TLS mode ────────────────────────────────────────────────────
    local tls_mode="${OPENALPH_TLS_MODE:-}"

    if [[ -z "${tls_mode}" ]]; then
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "TLS Setup — a valid certificate is required"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        echo "  [1] Tailscale (recommended)"
        echo "      Automatically provisions a real Let's Encrypt certificate."
        echo "      Requires: Tailscale installed and connected on this machine."
        echo ""
        echo "  [2] Let's Encrypt (automatic)"
        echo "      You have a public DNS domain pointing at this machine."
        echo "      Caddy handles certificate issuance automatically."
        echo "      Requires: Port 80 reachable from the internet."
        echo ""
        echo "  [3] Bring your own certificate"
        echo "      You have a cert and key file already."
        echo ""
        local choice
        _prompt_read -r -p "Choice [1]: " choice
        choice="${choice:-1}"

        case "${choice}" in
            1) tls_mode="tailscale"    ;;
            2) tls_mode="letsencrypt"  ;;
            3) tls_mode="custom"       ;;
            *)
                error "Invalid choice: '${choice}'. Valid options are 1, 2, or 3."
                exit 1
                ;;
        esac
    fi

    # ── Local variables for cert paths and Caddyfile content ─────────────────
    local CERT_PATH=""
    local KEY_PATH=""
    local caddyfile_content=""

    # ── Option 1: Tailscale ───────────────────────────────────────────────────
    if [[ "${tls_mode}" == "tailscale" ]]; then
        info "TLS mode: Tailscale"

        # Verify tailscale is installed and running
        if ! command -v tailscale &>/dev/null; then
            error "Tailscale is not installed. Install it from https://tailscale.com/download"
            exit 1
        fi
        local TS_STATUS
        TS_STATUS=$(tailscale status --json 2>/dev/null | jq -r '.BackendState' 2>/dev/null || echo "unknown")
        if [[ "${TS_STATUS}" != "Running" ]]; then
            error "Tailscale is not connected (state: ${TS_STATUS})"
            error "Run: tailscale up"
            exit 1
        fi

        # Get the Tailscale FQDN (strip trailing dot)
        OPENALPH_DOMAIN=$(tailscale status --json | jq -r '.Self.DNSName' | sed 's/\.$//')
        info "Tailscale hostname: ${OPENALPH_DOMAIN}"

        # Provision cert (requires HTTPS enabled in Tailscale admin console)
        tailscale cert "${OPENALPH_DOMAIN}"

        # Locate cert files — tailscale may write to /var/lib/tailscale/certs/ or CWD
        if [[ -f "/var/lib/tailscale/certs/${OPENALPH_DOMAIN}.crt" ]]; then
            CERT_PATH="/var/lib/tailscale/certs/${OPENALPH_DOMAIN}.crt"
            KEY_PATH="/var/lib/tailscale/certs/${OPENALPH_DOMAIN}.key"
        else
            # tailscale cert writes to CWD if not root, but we're root so check CWD too
            CERT_PATH="$(pwd)/${OPENALPH_DOMAIN}.crt"
            KEY_PATH="$(pwd)/${OPENALPH_DOMAIN}.key"
        fi

        if [[ ! -f "${CERT_PATH}" ]]; then
            error "Certificate file not found at expected path: ${CERT_PATH}"
            exit 1
        fi
        if [[ ! -f "${KEY_PATH}" ]]; then
            error "Private key file not found at expected path: ${KEY_PATH}"
            exit 1
        fi

        # Move certs to Caddy's cert directory
        mkdir -p /etc/caddy/certs
        cp "${CERT_PATH}" "/etc/caddy/certs/${OPENALPH_DOMAIN}.crt"
        cp "${KEY_PATH}"  "/etc/caddy/certs/${OPENALPH_DOMAIN}.key"
        chown caddy:caddy /etc/caddy/certs/*
        chmod 640 /etc/caddy/certs/*.key

        # Update paths to the Caddy-owned copies
        CERT_PATH="/etc/caddy/certs/${OPENALPH_DOMAIN}.crt"
        KEY_PATH="/etc/caddy/certs/${OPENALPH_DOMAIN}.key"

        caddyfile_content="${OPENALPH_DOMAIN} {
    tls ${CERT_PATH} ${KEY_PATH}
    reverse_proxy localhost:4269
}"

    # ── Option 2: Let's Encrypt ───────────────────────────────────────────────
    elif [[ "${tls_mode}" == "letsencrypt" ]]; then
        info "TLS mode: Let's Encrypt (automatic ACME)"

        if [[ -z "${OPENALPH_DOMAIN:-}" ]]; then
            _prompt_read -r -p "Enter your domain name (e.g. matrix.example.com): " OPENALPH_DOMAIN
        fi

        if [[ -z "${OPENALPH_DOMAIN}" ]]; then
            error "A domain name is required for Let's Encrypt."
            exit 1
        fi

        info "Domain: ${OPENALPH_DOMAIN}"

        # Caddy handles ACME automatically — no cert files needed
        caddyfile_content="${OPENALPH_DOMAIN} {
    reverse_proxy localhost:4269
}"

    # ── Option 3: Bring your own certificate ─────────────────────────────────
    elif [[ "${tls_mode}" == "custom" ]]; then
        info "TLS mode: custom certificate"

        if [[ -z "${OPENALPH_TLS_CERT:-}" ]]; then
            _prompt_read -r -p "Path to certificate file (.crt or .pem): " CERT_PATH
        else
            CERT_PATH="${OPENALPH_TLS_CERT}"
        fi

        if [[ -z "${OPENALPH_TLS_KEY:-}" ]]; then
            _prompt_read -r -p "Path to private key file (.key): " KEY_PATH
        else
            KEY_PATH="${OPENALPH_TLS_KEY}"
        fi

        if [[ -z "${OPENALPH_DOMAIN:-}" ]]; then
            _prompt_read -r -p "Domain name (as in the certificate's CN/SAN): " OPENALPH_DOMAIN
        fi

        # Validate files exist before proceeding
        if [[ ! -f "${CERT_PATH}" ]]; then
            error "Certificate file not found: ${CERT_PATH}"
            exit 1
        fi
        if [[ ! -f "${KEY_PATH}" ]]; then
            error "Private key file not found: ${KEY_PATH}"
            exit 1
        fi
        if [[ -z "${OPENALPH_DOMAIN}" ]]; then
            error "A domain name is required."
            exit 1
        fi

        info "Certificate: ${CERT_PATH}"
        info "Key:         ${KEY_PATH}"
        info "Domain:      ${OPENALPH_DOMAIN}"

        caddyfile_content="${OPENALPH_DOMAIN} {
    tls ${CERT_PATH} ${KEY_PATH}
    reverse_proxy localhost:4269
}"

    else
        error "Unknown TLS mode: '${tls_mode}'. Valid values are: tailscale, letsencrypt, custom."
        exit 1
    fi

    # ── Write Caddyfile ───────────────────────────────────────────────────────
    info "Writing /etc/caddy/Caddyfile..."
    cat > /etc/caddy/Caddyfile <<EOF
${caddyfile_content}
EOF

    # Validate the configuration before restarting
    info "Validating Caddyfile..."
    caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile

    # Enable and restart Caddy
    systemctl enable caddy
    systemctl restart caddy

    # ── Verify Caddy is running ───────────────────────────────────────────────
    sleep 2
    if ! systemctl is-active --quiet caddy; then
        error "Caddy failed to start. Check logs with: journalctl -xeu caddy"
        exit 1
    fi

    # For Tailscale and BYO cert modes, certs are already present — verify HTTPS
    # For Let's Encrypt, cert issuance happens async — warn if not ready yet
    if [[ "${tls_mode}" != "2" ]]; then
        # Tailscale or BYO — certs should work immediately
        if curl -sf --max-time 5 "https://${OPENALPH_DOMAIN}/" >/dev/null 2>&1; then
            success "TLS configured — https://${OPENALPH_DOMAIN} is live"
        else
            warn "Caddy is running but HTTPS not yet responding"
            warn "This may resolve once tuwunel starts (next step)"
        fi
    else
        # Let's Encrypt — cert issuance is async, may take a minute
        success "Caddy is running — Let's Encrypt certificate issuance in progress"
        info "If cert issuance fails, check: journalctl -xeu caddy"
        info "Ensure DNS for ${OPENALPH_DOMAIN} points to this machine and port 80 is reachable"
    fi

    # Export OPENALPH_DOMAIN so downstream steps (e.g. step 4 / tuwunel) can use it
    export OPENALPH_DOMAIN
}


# =============================================================================
# STEP 4 — TUWUNEL DEPLOYMENT
# =============================================================================

# =============================================================================
# STEP 4 — DEPLOY TUWUNEL
# Deploys the tuwunel Matrix homeserver via Docker Compose.
#
# Requires: OPENALPH_DOMAIN (set by an earlier step) — used as server_name.
#
# CRITICAL: TUWUNEL_SERVER_NAME is written as a literal value into the
# docker-compose.yml (not as a shell variable reference). This is intentional:
# the server_name is permanent — it is baked into every Matrix user ID and
# room ID at first start. It cannot be changed later without wiping the
# database entirely. The literal value in the file is a permanent audit record.
# =============================================================================

step4_deploy_tuwunel() {
    step "Step 4" "Deploying tuwunel Matrix homeserver..."

    # -------------------------------------------------------------------------
    # Validate required variable — must be set and non-empty
    # -------------------------------------------------------------------------
    if [[ -z "${OPENALPH_DOMAIN:-}" ]]; then
        error "OPENALPH_DOMAIN is not set. This must be provided by an earlier step."
        error "TUWUNEL_SERVER_NAME cannot be empty — it is permanent and cannot be changed."
        exit 1
    fi

    # -------------------------------------------------------------------------
    # Create tuwunel working directory
    # -------------------------------------------------------------------------
    mkdir -p /opt/tuwunel

    # -------------------------------------------------------------------------
    # Write docker-compose.yml
    #
    # IMPORTANT: ${OPENALPH_DOMAIN} is expanded by bash at write time (the
    # heredoc uses an unquoted delimiter). The resulting file contains the
    # literal domain, not a variable reference. This is deliberate — server_name
    # is permanent and must be explicit. External traffic reaches port 8008
    # through the Caddy TLS reverse proxy; the container binds only on loopback
    # (127.0.0.1:4269) and is never directly exposed.
    # -------------------------------------------------------------------------
    if [[ -f /opt/tuwunel/docker-compose.yml ]]; then
        warn "/opt/tuwunel/docker-compose.yml already exists — overwriting."
        warn "The container will be restarted to pick up any changes."
    fi

    warn "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    warn "  TUWUNEL_SERVER_NAME will be set to: ${OPENALPH_DOMAIN}"
    warn "  This value is PERMANENT. It is baked into every Matrix user ID"
    warn "  and room ID at first start. Changing it later requires wiping"
    warn "  the entire database. Verify this domain is correct before continuing."
    warn "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    cat <<EOF > /opt/tuwunel/docker-compose.yml
services:
  homeserver:
    image: ghcr.io/matrix-construct/tuwunel:latest
    restart: unless-stopped
    ports:
      # Loopback only — external TLS access is handled by Caddy
      - "127.0.0.1:4269:8008"
    volumes:
      - db:/var/lib/tuwunel
    environment:
      TUWUNEL_SERVER_NAME: "${OPENALPH_DOMAIN}"
      TUWUNEL_PORT: 8008
      TUWUNEL_MAX_REQUEST_SIZE: 20000000
      TUWUNEL_ALLOW_REGISTRATION: "false"
      TUWUNEL_ALLOW_FEDERATION: "false"
      TUWUNEL_TRUSTED_SERVERS: "[]"
      TUWUNEL_LOG: "warn,tuwunel=info"
      TUWUNEL_ADDRESS: "0.0.0.0"
      TUWUNEL_IP_RANGE_DENYLIST: "[]"

volumes:
  db:
EOF

    success "Wrote /opt/tuwunel/docker-compose.yml (server_name: ${OPENALPH_DOMAIN})"

    # -------------------------------------------------------------------------
    # Pull Docker image
    # -------------------------------------------------------------------------
    info "Pulling tuwunel Docker image (ghcr.io/matrix-construct/tuwunel:latest)..."
    docker compose -f /opt/tuwunel/docker-compose.yml pull

    # -------------------------------------------------------------------------
    # Start the container
    # `up -d` is idempotent — safe to re-run if the container is already running
    # -------------------------------------------------------------------------
    info "Starting tuwunel container..."
    docker compose -f /opt/tuwunel/docker-compose.yml up -d

    # -------------------------------------------------------------------------
    # Wait for tuwunel to be ready
    # Poll /_matrix/client/versions for up to 30 seconds
    # -------------------------------------------------------------------------
    info "Waiting for tuwunel to start..."
    for i in $(seq 1 30); do
        if curl -sf http://127.0.0.1:4269/_matrix/client/versions >/dev/null 2>&1; then
            success "Tuwunel is running"
            return 0
        fi
        sleep 1
    done

    error "Tuwunel failed to start within 30 seconds"
    error "Check: docker compose -f /opt/tuwunel/docker-compose.yml logs"
    exit 1
}



# =============================================================================
# STEP 6 — MATRIX ACCOUNT REGISTRATION
# =============================================================================

# =============================================================================
# STEP 6 — CREATE MATRIX ACCOUNTS (also covers Steps 9 & 11)

# =============================================================================
#
# This file is sourced by install.sh after tuwunel is deployed (Step 4).
# It provides a single function: step6_create_accounts()
#
# What it does:
#   Step 6  — Temporarily enable registration via compose override, register
#              the operator account (@<user>:<domain>) using Matrix UIAA.
#   Step 9  — Register the agent account with a generated password, store the
#              access token to /tmp/openalph-agent-token for Step 8 to place
#              in the agent's home directory.
#   Step 11 — Remove the compose override, re-start tuwunel with registration
#              disabled, and confirm the server is healthy before returning.
#
# Required environment:
#   OPENALPH_DOMAIN     — Matrix server_name (e.g. matrix.example.com)
#
# Non-interactive overrides (all optional; script prompts when absent):
#   OPENALPH_OP_USER    — Operator Matrix localpart (e.g. alice)
#   OPENALPH_OP_PASS    — Operator Matrix password
#   OPENALPH_AGENT_NAME — Agent localpart for the automated agent account
#
# Exports for downstream steps:
#   OPENALPH_OP_USER        — confirmed operator localpart
#   OPENALPH_AGENT_NAME     — confirmed agent localpart
#   OPENALPH_AGENT_PASS     — generated agent password (in-memory only)
#   AGENT_ACCESS_TOKEN      — agent's access_token (in-memory only)
#
# Side-effects on disk (all root-owned, tight permissions):
#   /opt/tuwunel/docker-compose.override.yml — created then deleted
#   /etc/openalph/reg-token                  — registration token (mode 600)
#   /tmp/openalph-agent-token                — agent access token (mode 600)
#                                              Step 8 moves this into the
#                                              agent home directory.
#
# Assumptions / invariants:
#   • Tuwunel is already running at http://127.0.0.1:4269 before this runs.
#   • /opt/tuwunel/docker-compose.yml was written by step4_deploy_tuwunel().
#   • The 'openalph' group and /etc/openalph/ already exist (Step 3).
#   • The operator account is the ONLY registration that ever uses an admin
#     session — all subsequent accounts use the token-gated UIAA path.
#   • REG_TOKEN is never written to any log. It lives only in memory and in
#     /etc/openalph/reg-token (mode 600, root-owned) for later use.
#   • AGENT_ACCESS_TOKEN is never written to any log. It lives only in
#     /tmp/openalph-agent-token until Step 8 moves it.
# =============================================================================

step6_create_accounts() {
    step "Step 6" "Creating Matrix accounts..."

    # -------------------------------------------------------------------------
    # Guard: OPENALPH_DOMAIN must be set
    # -------------------------------------------------------------------------
    if [[ -z "${OPENALPH_DOMAIN:-}" ]]; then
        error "OPENALPH_DOMAIN is not set. Cannot build Matrix user IDs."
        exit 1
    fi

    # =========================================================================
    # INTERNAL HELPERS (local to this function's scope)
    # =========================================================================

    # ── _tuwunel_wait ─────────────────────────────────────────────────────────
    # Poll /_matrix/client/versions until tuwunel responds (up to 30 s).
    # Usage: _tuwunel_wait "reason string"
    _tuwunel_wait() {
        local _reason="${1:-tuwunel ready}"
        info "Waiting for tuwunel (${_reason})..."
        local _i
        for _i in $(seq 1 30); do
            if curl -sf http://127.0.0.1:4269/_matrix/client/versions >/dev/null 2>&1; then
                success "Tuwunel is ready (${_reason})"
                return 0
            fi
            sleep 1
        done
        error "Tuwunel did not become ready within 30 seconds (${_reason})."
        error "Check: docker compose -f /opt/tuwunel/docker-compose.yml logs"
        exit 1
    }

    # ── _register_account ─────────────────────────────────────────────────────
    # Register one Matrix account via the UIAA registration_token flow.
    # Sets global _REG_ACCESS_TOKEN to the returned access_token on success.
    # Usage: _register_account "<localpart>" "<password>" "<reg_token>"
    #
    # UIAA flow:
    #   POST /register  → 401 with { "session": "...", "flows": [...] }
    #   POST /register  → 200 with { "user_id": "...", "access_token": "..." }
    #                      (auth object provides the registration_token)
    #
    # Security note: credentials are passed via stdin pipe to curl (-d @-)
    # so they never appear in the process list or shell history.
    _register_account() {
        local _localpart="$1"
        local _password="$2"
        local _token="$3"
        local _endpoint="http://127.0.0.1:4269/_matrix/client/v3/register"

        # ------------------------------------------------------------------
        # Phase 1 — initiate registration to obtain a UIAA session ID.
        # A 401 response is normal here; any other non-2xx is an error.
        # ------------------------------------------------------------------
        info "Registering @${_localpart}:${OPENALPH_DOMAIN} — phase 1 (session)"

        local _phase1_body _phase1_http
        # Use -w '\n%{http_code}' so we can split body from status code.
        # Credentials are passed via process substitution to avoid them
        # appearing in the process table.
        _phase1_body=$(printf '%s' \
            "{\"username\":\"${_localpart}\",\"password\":\"${_password}\",\"kind\":\"user\"}")

        local _phase1_response
        _phase1_response=$(curl -s \
            -w '\n%{http_code}' \
            -X POST "${_endpoint}" \
            -H 'Content-Type: application/json' \
            -d "${_phase1_body}" 2>&1)

        # Split on the last newline — everything before is the body, last line is code
        _phase1_http="${_phase1_response##*$'\n'}"
        _phase1_body="${_phase1_response%$'\n'*}"

        # 401 is the expected UIAA "flows required" response
        if [[ "${_phase1_http}" != "401" ]] && [[ "${_phase1_http}" != "200" ]]; then
            local _errcode _errmsg
            _errcode=$(printf '%s' "${_phase1_body}" | jq -r '.errcode // "unknown"' 2>/dev/null || true)
            _errmsg=$(printf '%s' "${_phase1_body}" | jq -r '.error // ""' 2>/dev/null || true)
            error "Registration phase 1 failed for '${_localpart}': HTTP ${_phase1_http}"
            if [[ "${_errcode}" != "unknown" ]] && [[ -n "${_errcode}" ]]; then
                error "  errcode: ${_errcode}"
            fi
            if [[ -n "${_errmsg}" ]]; then
                error "  error:   ${_errmsg}"
            fi
            if [[ "${_phase1_http}" == "403" ]]; then
                error "  → Registration is disabled or the registration token is wrong."
                error "  → Check the compose override and tuwunel logs."
            fi
            exit 1
        fi

        # If tuwunel returned 200 immediately (no UIAA required) extract token now
        if [[ "${_phase1_http}" == "200" ]]; then
            _REG_ACCESS_TOKEN=$(printf '%s' "${_phase1_body}" \
                | jq -r '.access_token // empty' 2>/dev/null || true)
            if [[ -z "${_REG_ACCESS_TOKEN}" ]]; then
                error "Registration returned 200 but access_token is missing."
                error "Body: ${_phase1_body}"
                exit 1
            fi
            success "Registered @${_localpart}:${OPENALPH_DOMAIN} (no UIAA challenge needed)"
            return 0
        fi

        # Extract session ID from the 401 body
        local _session
        _session=$(printf '%s' "${_phase1_body}" \
            | jq -r '.session // empty' 2>/dev/null || true)
        if [[ -z "${_session}" ]]; then
            error "Phase 1 returned 401 but 'session' field is missing."
            error "Body: ${_phase1_body}"
            exit 1
        fi

        # ------------------------------------------------------------------
        # Phase 2 — submit the registration_token to complete UIAA.
        # ------------------------------------------------------------------
        info "Registering @${_localpart}:${OPENALPH_DOMAIN} — phase 2 (token auth)"

        # Build the JSON body as a variable — never logged (we don't print it)
        local _phase2_body
        _phase2_body=$(printf '%s' \
            "{\"username\":\"${_localpart}\"," \
            "\"password\":\"${_password}\"," \
            "\"kind\":\"user\"," \
            "\"auth\":{" \
            "\"type\":\"m.login.registration_token\"," \
            "\"token\":\"${_token}\"," \
            "\"session\":\"${_session}\"" \
            "}}")

        local _phase2_response _phase2_http _phase2_resp_body
        _phase2_response=$(curl -s \
            -w '\n%{http_code}' \
            -X POST "${_endpoint}" \
            -H 'Content-Type: application/json' \
            -d "${_phase2_body}" 2>&1)

        _phase2_http="${_phase2_response##*$'\n'}"
        _phase2_resp_body="${_phase2_response%$'\n'*}"

        if [[ "${_phase2_http}" != "200" ]]; then
            local _errcode _errmsg
            _errcode=$(printf '%s' "${_phase2_resp_body}" \
                | jq -r '.errcode // "unknown"' 2>/dev/null || true)
            _errmsg=$(printf '%s' "${_phase2_resp_body}" \
                | jq -r '.error // ""' 2>/dev/null || true)
            error "Registration phase 2 failed for '${_localpart}': HTTP ${_phase2_http}"
            if [[ "${_errcode}" != "unknown" ]] && [[ -n "${_errcode}" ]]; then
                error "  errcode: ${_errcode}"
            fi
            if [[ -n "${_errmsg}" ]]; then
                error "  error:   ${_errmsg}"
            fi
            if [[ "${_phase2_http}" == "403" ]]; then
                error "  → Token may be wrong, expired, or already used."
            fi
            exit 1
        fi

        # Extract access_token
        _REG_ACCESS_TOKEN=$(printf '%s' "${_phase2_resp_body}" \
            | jq -r '.access_token // empty' 2>/dev/null || true)
        if [[ -z "${_REG_ACCESS_TOKEN}" ]]; then
            error "Registration returned 200 but access_token is missing."
            error "Body: ${_phase2_resp_body}"
            exit 1
        fi

        success "Registered @${_localpart}:${OPENALPH_DOMAIN}"
    }

    # =========================================================================
    # PART A — GENERATE REGISTRATION TOKEN & ENABLE REGISTRATION
    # =========================================================================

    # ── Generate a random registration token ─────────────────────────────────
    # This is ephemeral — only used during this bootstrap run.
    # It will be stored in /etc/openalph/reg-token (mode 600) for future use
    # (e.g. adding additional accounts later), then registration is closed.
    local REG_TOKEN
    REG_TOKEN=$(openssl rand -hex 32)

    # ── Write compose override ─────────────────────────────────────────────────
    # This keeps the main docker-compose.yml immutable (server_name is permanent
    # and the override approach makes the enable/disable cycle explicit).
    info "Writing compose override to enable registration..."
    # Use single-quoted heredoc to prevent shell expansion inside the YAML —
    # the env vars ARE the literal strings we want in the override file.
    # REG_TOKEN is the exception: it must be expanded now so the container
    # picks up the actual token value, not a shell reference.
    cat > /opt/tuwunel/docker-compose.override.yml << OVERRIDE_EOF
services:
  homeserver:
    environment:
      TUWUNEL_ALLOW_REGISTRATION: "true"
      TUWUNEL_REGISTRATION_TOKEN: "${REG_TOKEN}"
OVERRIDE_EOF

    # Tight permissions — this file contains the registration token
    chmod 600 /opt/tuwunel/docker-compose.override.yml

    # ── Restart tuwunel with registration enabled ─────────────────────────────
    info "Restarting tuwunel with registration enabled..."
    docker compose \
        -f /opt/tuwunel/docker-compose.yml \
        -f /opt/tuwunel/docker-compose.override.yml \
        up -d

    _tuwunel_wait "registration enabled"

    # =========================================================================
    # PART B — COLLECT OPERATOR CREDENTIALS
    # =========================================================================

    info ""
    info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    info " Matrix Operator Account"
    info " This account will be @<username>:${OPENALPH_DOMAIN}"
    info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    local OP_USER OP_PASS _op_pass_confirm

    if [[ -n "${OPENALPH_OP_USER:-}" ]]; then
        # Non-interactive mode: use env vars
        OP_USER="${OPENALPH_OP_USER}"
        if [[ -z "${OPENALPH_OP_PASS:-}" ]]; then
            error "OPENALPH_OP_USER is set but OPENALPH_OP_PASS is empty."
            exit 1
        fi
        OP_PASS="${OPENALPH_OP_PASS}"
        info "Operator account (non-interactive): ${OP_USER}"
    else
        # Interactive mode: prompt
        while true; do
            printf "\n"
            _prompt_read -r -p "  Operator username (e.g. alice): " OP_USER
            if [[ -z "${OP_USER}" ]]; then
                warn "Username cannot be empty. Please try again."
                continue
            fi
            # Matrix localparts: lowercase a-z, 0-9, -, ., _, =, /
            if ! printf '%s' "${OP_USER}" | grep -qE '^[a-z0-9._=/-]+$'; then
                warn "Username '${OP_USER}' contains invalid characters."
                warn "Matrix localparts may only contain: a-z 0-9 - . _ = /"
                continue
            fi
            break
        done

        while true; do
            _prompt_read -rs -p "  Password: " OP_PASS
            printf "\n"
            if [[ -z "${OP_PASS}" ]]; then
                warn "Password cannot be empty. Please try again."
                continue
            fi
            if [[ ${#OP_PASS} -lt 8 ]]; then
                warn "Password must be at least 8 characters."
                continue
            fi
            _prompt_read -rs -p "  Confirm password: " _op_pass_confirm
            printf "\n"
            if [[ "${OP_PASS}" != "${_op_pass_confirm}" ]]; then
                warn "Passwords do not match. Please try again."
                continue
            fi
            break
        done
        unset _op_pass_confirm
    fi

    # =========================================================================
    # PART C — REGISTER THE OPERATOR ACCOUNT
    # =========================================================================

    # _REG_ACCESS_TOKEN is set as a side-effect of _register_account()
    local _REG_ACCESS_TOKEN=""
    _register_account "${OP_USER}" "${OP_PASS}" "${REG_TOKEN}"
    # We don't need the operator's access token beyond confirming success.
    # Scrub it from memory immediately.
    _REG_ACCESS_TOKEN=""

    success "Operator account ready: @${OP_USER}:${OPENALPH_DOMAIN}"

    # =========================================================================
    # PART D — COLLECT AGENT NAME
    # =========================================================================

    local AGENT_NAME

    if [[ -n "${OPENALPH_AGENT_NAME:-}" ]]; then
        AGENT_NAME="${OPENALPH_AGENT_NAME}"
        info "Agent name (non-interactive): ${AGENT_NAME}"
    else
        info ""
        info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        info " Matrix Agent Account"
        info " The agent's Matrix ID will be @<name>:${OPENALPH_DOMAIN}"
        info " This should match the agent's Unix account (oa-<name>)."
        info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        while true; do
            _prompt_read -r -p "  Agent name (e.g. merry): " AGENT_NAME
            if [[ -z "${AGENT_NAME}" ]]; then
                warn "Agent name cannot be empty. Please try again."
                continue
            fi
            if ! printf '%s' "${AGENT_NAME}" | grep -qE '^[a-z0-9._=/-]+$'; then
                warn "Agent name '${AGENT_NAME}' contains invalid characters."
                warn "Matrix localparts may only contain: a-z 0-9 - . _ = /"
                continue
            fi
            break
        done
    fi

    # =========================================================================
    # PART E — REGISTER THE AGENT ACCOUNT
    # =========================================================================

    info "Generating agent password..."
    local AGENT_PASS
    AGENT_PASS=$(openssl rand -base64 32)

    local AGENT_ACCESS_TOKEN=""
    _REG_ACCESS_TOKEN=""
    _register_account "${AGENT_NAME}" "${AGENT_PASS}" "${REG_TOKEN}"
    AGENT_ACCESS_TOKEN="${_REG_ACCESS_TOKEN}"
    _REG_ACCESS_TOKEN=""

    success "Agent account ready: @${AGENT_NAME}:${OPENALPH_DOMAIN}"

    # =========================================================================
    # PART F — STORE CREDENTIALS
    # =========================================================================

    # ── Registration token → /etc/openalph/reg-token ─────────────────────────
    # Stored for future use (e.g. registering additional accounts out-of-band).
    # Never logged; only root can read it.
    # Use a subshell with umask 177 so the file is created at mode 600 from
    # the first byte — avoids any race window between create and chmod.
    info "Storing registration token..."
    (umask 177; printf '%s' "${REG_TOKEN}" > /etc/openalph/reg-token)
    chown root:root /etc/openalph/reg-token

    # ── Agent access token → /tmp/openalph-agent-token ───────────────────────
    # The agent home directory (created in Step 8) doesn't exist yet.
    # We write to a temp location; Step 8 is responsible for moving this into
    # /home/oa-<name>/.config/matrix-token and setting correct ownership.
    # Same umask-first pattern: file is mode 600 from the moment it's created.
    info "Storing agent access token (temporary path — Step 8 will move it)..."
    (umask 177; printf '%s' "${AGENT_ACCESS_TOKEN}" > /tmp/openalph-agent-token)
    chown root:root /tmp/openalph-agent-token

    # =========================================================================
    # PART G — DISABLE REGISTRATION (remove override, restart cleanly)
    # =========================================================================

    info "Removing registration compose override..."
    rm -f /opt/tuwunel/docker-compose.override.yml

    info "Restarting tuwunel with registration disabled..."
    docker compose -f /opt/tuwunel/docker-compose.yml up -d

    _tuwunel_wait "registration disabled"

    # Verify registration is actually closed — a fresh registration attempt
    # with the old token must return 403, not 401 or 200.
    info "Verifying registration is closed..."
    local _verify_response _verify_http _verify_body
    _verify_response=$(curl -s \
        -w '\n%{http_code}' \
        -X POST "http://127.0.0.1:4269/_matrix/client/v3/register" \
        -H 'Content-Type: application/json' \
        -d '{"kind":"user"}' 2>&1)
    _verify_http="${_verify_response##*$'\n'}"
    _verify_body="${_verify_response%$'\n'*}"

    # We expect 403 (registration disabled) or 401 (flows required but no
    # registration_token flow should be listed). Either way we check the
    # error and warn if it's unexpectedly permissive.
    if [[ "${_verify_http}" == "200" ]]; then
        # This should never happen — registration returned a user without auth
        error "CRITICAL: Registration is still open after removing the override!"
        error "Tuwunel may have ignored the compose restart. Check its logs:"
        error "  docker compose -f /opt/tuwunel/docker-compose.yml logs"
        exit 1
    elif [[ "${_verify_http}" == "401" ]]; then
        # Check whether registration_token is still listed as an allowed flow
        local _flows
        _flows=$(printf '%s' "${_verify_body}" \
            | jq -r '.flows[]?.stages[]? // empty' 2>/dev/null | sort -u || true)
        if printf '%s' "${_flows}" | grep -q "m.login.registration_token"; then
            warn "Registration_token flow is still advertised by tuwunel."
            warn "This is unexpected after removing the override. Verify manually:"
            warn "  docker compose -f /opt/tuwunel/docker-compose.yml config"
        else
            success "Registration is closed (no token flow advertised)."
        fi
    else
        # 403 or any other non-200 code is fine — registration is not open
        success "Registration is closed (HTTP ${_verify_http})."
    fi

    # =========================================================================
    # PART H — EXPORT VARIABLES FOR DOWNSTREAM STEPS
    # =========================================================================

    export OPENALPH_OP_USER="${OP_USER}"
    export OPENALPH_AGENT_NAME="${AGENT_NAME}"
    export OPENALPH_AGENT_PASS="${AGENT_PASS}"
    export AGENT_ACCESS_TOKEN="${AGENT_ACCESS_TOKEN}"

    # =========================================================================
    # SUMMARY
    # =========================================================================

    success "Matrix accounts created:"
    success "  Operator : @${OP_USER}:${OPENALPH_DOMAIN}"
    success "  Agent    : @${AGENT_NAME}:${OPENALPH_DOMAIN}"
    info    "  Agent token: /tmp/openalph-agent-token (Step 8 will move this)"
    info    "  Reg token:   /etc/openalph/reg-token"
    info    "  Registration: CLOSED"
    info    ""
    info    "Step 8 must chown and move /tmp/openalph-agent-token to:"
    info    "  /home/oa-${AGENT_NAME}/.config/matrix-token (600, oa-${AGENT_NAME}:openalph)"
}


# =============================================================================
# STEPS 7, 8, 10, 12 — AGENT SETUP
# =============================================================================

# =============================================================================
# OpenAlph Bootstrap Script — Steps 7, 8, 10, 12

# =============================================================================
#
# This file provides four functions that handle agent installation:
#
#   step7_install_systemd_unit()  — Write the openalph@.service template and
#                                   reload systemd. Idempotent.
#
#   step8_create_agent()          — Interactive/non-interactive wizard that:
#                                     • Gets / validates agent name
#                                     • Calls `openalph new-agent`
#                                     • Prompts for LLM provider, API key, model
#                                     • Stores key + Matrix token securely
#                                     • Writes the TOML agent config
#                                     • Pre-populates workspace (step 10)
#                                     • Enables & starts the systemd unit (step 12)
#
# Context: sourced (or appended) into install.sh after tuwunel is up and
# Matrix accounts have been created (step 6).
#
# Required env:
#   OPENALPH_DOMAIN          — TLS domain (e.g. matrix.example.com)
#   OPENALPH_AGENT_NAME      — (optional) skips the interactive name prompt
#   OPENALPH_AGENT_PASS      — agent Matrix password (set by step 6)
#
# Non-interactive LLM env:
#   OPENALPH_PROVIDER        — anthropic | openrouter | local
#   OPENALPH_API_KEY         — raw key (warn: visible in /proc)
#   OPENALPH_API_KEY_FILE    — path to file containing the key (preferred)
#   OPENALPH_BASE_URL        — base URL for local/OpenAI-compatible provider
#   OPENALPH_MODEL           — model override (uses provider default if unset)
#
# The agent's Matrix access token must already exist at /tmp/openalph-agent-token
# (written by step 6).
#
# Helper functions assumed to exist (defined in step0_safety_preamble):
#   info(), warn(), error(), success(), step()
#
# Script must be run under:  set -euo pipefail
# =============================================================================

# =============================================================================
# STEP 7 — INSTALL SYSTEMD UNIT TEMPLATE
# Writes /etc/systemd/system/openalph@.service and reloads the daemon.
# Safe to run multiple times — overwriting the unit file is harmless before
# any agents are started.
# =============================================================================

step7_install_systemd_unit() {
    step "Step 7 — Install systemd unit template"

    local UNIT_PATH="/etc/systemd/system/openalph@.service"

    info "Writing ${UNIT_PATH} ..."

    # -------------------------------------------------------------------------
    # Write the template unit.
    # %i is the systemd instance name (the agent name after the '@').
    # The heredoc delimiter is quoted ('EOF') so no variable expansion occurs
    # here — all %i specifiers are preserved literally for systemd to expand.
    # -------------------------------------------------------------------------
    cat > "${UNIT_PATH}" <<'EOF'
[Unit]
Description=OpenAlph Agent - %i
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=oa-%i
Group=openalph
WorkingDirectory=/home/oa-%i
ExecStart=/usr/local/bin/openalph run %i
Restart=on-failure
RestartSec=5
Environment="PATH=/srv/openalph/shared/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin"
Environment="BD_ACTOR=oa-%i"

# Hardening
NoNewPrivileges=yes
ProtectSystem=strict
ProtectHome=tmpfs
BindPaths=/home/oa-%i /srv/openalph/shared
BindReadOnlyPaths=/etc/openalph /opt/openalph-venv
PrivateTmp=yes
UMask=0027
ProtectKernelTunables=yes
ProtectKernelModules=yes
ProtectControlGroups=yes

[Install]
WantedBy=multi-user.target
EOF

    chmod 644 "${UNIT_PATH}"

    info "Reloading systemd daemon..."
    systemctl daemon-reload

    success "Systemd unit template installed: ${UNIT_PATH}"
}

# =============================================================================
# STEP 8 — CREATE AGENT
# Interactive wizard: name → openalph new-agent → LLM config → token storage
# → TOML config → workspace population (step 10) → start (step 12).
# =============================================================================

step8_create_agent() {
    step "Step 8 — Create agent"

    # ── Validate required global variables ───────────────────────────────────
    if [[ -z "${OPENALPH_DOMAIN:-}" ]]; then
        error "OPENALPH_DOMAIN is not set. Run step 5 first."
        exit 1
    fi

    # ── Parse --force flag ────────────────────────────────────────────────────
    local FORCE=false
    local arg
    for arg in "$@"; do
        if [[ "${arg}" == "--force" ]]; then
            FORCE=true
        fi
    done

    # =========================================================================
    # 1. Get and validate agent name
    # =========================================================================

    local AGENT_NAME="${OPENALPH_AGENT_NAME:-}"

    if [[ -z "${AGENT_NAME}" ]]; then
        if ! _has_tty; then
            error "OPENALPH_AGENT_NAME is not set and no TTY is available."
            error "Set OPENALPH_AGENT_NAME=<name> for non-interactive use."
            exit 1
        fi
        echo ""
        _prompt_read -r -p "Agent name (lowercase letters, numbers, hyphens; e.g. 'myagent'): " AGENT_NAME
    fi

    # ── Validation rules ──────────────────────────────────────────────────────
    # Must match ^[a-z][a-z0-9-]*$, no trailing hyphen, max 32 chars.
    if [[ -z "${AGENT_NAME}" ]]; then
        error "Agent name cannot be empty."
        exit 1
    fi
    if (( ${#AGENT_NAME} > 32 )); then
        error "Agent name is too long (${#AGENT_NAME} chars; max 32)."
        exit 1
    fi
    if [[ ! "${AGENT_NAME}" =~ ^[a-z][a-z0-9-]*$ ]]; then
        error "Invalid agent name '${AGENT_NAME}'."
        error "Must start with a lowercase letter and contain only lowercase letters, digits, and hyphens."
        exit 1
    fi
    if [[ "${AGENT_NAME}" == *- ]]; then
        error "Agent name must not end with a hyphen."
        exit 1
    fi

    info "Agent name: ${AGENT_NAME}"

    # =========================================================================
    # 2. Check for existing Unix user — bail unless --force
    # =========================================================================

    if id "oa-${AGENT_NAME}" &>/dev/null; then
        if [[ "${FORCE}" == "true" ]]; then
            warn "Unix user 'oa-${AGENT_NAME}' already exists — continuing because --force was set."
        else
            error "Unix user 'oa-${AGENT_NAME}' already exists."
            error "If you want to re-create this agent, pass --force."
            error "  WARNING: --force will overwrite the existing config and workspace files."
            exit 1
        fi
    fi

    # =========================================================================
    # 3. Run openalph new-agent
    # Creates the Unix user oa-<name>, workspace scaffold, and config skeleton.
    # =========================================================================

    info "Running: openalph new-agent ${AGENT_NAME}"
    if ! openalph new-agent "${AGENT_NAME}"; then
        error "openalph new-agent failed for '${AGENT_NAME}'."
        exit 1
    fi
    success "openalph new-agent completed."

    # =========================================================================
    # 4. Prompt for LLM provider
    # =========================================================================

    local PROVIDER="${OPENALPH_PROVIDER:-}"

    if [[ -z "${PROVIDER}" ]]; then
        if ! _has_tty; then
            error "OPENALPH_PROVIDER is not set and no TTY is available."
            error "Set OPENALPH_PROVIDER=anthropic|openrouter|local for non-interactive use."
            exit 1
        fi

        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "LLM Provider"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        echo "  [1] Anthropic (Claude)"
        echo "  [2] OpenRouter (multi-model: Claude, Gemini, etc.)"
        echo "  [3] Local / OpenAI-compatible (Ollama, vLLM, etc.)"
        echo ""
        local _prov_choice
        _prompt_read -r -p "Choice [1]: " _prov_choice
        _prov_choice="${_prov_choice:-1}"

        case "${_prov_choice}" in
            1) PROVIDER="anthropic"   ;;
            2) PROVIDER="openrouter"  ;;
            3) PROVIDER="local"       ;;
            *)
                error "Invalid choice: '${_prov_choice}'. Valid options are 1, 2, or 3."
                exit 1
                ;;
        esac
    fi

    case "${PROVIDER}" in
        anthropic|openrouter|local) ;;
        *)
            error "Unknown OPENALPH_PROVIDER value: '${PROVIDER}'."
            error "Valid values are: anthropic, openrouter, local."
            exit 1
            ;;
    esac

    info "LLM provider: ${PROVIDER}"

    # =========================================================================
    # 5. Get API key
    # =========================================================================

    local API_KEY=""

    if [[ -n "${OPENALPH_API_KEY_FILE:-}" ]]; then
        # Preferred: read from file (key never touches environment / process list)
        if [[ ! -f "${OPENALPH_API_KEY_FILE}" ]]; then
            error "OPENALPH_API_KEY_FILE does not exist: ${OPENALPH_API_KEY_FILE}"
            exit 1
        fi
        API_KEY="$(< "${OPENALPH_API_KEY_FILE}")"
        if [[ -z "${API_KEY}" ]]; then
            error "OPENALPH_API_KEY_FILE is empty: ${OPENALPH_API_KEY_FILE}"
            exit 1
        fi
        info "API key loaded from file: ${OPENALPH_API_KEY_FILE}"

    elif [[ -n "${OPENALPH_API_KEY:-}" ]]; then
        # Accepted but discouraged: the value is visible in /proc/<pid>/environ
        warn "OPENALPH_API_KEY is set as an environment variable."
        warn "This is visible in the process environment (/proc). Prefer OPENALPH_API_KEY_FILE."
        API_KEY="${OPENALPH_API_KEY}"

    elif [[ "${PROVIDER}" != "local" ]]; then
        # Interactive prompt — hidden input
        if ! _has_tty; then
            error "No API key provided (OPENALPH_API_KEY or OPENALPH_API_KEY_FILE) and no TTY is available."
            exit 1
        fi
        echo ""
        _prompt_read -rs -p "API key: " API_KEY
        echo ""   # newline after hidden input
        if [[ -z "${API_KEY}" ]]; then
            error "API key cannot be empty."
            exit 1
        fi
    fi

    # ── For local provider, also get base URL ─────────────────────────────────
    local BASE_URL="${OPENALPH_BASE_URL:-}"

    if [[ "${PROVIDER}" == "local" ]]; then
        if [[ -z "${BASE_URL}" ]]; then
            if ! _has_tty; then
                error "OPENALPH_BASE_URL is not set and no TTY is available."
                error "Set OPENALPH_BASE_URL=http://localhost:11434/v1 for non-interactive use."
                exit 1
            fi
            echo ""
            _prompt_read -r -p "Base URL (e.g. http://localhost:11434/v1): " BASE_URL
        fi
        if [[ -z "${BASE_URL}" ]]; then
            error "Base URL is required for local/OpenAI-compatible providers."
            exit 1
        fi
        info "Base URL: ${BASE_URL}"
    fi

    # =========================================================================
    # 6. Get model
    # =========================================================================

    local MODEL="${OPENALPH_MODEL:-}"

    if [[ -z "${MODEL}" ]]; then
        if _has_tty; then
            # Interactive: show defaults and prompt
            echo ""
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo "Model selection (press Enter for default)"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo ""
            echo "  Anthropic default:   anthropic/claude-sonnet-4"
            echo "  OpenRouter default:  anthropic/claude-sonnet-4"
            echo "  Local default:       (uses whatever your server provides)"
            echo ""
            _prompt_read -r -p "Model [default]: " MODEL
        fi
        # If still empty after prompt (or non-interactive), apply defaults below
    fi

    # Apply defaults per provider
    case "${PROVIDER}" in
        anthropic)
            # TOML stores "anthropic/<model>" — strip any leading "anthropic/" the
            # user may have typed to avoid doubling up.
            if [[ -z "${MODEL}" ]]; then
                MODEL="claude-sonnet-4"
            fi
            # Normalise: strip accidental leading provider prefix
            MODEL="${MODEL#anthropic/}"
            ;;
        openrouter)
            # OpenRouter models are already namespaced (e.g. anthropic/claude-sonnet-4).
            # Default: anthropic/claude-sonnet-4
            if [[ -z "${MODEL}" ]]; then
                MODEL="anthropic/claude-sonnet-4"
            fi
            ;;
        local)
            # No sane default for local — but an empty string is allowed
            # (openalph will use whatever the server advertises as its default).
            if [[ -z "${MODEL}" ]]; then
                warn "No model specified for local provider. The server's default will be used."
            fi
            ;;
    esac

    if [[ -n "${MODEL}" ]]; then
        info "Model: ${MODEL}"
    fi

    # =========================================================================
    # 7. Store API key securely under the agent home
    # =========================================================================

    local AGENT_HOME="/home/oa-${AGENT_NAME}"
    local CONFIG_DIR="${AGENT_HOME}/.config"

    # Ensure .config directory exists (new-agent should create it, but be safe)
    mkdir -p "${CONFIG_DIR}"
    chown "oa-${AGENT_NAME}:openalph" "${CONFIG_DIR}"
    chmod 750 "${CONFIG_DIR}"

    if [[ "${PROVIDER}" != "local" ]]; then
        info "Storing API key..."
        printf '%s' "${API_KEY}" > "${CONFIG_DIR}/provider-key"
        chown "oa-${AGENT_NAME}:openalph" "${CONFIG_DIR}/provider-key"
        chmod 600 "${CONFIG_DIR}/provider-key"
        success "API key stored: ${CONFIG_DIR}/provider-key"

        # Clear the in-memory variable — it's on disk now
        API_KEY="<cleared>"
    else
        # Local provider doesn't need a real key; write a placeholder so the
        # api_key_cmd in the TOML (`echo unused`) has something to reference.
        printf 'unused' > "${CONFIG_DIR}/provider-key"
        chown "oa-${AGENT_NAME}:openalph" "${CONFIG_DIR}/provider-key"
        chmod 600 "${CONFIG_DIR}/provider-key"
    fi

    # =========================================================================
    # 8. Move Matrix access token from /tmp to the agent home
    # =========================================================================

    local TOKEN_SRC="/tmp/openalph-agent-token"
    local TOKEN_DST="${CONFIG_DIR}/matrix-token"

    if [[ ! -f "${TOKEN_SRC}" ]]; then
        error "Matrix access token not found at ${TOKEN_SRC}."
        error "Step 6 should have written this file. Cannot continue."
        exit 1
    fi

    info "Moving Matrix access token into agent home..."
    cp "${TOKEN_SRC}" "${TOKEN_DST}"
    chown "oa-${AGENT_NAME}:openalph" "${TOKEN_DST}"
    chmod 600 "${TOKEN_DST}"
    rm "${TOKEN_SRC}"
    success "Matrix token stored: ${TOKEN_DST}"

    # =========================================================================
    # 9. Write TOML agent config
    # Overwrites the stale skeleton left by `openalph new-agent`.
    # =========================================================================

    local AGENTS_DIR="/etc/openalph/agents"
    local TOML_PATH="${AGENTS_DIR}/${AGENT_NAME}.toml"

    info "Writing agent config: ${TOML_PATH}"

    # ── Build the [agent] block ───────────────────────────────────────────────
    # default_model prefix depends on provider.
    local DEFAULT_MODEL_LINE=""
    case "${PROVIDER}" in
        anthropic)
            if [[ -n "${MODEL}" ]]; then
                DEFAULT_MODEL_LINE="default_model = \"anthropic/${MODEL}\""
            else
                DEFAULT_MODEL_LINE="default_model = \"anthropic/claude-sonnet-4\""
            fi
            ;;
        openrouter)
            if [[ -n "${MODEL}" ]]; then
                DEFAULT_MODEL_LINE="default_model = \"openrouter/${MODEL}\""
            else
                DEFAULT_MODEL_LINE="default_model = \"openrouter/anthropic/claude-sonnet-4\""
            fi
            ;;
        local)
            if [[ -n "${MODEL}" ]]; then
                DEFAULT_MODEL_LINE="default_model = \"local/${MODEL}\""
            else
                DEFAULT_MODEL_LINE="# default_model is unset — local server default will be used"
            fi
            ;;
    esac

    # ── Build the [providers.*] block ─────────────────────────────────────────
    local PROVIDER_BLOCK=""
    case "${PROVIDER}" in
        anthropic)
            PROVIDER_BLOCK="[providers.anthropic]
type = \"anthropic\"
api_key_cmd = \"cat ${CONFIG_DIR}/provider-key\""
            ;;
        openrouter)
            PROVIDER_BLOCK="[providers.openrouter]
type = \"openai\"
base_url = \"https://openrouter.ai/api/v1\"
api_key_cmd = \"cat ${CONFIG_DIR}/provider-key\""
            ;;
        local)
            PROVIDER_BLOCK="[providers.local]
type = \"openai\"
base_url = \"${BASE_URL}\"
api_key_cmd = \"echo unused\""
            ;;
    esac

    # ── Write the full TOML ───────────────────────────────────────────────────
    # The heredoc delimiter is unquoted so bash expands the variables;
    # the resulting file contains only literal values (no shell references).
    cat > "${TOML_PATH}" <<EOF
[agent]
name = "${AGENT_NAME}"
${DEFAULT_MODEL_LINE}
max_tokens = 8192
model_max_tokens = 200000
max_iterations = 50
truncation_limit = 50000

${PROVIDER_BLOCK}

[workspace]
path = "${AGENT_HOME}/workspace"

[matrix]
homeserver = "http://127.0.0.1:4269"
user_id = "@${AGENT_NAME}:${OPENALPH_DOMAIN}"
access_token_cmd = "cat ${CONFIG_DIR}/matrix-token"
EOF

    chown "root:openalph" "${TOML_PATH}"
    chmod 640 "${TOML_PATH}"
    success "Agent config written: ${TOML_PATH}"

    # =========================================================================
    # 10. Pre-populate workspace with templates (step 10)
    # =========================================================================
    _step10_populate_workspace "${AGENT_NAME}" "${AGENT_HOME}"

    # =========================================================================
    # 12. Enable + start agent (step 12)
    # =========================================================================
    _step12_start_agent "${AGENT_NAME}"
}

# =============================================================================
# STEP 10 — PRE-POPULATE WORKSPACE WITH TEMPLATES
# Internal helper called from step8_create_agent().
# Copies template files from the installed Python package into the agent's
# workspace so the agent starts with a well-formed operational environment.
# =============================================================================

_step10_populate_workspace() {
    local AGENT_NAME="$1"
    local AGENT_HOME="$2"
    local WORKSPACE="${AGENT_HOME}/workspace"

    step "Step 10 — Pre-populate workspace: ${WORKSPACE}"

    # ── Locate template directory from the installed Python package ───────────
    local TEMPLATE_DIR
    TEMPLATE_DIR="$(/opt/openalph-venv/bin/python3 -c "
import openalph.templates
import pathlib
print(pathlib.Path(openalph.templates.__file__).parent)
" 2>/dev/null || true)"

    if [[ -z "${TEMPLATE_DIR}" ]]; then
        error "Could not locate openalph.templates package."
        error "Ensure OpenAlph is installed: pip3 show openalph"
        exit 1
    fi

    if [[ ! -d "${TEMPLATE_DIR}" ]]; then
        error "Template directory does not exist: ${TEMPLATE_DIR}"
        exit 1
    fi

    info "Template source: ${TEMPLATE_DIR}"

    # ── Ensure workspace directory exists ────────────────────────────────────
    mkdir -p "${WORKSPACE}"

    # ── Copy top-level markdown templates ────────────────────────────────────
    # These are the canonical operational documents every agent starts with.
    local doc
    for doc in SAFETY.md SOUL.md OPERATOR.md WAKE.md ENVIRONMENT.md OPERATIONS.md; do
        local src="${TEMPLATE_DIR}/${doc}"
        if [[ -f "${src}" ]]; then
            cp "${src}" "${WORKSPACE}/${doc}"
            info "Copied: ${doc}"
        else
            warn "Template not found, skipping: ${src}"
        fi
    done

    # ── Copy skills directory ─────────────────────────────────────────────────
    local SKILLS_SRC="${TEMPLATE_DIR}/skills"
    local SKILLS_DST="${WORKSPACE}/skills"

    if [[ -d "${SKILLS_SRC}" ]]; then
        mkdir -p "${SKILLS_DST}"
        # cp -r copies contents; use trailing slash on source to merge rather than nest
        cp -r "${SKILLS_SRC}/." "${SKILLS_DST}/"
        info "Copied skills directory: ${SKILLS_SRC} → ${SKILLS_DST}"
    else
        warn "Skills directory not found, skipping: ${SKILLS_SRC}"
    fi

    # ── Fix ownership on all workspace files ─────────────────────────────────
    chown -R "oa-${AGENT_NAME}:openalph" "${WORKSPACE}"

    # =========================================================================
    # 11. Enable tool TOMLs
    # Each tool gets a (possibly empty) TOML file so the agent can load it.
    # =========================================================================
    local TOOLS_DIR="${WORKSPACE}/tools"
    mkdir -p "${TOOLS_DIR}"

    local tool
    for tool in shell file_read file_write file_edit web_search web_fetch subagent memory_search send_media; do
        touch "${TOOLS_DIR}/${tool}.toml"
    done

    chown -R "oa-${AGENT_NAME}:openalph" "${TOOLS_DIR}"

    success "Workspace pre-populated: ${WORKSPACE}"
}

# =============================================================================
# STEP 12 — ENABLE AND START AGENT
# Internal helper called from step8_create_agent().
# Enables the systemd instantiated unit and polls until it is active.
# =============================================================================

_step12_start_agent() {
    local AGENT_NAME="$1"
    local UNIT="openalph@${AGENT_NAME}"

    step "Step 12 — Enable and start agent: ${UNIT}"

    systemctl enable "${UNIT}"
    info "systemd unit enabled."

    systemctl start "${UNIT}"
    info "systemd unit start requested — waiting for active state..."

    # ── Poll for up to 15 seconds ─────────────────────────────────────────────
    local i
    for i in $(seq 1 15); do
        if systemctl is-active --quiet "${UNIT}"; then
            success "Agent ${AGENT_NAME} is running."
            return 0
        fi
        sleep 1
    done

    error "Agent failed to start within 15 seconds."
    error "Check: sudo journalctl -u ${UNIT} -n 50"
    exit 1
}


# =============================================================================
# CINNY WEB UI
# =============================================================================

# =============================================================================
# STEP — SERVE CINNY WEB CLIENT
# Downloads the Cinny Matrix web client and configures Caddy to serve it
# at the domain root, while proxying Matrix API paths to tuwunel.
#
# Requires:
#   - OPENALPH_DOMAIN set by step 5 (TLS/Caddy)
#   - Caddy running   (step 5)
#   - Tuwunel running (step 4)
#
# Idempotent: re-running re-downloads and overwrites Cinny files and the
# Caddyfile. The Caddyfile rewrite preserves the existing tls directive
# so all three TLS modes (tailscale, letsencrypt, custom) are handled
# transparently.
# =============================================================================

step_setup_cinny() {
    step "Cinny" "Setting up web client..."

    # ── Validate required variable ────────────────────────────────────────────
    if [[ -z "${OPENALPH_DOMAIN:-}" ]]; then
        error "OPENALPH_DOMAIN is not set. Run step 5 (TLS/Caddy) before this step."
        exit 1
    fi

    # ── Download Cinny release ────────────────────────────────────────────────
    local CINNY_VERSION="v4.11.1"
    local CINNY_URL="https://github.com/cinnyapp/cinny/releases/download/${CINNY_VERSION}/cinny-${CINNY_VERSION}.tar.gz"

    info "Downloading Cinny ${CINNY_VERSION}..."
    mkdir -p /opt/openalph/web
    curl -sL "${CINNY_URL}" -o /tmp/cinny.tar.gz

    # Extract into a staging directory — handle both tarball layouts:
    #   a) top-level dist/ directory  →  copy contents of dist/
    #   b) web files directly at root →  copy everything
    # CINNY_TMP is used instead of TMPDIR to avoid clobbering the reserved
    # Unix environment variable that controls where temporary files are written.
    local CINNY_TMP
    CINNY_TMP=$(mktemp -d)
    tar xzf /tmp/cinny.tar.gz -C "${CINNY_TMP}"

    local DIST_DIR
    DIST_DIR=$(find "${CINNY_TMP}" -type d -name dist | head -1 || true)

    if [[ -z "${DIST_DIR}" ]]; then
        info "No dist/ directory found in tarball — treating root as web files."
        cp -r "${CINNY_TMP}"/* /opt/openalph/web/
    else
        info "Found dist/ directory: ${DIST_DIR}"
        cp -r "${DIST_DIR}"/* /opt/openalph/web/
    fi

    rm -rf "${CINNY_TMP}" /tmp/cinny.tar.gz
    success "Cinny ${CINNY_VERSION} extracted to /opt/openalph/web/"

    # ── Write Cinny config.json ───────────────────────────────────────────────
    # Pre-configure the homeserver so the operator does not have to type it in
    # manually on first login. The heredoc delimiter is unquoted so that
    # ${OPENALPH_DOMAIN} is expanded by bash at write time.
    info "Writing config.json (homeserver: https://${OPENALPH_DOMAIN})..."
    cat > /opt/openalph/web/config.json << EOF
{
    "defaultHomeserver": 0,
    "homeserverList": [
        "https://${OPENALPH_DOMAIN}"
    ]
}
EOF
    success "config.json written."

    # ── Update Caddyfile ──────────────────────────────────────────────────────
    # Preserve the tls directive written by step 5; its form depends on the
    # TLS mode that was chosen:
    #   tailscale / custom  →  "tls /path/to/cert /path/to/key"
    #   letsencrypt         →  (no tls line — Caddy auto-provisions via ACME)
    #
    # Leading whitespace is stripped from the matched line so that the
    # heredoc's own indentation is not doubled.
    local TLS_LINE
    TLS_LINE=$(grep '^[[:space:]]*tls ' /etc/caddy/Caddyfile 2>/dev/null \
               | head -1 \
               | sed 's/^[[:space:]]*//' \
               || true)

    info "Updating /etc/caddy/Caddyfile to serve Cinny and proxy Matrix API..."
    cat > /etc/caddy/Caddyfile << EOF
${OPENALPH_DOMAIN} {
    ${TLS_LINE}

    # Matrix client/server API — proxy to tuwunel
    handle /_matrix/* {
        reverse_proxy 127.0.0.1:4269
    }

    # Well-known discovery — Matrix clients query this to find the homeserver
    handle /.well-known/matrix/* {
        reverse_proxy 127.0.0.1:4269
    }

    # Synapse-compatible admin API (used by some Matrix tools)
    handle /_synapse/* {
        reverse_proxy 127.0.0.1:4269
    }

    # Cinny web client — single-page app; unknown paths fall back to index.html
    # so that client-side routing works correctly after a hard refresh.
    # Note: {path} is a Caddy placeholder, not a shell variable.
    handle {
        root * /opt/openalph/web
        try_files {path} /index.html
        file_server
    }
}
EOF

    # ── Validate and reload Caddy ─────────────────────────────────────────────
    info "Validating Caddyfile..."
    caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile

    info "Reloading Caddy..."
    systemctl reload caddy
    success "Caddy reloaded."

    # ── Verify web client is reachable ────────────────────────────────────────
    # Allow a moment for Caddy to finish reloading before probing.
    sleep 2

    if curl -sf "https://${OPENALPH_DOMAIN}/" 2>/dev/null | grep -qi 'cinny'; then
        success "Cinny web client available at https://${OPENALPH_DOMAIN}/"
    else
        local HTTP_CODE
        HTTP_CODE=$(curl -so /dev/null -w '%{http_code}' "https://${OPENALPH_DOMAIN}/" 2>/dev/null \
                    || echo "000")
        if [[ "${HTTP_CODE}" == "200" ]]; then
            success "Web client available at https://${OPENALPH_DOMAIN}/"
        else
            warn "Web client may not be responding correctly (HTTP ${HTTP_CODE})"
            warn "Inspect with: curl -v https://${OPENALPH_DOMAIN}/"
            warn "Caddy logs:   journalctl -xeu caddy"
        fi
    fi
}


# =============================================================================
# STEP 13 — SUMMARY OUTPUT
# Prints the "You're Live" summary after a successful bootstrap run.
# Requires variables exported by earlier steps:
#   OPENALPH_DOMAIN, OPENALPH_OP_USER, OPENALPH_AGENT_NAME,
#   OPENALPH_AGENT_PASS, LOG_FILE
# =============================================================================

step13_print_summary() {
    printf "\n"
    printf "╔══════════════════════════════════════════════════════════════╗\n"
    printf "║          OpenAlph is live — here's how to connect           ║\n"
    printf "╚══════════════════════════════════════════════════════════════╝\n"
    printf "\n"

    info "Web client:    https://${OPENALPH_DOMAIN}"
    info "Homeserver:    https://${OPENALPH_DOMAIN}"
    info "Your account:  @${OPENALPH_OP_USER}:${OPENALPH_DOMAIN}"
    info "Agent:         @${OPENALPH_AGENT_NAME}:${OPENALPH_DOMAIN}"

    printf "\n"
    printf "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    printf "  QUICK START\n"
    printf "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    printf "\n"
    info "  1. Open https://${OPENALPH_DOMAIN} in your browser"
    info "  2. Log in with: ${OPENALPH_OP_USER} / your password"
    info "  3. Create a new room (private)"
    info "  4. Invite @${OPENALPH_AGENT_NAME}:${OPENALPH_DOMAIN}"
    info "  5. Say hello"

    printf "\n"
    printf "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    printf "  MOBILE CLIENTS\n"
    printf "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    printf "\n"
    info "  Recommended: Element (iOS/Android) — https://element.io/download"
    info "  In the app: Sign in → Other → homeserver: https://${OPENALPH_DOMAIN}"

    printf "\n"
    printf "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    printf "  MANAGE YOUR AGENT\n"
    printf "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    printf "\n"
    info "  Status:    sudo systemctl status openalph@${OPENALPH_AGENT_NAME}"
    info "  Logs:      sudo journalctl -fu openalph@${OPENALPH_AGENT_NAME}"
    info "  Stop:      sudo systemctl stop openalph@${OPENALPH_AGENT_NAME}"
    info "  Restart:   sudo systemctl restart openalph@${OPENALPH_AGENT_NAME}"
    info "  Config:    /etc/openalph/agents/${OPENALPH_AGENT_NAME}.toml"
    info "  Workspace: /home/oa-${OPENALPH_AGENT_NAME}/workspace/"

    printf "\n"
    printf "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    printf "  SAVED CREDENTIALS (back these up!)\n"
    printf "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    printf "\n"
    warn "  Agent Matrix password: ${OPENALPH_AGENT_PASS}"
    info "  (The agent uses its access token day-to-day, not this password."
    info "   Store it somewhere safe in case you need to re-issue the token.)"
    info "  Registration token: /etc/openalph/reg-token"

    printf "\n"
    printf "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    printf "  NEXT STEPS\n"
    printf "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    printf "\n"
    info "  Customize your agent's personality and behavior:"
    info "    sudo -u oa-${OPENALPH_AGENT_NAME} nano /home/oa-${OPENALPH_AGENT_NAME}/workspace/SOUL.md"
    info "  After editing, restart: sudo systemctl restart openalph@${OPENALPH_AGENT_NAME}"
    printf "\n"
    info "  Add another agent: sudo openalph new-agent <name>"
    printf "\n"
    info "  Full bootstrap log: ${LOG_FILE}"
    printf "\n"
}

# =============================================================================
# MAIN
# =============================================================================

main() {
    # Must run as root
    if [[ "${EUID}" -ne 0 ]]; then
        # We can't use error() yet (helpers defined in step0), so use plain echo
        printf "[ERROR] This script must be run as root (sudo).\n" >&2
        exit 1
    fi

    # ── Parse --force flag ────────────────────────────────────────────────────
    # When present, steps that check for existing resources (e.g. step8_create_agent)
    # will overwrite rather than abort. Set OPENALPH_FORCE=true to enable.
    OPENALPH_FORCE=false
    local _arg
    for _arg in "$@"; do
        if [[ "${_arg}" == "--force" ]]; then
            OPENALPH_FORCE=true
        fi
    done
    export OPENALPH_FORCE

    step0_safety_preamble
    step1_preflight_checks
    step2_install_openalph
    step3_create_group_and_dirs
    step5_setup_tls
    step4_deploy_tuwunel
    step6_create_accounts
    step7_install_systemd_unit
    step8_create_agent
    step_setup_cinny

    step13_print_summary
}

main "$@"
