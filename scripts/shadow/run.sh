#!/usr/bin/env bash
# Shadow validator: runs the affine loop against finney, provisions H200
# slots on Targon, evaluates challengers, and records everything — but
# never submits weights to chain.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

VENV="$PROJECT_DIR/.shadow-venv"
SHADOW_DIR="$PROJECT_DIR/.shadow"
mkdir -p "$SHADOW_DIR"

log() { printf '\033[1;34m==> %s\033[0m\n' "$*"; }
die() { printf '\033[1;31mFATAL: %s\033[0m\n' "$*" >&2; exit 1; }

# --- load TARGON_API_KEY from .env ---
[ -f "$PROJECT_DIR/.env" ] || die ".env not found at $PROJECT_DIR/.env"
set -a
# shellcheck disable=SC1091
source "$PROJECT_DIR/.env"
set +a
[ -n "${TARGON_API_KEY:-}" ] || die "TARGON_API_KEY missing from .env"

# --- build/refresh venv ---
if [ ! -x "$VENV/bin/python" ]; then
    log "creating venv at $VENV"
    python3 -m venv "$VENV"
fi

"$VENV/bin/pip" install --upgrade pip >/dev/null

log "installing affine + deps into $VENV"
"$VENV/bin/pip" install -e "$PROJECT_DIR" >/dev/null

# The Riot `targon` PyPI package shares the `targon/` import path and
# silently clobbers the real SDK's __init__. Force reinstall the SDK
# last so its files win.
"$VENV/bin/pip" uninstall -y targon 2>/dev/null || true
"$VENV/bin/pip" install --force-reinstall --no-deps targon-sdk >/dev/null

# --- sanity check imports ---
"$VENV/bin/python" - <<'PY' || exit 1
from targon import Client, App, Image, Compute
assert Compute.H200_SMALL == "h200-small", Compute.H200_SMALL
print("targon-sdk OK; H200_SMALL =", Compute.H200_SMALL)
PY

# --- env for the shadow run ---
export AFFINE_DRY_RUN=1
export LOG_LEVEL="${LOG_LEVEL:-DEBUG}"
export NETUID="${NETUID:-120}"
export SUBTENSOR_ENDPOINT="${SUBTENSOR_ENDPOINT:-finney}"
export BT_WALLET_COLD="${BT_WALLET_COLD:-default}"
export BT_WALLET_HOT="${BT_WALLET_HOT:-default}"
export AFFINE_CONFIG_SPEC="${AFFINE_CONFIG_SPEC:-$SHADOW_DIR/config.json}"
export AFFINE_PROVISION_TIMEOUT="${AFFINE_PROVISION_TIMEOUT:-1200}"

TS="$(date -u +%Y%m%dT%H%M%SZ)"
export AFFINE_SHADOW_LOG="$SHADOW_DIR/duels-$TS.jsonl"
STDERR_LOG="$SHADOW_DIR/stderr-$TS.log"

log "shadow JSONL:  $AFFINE_SHADOW_LOG"
log "shadow stderr: $STDERR_LOG"
log "config: NETUID=$NETUID SUBTENSOR=$SUBTENSOR_ENDPOINT SPEC=$AFFINE_CONFIG_SPEC wallet=$BT_WALLET_COLD/$BT_WALLET_HOT"

# --- launch ---
# Prepend venv bin so the `targon` subprocess invoked by TargonSlots resolves
# to the venv's targon-sdk CLI, not the broken system /usr/bin copy.
export PATH="$VENV/bin:$PATH"
exec "$VENV/bin/affine" 2>&1 | tee "$STDERR_LOG"
