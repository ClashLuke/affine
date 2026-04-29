#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

# --- defaults (override via env) ---
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
SUBTENSOR_IMAGE="${SUBTENSOR_IMAGE:-ghcr.io/opentensor/subtensor-localnet:devnet-ready}"
SUBTENSOR_CONTAINER="${SUBTENSOR_CONTAINER:-subtensor-dev}"
VLLM_PORT_A="${VLLM_PORT_A:-8000}"
VLLM_PORT_B="${VLLM_PORT_B:-8001}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.28}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
GATE_STAGE="${GATE_STAGE:-all}"
SMOKE_TIMEOUT="${SMOKE_TIMEOUT:-2400}"
FULL_TIMEOUT="${FULL_TIMEOUT:-7200}"
DETHRONE_TIMEOUT="${DETHRONE_TIMEOUT:-900}"
SKIP_TESTS="${SKIP_TESTS:-}"
SKIP_PULL="${SKIP_PULL:-}"

log() { printf '\033[1;34m==> %s\033[0m\n' "$*"; }
die() { printf '\033[1;31mFATAL: %s\033[0m\n' "$*" >&2; exit 1; }

# --- stage selection ---
declare -a STAGES
case "$GATE_STAGE" in
    all)      STAGES=(dethrone smoke) ;;
    dethrone) STAGES=(dethrone) ;;
    smoke)    STAGES=(smoke) ;;
    full)     STAGES=(full) ;;
    both)     STAGES=(smoke full) ;;
    *)        die "unknown GATE_STAGE=$GATE_STAGE (valid: all, dethrone, smoke, full, both)" ;;
esac

# --- process management ---
PORT_A_PID=""
VLLM_PID_B=""

stop_proc() {
    local pid="$1" name="$2"
    [ -z "$pid" ] && return
    local children
    children=$(pgrep -P "$pid" 2>/dev/null || true)
    kill -0 "$pid" 2>/dev/null || {
        for cpid in $children; do
            kill -9 "$cpid" 2>/dev/null || true
        done
        [ -n "$children" ] && log "$name orphan children killed"
        return
    }
    log "stopping $name (pid $pid)"
    kill -INT "$pid" 2>/dev/null
    for _ in $(seq 1 15); do
        kill -0 "$pid" 2>/dev/null || { log "$name stopped"; return; }
        sleep 1
    done
    log "$name did not exit after SIGINT, sending SIGKILL"
    kill -9 "$pid" 2>/dev/null
    for cpid in $(pgrep -P "$pid" 2>/dev/null || true); do
        kill -9 "$cpid" 2>/dev/null || true
    done
    wait "$pid" 2>/dev/null || true
}

stop_port_a() {
    if [ -n "$PORT_A_PID" ]; then
        stop_proc "$PORT_A_PID" "port-$VLLM_PORT_A"
        PORT_A_PID=""
    fi
}

cleanup() {
    log "cleaning up"
    stop_port_a
    stop_proc "${VLLM_PID_B:-}" "vLLM-B"
    # vLLM spawns child processes (EngineCore) that survive parent death and
    # get reparented to init, leaking GPU memory. Kill anything still on our
    # ports as a safety net.
    for port in "$VLLM_PORT_A" "$VLLM_PORT_B"; do
        local pids
        pids=$(lsof -i :"$port" -t 2>/dev/null || true)
        if [ -n "$pids" ]; then
            echo "$pids" | xargs kill -9 2>/dev/null || true
            log "killed leaked processes on port $port"
        fi
    done
    docker rm -f "$SUBTENSOR_CONTAINER" 2>/dev/null || true
}
trap cleanup EXIT

wait_healthy() {
    local url="$1" name="$2" timeout="${3:-120}"
    for i in $(seq 1 "$timeout"); do
        if curl -sf "$url/models" >/dev/null 2>&1; then
            log "$name healthy after ${i}s"
            return 0
        fi
        sleep 1
    done
    die "$name not healthy after ${timeout}s"
}

start_stub_on_a() {
    log "starting stub vLLM on port $VLLM_PORT_A"
    python3 scripts/e2e/stub_vllm.py \
        --port "$VLLM_PORT_A" \
        --model "$MODEL" \
        >"$PROJECT_DIR/.e2e/stub.log" 2>&1 &
    PORT_A_PID=$!
    wait_healthy "http://localhost:$VLLM_PORT_A/v1" "stub-vllm" 10

    curl -sf "http://localhost:$VLLM_PORT_A/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}" \
        | python3 -c "import sys,json; d=json.load(sys.stdin); assert d['choices'][0]['message']['content']" \
        || die "stub server returning invalid chat completions"
    log "stub sanity check passed"
}

start_vllm_on_a() {
    log "starting vLLM-A on port $VLLM_PORT_A"
    vllm serve "$MODEL" \
        --port "$VLLM_PORT_A" \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        --max-model-len "$MAX_MODEL_LEN" \
        >"$PROJECT_DIR/.e2e/vllm_a.log" 2>&1 &
    PORT_A_PID=$!
    wait_healthy "http://localhost:$VLLM_PORT_A/v1" "vLLM-A" 120
}

# ------------------------------------------------------------------
# 1. unit tests
# ------------------------------------------------------------------
if [ -z "$SKIP_TESTS" ]; then
    log "running unit tests"
    python3 -m pytest tests/ -v || die "unit tests failed"
fi

# ------------------------------------------------------------------
# 2. pull docker images
# ------------------------------------------------------------------
if [ -z "$SKIP_PULL" ]; then
    log "pulling subtensor image"
    docker pull "$SUBTENSOR_IMAGE"
fi

# ------------------------------------------------------------------
# 3. start subtensor dev chain
# ------------------------------------------------------------------
log "starting subtensor dev chain"
docker rm -f "$SUBTENSOR_CONTAINER" 2>/dev/null || true
docker run -d --name "$SUBTENSOR_CONTAINER" \
    -p 9944:9944 -p 9945:9945 \
    "$SUBTENSOR_IMAGE" True

log "waiting for block production (up to 60s)"
for i in $(seq 1 60); do
    if docker logs "$SUBTENSOR_CONTAINER" 2>&1 | grep "Imported #1" >/dev/null; then
        log "subtensor producing blocks (${i}s)"
        break
    fi
    if [ "$i" -eq 60 ]; then
        docker logs "$SUBTENSOR_CONTAINER" 2>&1 | tail -20
        die "subtensor did not produce blocks within 60s"
    fi
    sleep 1
done

# ------------------------------------------------------------------
# 4. start vLLM-B (stays up for all stages)
# ------------------------------------------------------------------
mkdir -p "$PROJECT_DIR/.e2e"

for port in "$VLLM_PORT_A" "$VLLM_PORT_B"; do
    if lsof -i :"$port" -t >/dev/null 2>&1; then
        die "port $port already in use — kill the existing process first"
    fi
done

log "starting vLLM-B on port $VLLM_PORT_B (shared across stages)"
vllm serve "$MODEL" \
    --port "$VLLM_PORT_B" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    >"$PROJECT_DIR/.e2e/vllm_b.log" 2>&1 &
VLLM_PID_B=$!

wait_healthy "http://localhost:$VLLM_PORT_B/v1" "vLLM-B" 120

# ------------------------------------------------------------------
# 5. bootstrap dev chain state
# ------------------------------------------------------------------
log "bootstrapping dev chain (wallets, subnet, registrations, commitments)"
python3 scripts/e2e/bootstrap_dev_chain.py \
    --endpoint ws://127.0.0.1:9944 \
    --model "$MODEL" \
    --revision-a stub \
    --revision-b real

# ------------------------------------------------------------------
# 6. read bootstrap output and export env
# ------------------------------------------------------------------
STATE_FILE="$PROJECT_DIR/.e2e/dev_chain_state.json"
[ -f "$STATE_FILE" ] || die "bootstrap did not produce $STATE_FILE"

export SUBTENSOR_ENDPOINT
SUBTENSOR_ENDPOINT=$(python3 -c "import json; print(json.load(open('$STATE_FILE'))['endpoint'])")
export NETUID
NETUID=$(python3 -c "import json; print(json.load(open('$STATE_FILE'))['netuid'])")
export BT_WALLET_COLD
BT_WALLET_COLD=$(python3 -c "import json; print(json.load(open('$STATE_FILE'))['validator_wallet'])")
export BT_WALLET_HOT
BT_WALLET_HOT=$(python3 -c "import json; print(json.load(open('$STATE_FILE'))['validator_hotkey'])")
export BT_WALLET_PATH
BT_WALLET_PATH=$(python3 -c "import json; print(json.load(open('$STATE_FILE'))['wallet_path'])")
export AFFINE_LOCAL=1
export CHAMPION_URL="http://localhost:$VLLM_PORT_A/v1"
export CHALLENGER_URL="http://localhost:$VLLM_PORT_B/v1"
export LOG_LEVEL=DEBUG
export AFFINE_DB_PATH="$PROJECT_DIR/.e2e/affine.sqlite3"
export AFFINE_BASELINE_MODEL="$MODEL"
export AFFINE_BASELINE_REVISION="stub"

rm -f "$AFFINE_DB_PATH" "$AFFINE_DB_PATH-wal" "$AFFINE_DB_PATH-shm"

log "env: SUBTENSOR_ENDPOINT=$SUBTENSOR_ENDPOINT NETUID=$NETUID"
log "env: BT_WALLET_COLD=$BT_WALLET_COLD BT_WALLET_HOT=$BT_WALLET_HOT"
log "stages: ${STAGES[*]}"

# ------------------------------------------------------------------
# 7. run gate stages
# ------------------------------------------------------------------
for stage in "${STAGES[@]}"; do
    stop_port_a

    case "$stage" in
        dethrone)
            start_stub_on_a
            timeout="$DETHRONE_TIMEOUT"
            ;;
        smoke)
            start_vllm_on_a
            timeout="$SMOKE_TIMEOUT"
            ;;
        full)
            start_vllm_on_a
            timeout="$FULL_TIMEOUT"
            ;;
    esac

    log "running gate: stage=$stage timeout=${timeout}s"
    python3 scripts/e2e/run_gate.py \
        --stage "$stage" \
        --local \
        --smoke-timeout "$SMOKE_TIMEOUT" \
        --full-timeout "$FULL_TIMEOUT" \
        --dethrone-timeout "$DETHRONE_TIMEOUT"

    log "$stage gate passed"
done

log "all E2E gates passed: ${STAGES[*]}"
