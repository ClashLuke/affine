# Affine Rewrite E2E Gate

## Quick Start

```bash
# Full pipeline: tests → subtensor → bootstrap → dethrone + smoke
scripts/e2e/run_e2e.sh

# Individual stages
GATE_STAGE=dethrone scripts/e2e/run_e2e.sh
GATE_STAGE=smoke scripts/e2e/run_e2e.sh
GATE_STAGE=full FULL_TIMEOUT=7200 scripts/e2e/run_e2e.sh

# Skip slow steps
SKIP_PULL=1 SKIP_TESTS=1 scripts/e2e/run_e2e.sh
```

Requires: Docker, a GPU with enough VRAM for two model instances (0.35 each), `vllm` and `affine` installed.

## What It Does

`run_e2e.sh` runs a unified pipeline. Shared infra (subtensor, vLLM-B on port 8001, bootstrap) starts once. Port 8000 swaps between a stub server and a real vLLM depending on the stage.

1. **Unit tests** — `pytest tests/ -v`
2. **Pull images** — subtensor devnet + environment containers
3. **Start subtensor** — local dev chain with instant block sealing
4. **Start vLLM-B** — real model on port 8001 (stays up for all stages)
5. **Bootstrap chain** — wallets, subnet, registrations, commitments
6. **Run stages** — each stage swaps port 8000's server, then runs the gate

## Stages

| Stage | Port 8000 | Port 8001 | Tests |
|-------|-----------|-----------|-------|
| `dethrone` | stub (garbage completions) | real vLLM | Decisive outcomes, dethronement, weight transition, 2+ duels, queue exhaustion |
| `smoke` | real vLLM | real vLLM | Same-model ties, CHAMPION_HOLDS by default, all 4 env summaries |
| `full` | real vLLM | real vLLM | Same as smoke with larger task budget |

`GATE_STAGE=all` (default) runs `dethrone` then `smoke`. `GATE_STAGE=both` runs `smoke` then `full` (legacy).

### Dethrone Stage

The stub server returns `"I don't know"` to every chat completion — a syntactically valid but always-wrong answer. Cold start picks the stub as champion (first-committed miner → FIFO slot → port 8000). The duel produces decisive outcomes where the real model wins. Expected flow:

1. Cold start → stub champion
2. Duel 1: stub vs real → CHALLENGER_WINS (z > k)
3. Dethronement → real model becomes champion, weights updated
4. Duel 2: real vs stub → CHAMPION_HOLDS (hopelessness)
5. Queue exhausted

### Smoke/Full Stage

Both servers run the same model. All evaluations tie (0 decisive outcomes). Tests the default CHAMPION_HOLDS path, environment loading, health checks, weight setting, and clean shutdown.

## Environment Variables

| Var | Default | Purpose |
|-----|---------|---------|
| `MODEL` | `Qwen/Qwen2.5-0.5B-Instruct` | Model for vLLM servers |
| `VLLM_PORT_A` | `8000` | Champion/stub port (swapped per stage) |
| `VLLM_PORT_B` | `8001` | Challenger port (shared) |
| `GPU_MEM_UTIL` | `0.35` | Per-instance GPU memory fraction |
| `MAX_MODEL_LEN` | `2048` | Max sequence length |
| `GATE_STAGE` | `all` | `all`, `dethrone`, `smoke`, `full`, `both` |
| `SMOKE_TIMEOUT` | `2400` | Smoke stage timeout (seconds) |
| `FULL_TIMEOUT` | `7200` | Full stage timeout (seconds) |
| `DETHRONE_TIMEOUT` | `900` | Dethrone stage timeout (seconds) |
| `SKIP_TESTS` | unset | Skip unit tests |
| `SKIP_PULL` | unset | Skip docker image pulls |

## Manual Steps

```bash
# 1. subtensor
docker run -d --name subtensor-dev -p 9944:9944 -p 9945:9945 \
    ghcr.io/opentensor/subtensor-localnet:devnet-ready True

# 2a. dethrone: stub + one vLLM
python3 scripts/e2e/stub_vllm.py --port 8000 --model Qwen/Qwen2.5-0.5B-Instruct &
vllm serve Qwen/Qwen2.5-0.5B-Instruct --port 8001 --gpu-memory-utilization 0.35

# 2b. smoke: two vLLMs
vllm serve Qwen/Qwen2.5-0.5B-Instruct --port 8000 --gpu-memory-utilization 0.35
vllm serve Qwen/Qwen2.5-0.5B-Instruct --port 8001 --gpu-memory-utilization 0.35

# 3. bootstrap
python3 scripts/e2e/bootstrap_dev_chain.py --endpoint ws://127.0.0.1:9944

# 4. export (copy from bootstrap output)
export SUBTENSOR_ENDPOINT=ws://127.0.0.1:9944 NETUID=2
export BT_WALLET_COLD=affine-e2e-validator BT_WALLET_HOT=validator
export AFFINE_LOCAL=1
export CHAMPION_URL=http://localhost:8000/v1 CHALLENGER_URL=http://localhost:8001/v1

# 5. gate
python3 scripts/e2e/run_gate.py --stage dethrone --local
python3 scripts/e2e/run_gate.py --stage smoke --local
```

## Artifacts

| Path | Contents |
|------|----------|
| `.e2e/dev_chain_state.json` | Bootstrap output (netuid, wallets, hotkeys) |
| `.e2e/logs/{stage}.log` | Full process output per stage |
| `.e2e/results/{stage}.json` | Gate assertion results per stage |
| `.e2e/stub.log` | Stub server log |
| `.e2e/vllm_a.log` | vLLM-A log (port 8000) |
| `.e2e/vllm_b.log` | vLLM-B log (port 8001) |
