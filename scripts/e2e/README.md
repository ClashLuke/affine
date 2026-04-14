# Affine Rewrite E2E Gate

## Quick Start

```bash
# Full E2E: tests → subtensor → vLLM → bootstrap → gate
scripts/e2e/run_e2e.sh

# Skip image pulls (already cached)
SKIP_PULL=1 scripts/e2e/run_e2e.sh

# Skip unit tests
SKIP_TESTS=1 scripts/e2e/run_e2e.sh

# Run full stage instead of smoke
GATE_STAGE=full GATE_TIMEOUT=7200 scripts/e2e/run_e2e.sh

# Custom model
MODEL=meta-llama/Llama-3.2-1B-Instruct scripts/e2e/run_e2e.sh
```

Requires: Docker, a GPU with enough VRAM for two model instances, `vllm` and `affine` installed.

## What It Does

`run_e2e.sh` automates the full pipeline:

1. **Unit tests** — `pytest tests/ -v` (47 tests)
2. **Pull images** — subtensor devnet + 4 environment containers
3. **Start subtensor** — local dev chain with instant block sealing
4. **Start vLLM** — two instances sharing the GPU (0.35 util each)
5. **Bootstrap chain** — wallets, subnet, registrations, commitments
6. **Run gate** — launches `affine` and asserts all milestones are hit

The gate passes when the log shows:
- subtensor connected
- non-empty challenger queue
- slot health checks (≥2)
- successful weight set
- duel summary
- all 4 env summaries (`affine:ded`, `affine:abd`, `game`, `distill`)
- clean shutdown

## Environment Variables

| Var | Default | Purpose |
|-----|---------|---------|
| `MODEL` | `Qwen/Qwen2.5-0.5B-Instruct` | Model for both vLLM servers |
| `VLLM_PORT_A` | `8000` | Champion model server port |
| `VLLM_PORT_B` | `8001` | Challenger model server port |
| `GPU_MEM_UTIL` | `0.35` | Per-instance GPU memory fraction |
| `MAX_MODEL_LEN` | `2048` | Max sequence length |
| `GATE_STAGE` | `smoke` | Gate stage: `smoke`, `full`, or `both` |
| `GATE_TIMEOUT` | `600` | Per-stage timeout in seconds |
| `SKIP_TESTS` | unset | Set to skip unit tests |
| `SKIP_PULL` | unset | Set to skip docker pulls |

## Manual Steps

If you prefer to run each piece separately:

```bash
# 1. subtensor
docker run -d --name subtensor-dev -p 9944:9944 -p 9945:9945 \
    ghcr.io/opentensor/subtensor-localnet:devnet-ready True

# 2. vLLM (two terminals)
vllm serve Qwen/Qwen2.5-0.5B-Instruct --port 8000 --gpu-memory-utilization 0.35
vllm serve Qwen/Qwen2.5-0.5B-Instruct --port 8001 --gpu-memory-utilization 0.35

# 3. bootstrap
python3 scripts/e2e/bootstrap_dev_chain.py --endpoint ws://127.0.0.1:9944

# 4. export (copy from bootstrap output)
export SUBTENSOR_ENDPOINT=ws://127.0.0.1:9944
export NETUID=2
export BT_WALLET_COLD=affine-e2e-validator
export BT_WALLET_HOT=validator
export AFFINE_LOCAL=1
export CHAMPION_URL=http://localhost:8000/v1
export CHALLENGER_URL=http://localhost:8001/v1

# 5. gate
python3 scripts/e2e/run_gate.py --stage smoke --local
```

## Artifacts

| Path | Contents |
|------|----------|
| `.e2e/dev_chain_state.json` | Bootstrap output (netuid, wallets, hotkeys) |
| `.e2e/logs/smoke.log` | Full process output from smoke stage |
| `.e2e/results/smoke.json` | Gate assertion results |
| `.e2e/vllm_a.log` | vLLM server A log (script mode only) |
| `.e2e/vllm_b.log` | vLLM server B log (script mode only) |
