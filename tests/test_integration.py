"""Integration test: real vLLM + real Docker env + real duel scoring.

Requires:
  - vLLM server running on http://localhost:8000/v1
  - Docker daemon running
  - number-guess:latest image built

Run: python -m pytest tests/test_integration.py -v -s
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import affinetes as af
from affine.duel import run_duel, _eval
from affine.vllm import Slot, health_ping, LocalSlots
from affine.wilson import Verdict
from affine.config import EnvSpec

VLLM_URL = "http://localhost:8000/v1"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ENV_IMAGE = "number-guess:latest"
ENV_VARS = {"OPENAI_API_KEY": "dummy", "CHUTES_API_KEY": "dummy"}


async def check_prereqs():
    if not await health_ping(VLLM_URL):
        print(f"SKIP: vLLM not running at {VLLM_URL}")
        return False
    return True


def load_env():
    return af.load_env(
        image=ENV_IMAGE, mode="docker", mem_limit="512m",
        pull=False, cleanup=True, host_network=True, host_port=8001,
        env_vars=ENV_VARS,
    )


async def test_single_eval():
    """Run a single environment evaluation with real inference."""
    if not await check_prereqs():
        return

    env = load_env()

    try:
        result = await env.evaluate(
            model=MODEL,
            base_url=VLLM_URL,
            seed=42,
            task_id=1,
            temperature=0.0,
            timeout=60,
        )
        print(f"  single eval: success={result.get('success')}, score={result.get('score')}, time={result.get('time_taken', 0):.1f}s")
        assert isinstance(result, dict)
        assert "success" in result or "score" in result
    finally:
        await env.cleanup()


async def test_duel_same_model():
    """Run a full duel where both sides use the same model.

    Expected: mostly ties (same model, same seed → same answers).
    The duel should complete and produce a verdict, not crash.
    """
    if not await check_prereqs():
        return

    champion = Slot(model=MODEL, revision="main", base_url=VLLM_URL)
    challenger = Slot(model=MODEL, revision="main", base_url=VLLM_URL)
    spec = EnvSpec(name="game:number_guess", image=ENV_IMAGE, params={"temperature": 0.0, "timeout": 60})
    env = load_env()

    try:
        envs = {"game:number_guess": (env, spec)}
        verdict = await run_duel(
            envs, champion, challenger,
            max_tasks=8,
            tasks_per_batch=2,
            z=1.96,
            nonce=1,
        )
        print(f"  duel (same model): {verdict.name}")
        assert isinstance(verdict, Verdict)
    finally:
        await env.cleanup()


async def test_duel_dead_champion():
    """Duel where champion endpoint doesn't exist. Challenger should win by default."""
    if not await check_prereqs():
        return

    champion = Slot(model="dead-model", revision="main", base_url="http://localhost:9999/v1")
    challenger = Slot(model=MODEL, revision="main", base_url=VLLM_URL)
    spec = EnvSpec(name="game:number_guess", image=ENV_IMAGE, params={"temperature": 0.0, "timeout": 10})
    env = load_env()

    try:
        envs = {"game:number_guess": (env, spec)}
        verdict = await run_duel(
            envs, champion, challenger,
            max_tasks=8,
            tasks_per_batch=2,
            z=1.96,
            nonce=2,
        )
        print(f"  duel (dead champion): {verdict.name}")
        assert verdict is Verdict.CHALLENGER_WINS, f"expected CHALLENGER_WINS, got {verdict.name}"
    finally:
        await env.cleanup()


async def test_local_slots_duel():
    """Full slot lifecycle: provision, health check, duel, teardown."""
    if not await check_prereqs():
        return

    slots = LocalSlots(VLLM_URL, VLLM_URL)
    spec = EnvSpec(name="game:number_guess", image=ENV_IMAGE, params={"temperature": 0.0, "timeout": 60})
    env = load_env()

    try:
        champ_slot = await slots.provision(MODEL, "main")
        assert await health_ping(champ_slot.base_url)

        chall_slot = await slots.provision(MODEL, "main")
        assert await health_ping(chall_slot.base_url)

        envs = {"game:number_guess": (env, spec)}
        verdict = await run_duel(
            envs, champ_slot, chall_slot,
            max_tasks=4,
            tasks_per_batch=2,
            z=1.96,
            nonce=3,
        )
        print(f"  local slots duel: {verdict.name}")

        await slots.teardown(chall_slot)
        await slots.teardown(champ_slot)

        # Verify slots returned to pool
        s = await slots.provision("test", "r")
        await slots.teardown(s)
    finally:
        await env.cleanup()


async def main():
    tests = [
        ("single eval", test_single_eval),
        ("duel (same model)", test_duel_same_model),
        ("duel (dead champion)", test_duel_dead_champion),
        ("local slots lifecycle", test_local_slots_duel),
    ]

    passed, failed = 0, 0
    for name, fn in tests:
        print(f"\n--- {name} ---")
        try:
            await fn()
            print(f"  PASS")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            failed += 1

    print(f"\n{'='*40}")
    print(f"{passed} passed, {failed} failed out of {len(tests)}")
    return failed == 0


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    ok = asyncio.run(main())
    sys.exit(0 if ok else 1)
