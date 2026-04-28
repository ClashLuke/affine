import json
import time

from affine.envs.boolean_circuit import BooleanCircuitEnv


def _score(env, body: str) -> float:
    return env.step(f"<ANSWER>{body}</ANSWER>")[1]


def _wrong(env) -> str:
    return json.dumps({"count": env._count + 1})


def test_reset_is_deterministic():
    a, _ = BooleanCircuitEnv().reset(seed=17)
    b, _ = BooleanCircuitEnv().reset(seed=17)
    assert a == b


def test_different_seeds_generate_unique_tasks():
    tasks = {BooleanCircuitEnv().reset(seed=i)[0] for i in range(10)}
    assert len(tasks) == 10


def test_known_correct_scores_one():
    env = BooleanCircuitEnv()
    env.reset(seed=4)
    assert _score(env, env._target) == 1.0
    assert env._influence >= env.options["min_influence"]


def test_fallback_preserves_generation_constraints():
    env = BooleanCircuitEnv()
    env.reset(seed=387)
    assert _score(env, env._target) == 1.0
    assert env._influence >= env.options["min_influence"]
    assert (1 << env.options["variables"]) // 8 <= env._count <= (1 << env.options["variables"]) * 7 // 8


def test_extreme_valid_config_uses_bounded_generation():
    t0 = time.monotonic()
    prompts = set()
    for seed in range(5):
        env = BooleanCircuitEnv(variables=12, gates=64, min_influence=11)
        prompt, _info = env.reset(seed=seed)
        prompts.add(prompt)
        assert _score(env, env._target) == 1.0
        assert env._influence >= env.options["min_influence"]
    assert len(prompts) > 1
    assert time.monotonic() - t0 < 10.0


def test_fallback_varies_by_seed_for_tight_valid_config():
    prompts = set()
    counts = set()
    for seed in range(20):
        env = BooleanCircuitEnv()
        prompt, _info = env.reset(seed=seed, options={"variables": 9, "gates": 6, "min_influence": 7})
        prompts.add(prompt)
        counts.add(env._count)
        assert env._influence >= env.options["min_influence"]
    assert len(prompts) > 1
    assert len(counts) > 1


def test_known_wrong_scores_zero():
    env = BooleanCircuitEnv()
    env.reset(seed=4)
    assert _score(env, _wrong(env)) == 0.0


def test_score_is_binary_and_stable():
    env = BooleanCircuitEnv()
    env.reset(seed=9)
    scores = [_score(env, env._target), _score(env, _wrong(env)), _score(env, env._target)]
    assert scores == [1.0, 0.0, 1.0]
    assert set(scores) <= {0.0, 1.0}


def test_rejects_multiple_answer_blocks_and_duplicate_keys():
    env = BooleanCircuitEnv()
    env.reset(seed=11)
    assert env.step(f"<ANSWER>{env._target}</ANSWER><ANSWER>{env._target}</ANSWER>")[1] == 0.0
    assert _score(env, f'{{"count":{env._count},"count":{env._count}}}') == 0.0
