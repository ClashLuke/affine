import json
import math

from affine.envs.modular_crt import ModularCRTEnv


def _score(env, body: str) -> float:
    return env.step(f"<ANSWER>{body}</ANSWER>")[1]


def _wrong(env) -> str:
    return json.dumps({
        "residues": {str(mod): residue for mod, residue in env._residues.items()},
        "crt": (env._crt + 1) % math.prod(env._residues),
    })


def test_reset_is_deterministic():
    a, _ = ModularCRTEnv().reset(seed=17)
    b, _ = ModularCRTEnv().reset(seed=17)
    assert a == b


def test_different_seeds_generate_unique_tasks():
    tasks = {ModularCRTEnv().reset(seed=i)[0] for i in range(10)}
    assert len(tasks) == 10


def test_known_correct_scores_one():
    env = ModularCRTEnv()
    env.reset(seed=4)
    assert _score(env, env._target) == 1.0


def test_known_wrong_scores_zero():
    env = ModularCRTEnv()
    env.reset(seed=4)
    assert _score(env, _wrong(env)) == 0.0


def test_score_is_binary_and_stable():
    env = ModularCRTEnv()
    env.reset(seed=9)
    scores = [_score(env, env._target), _score(env, _wrong(env)), _score(env, env._target)]
    assert scores == [1.0, 0.0, 1.0]
    assert set(scores) <= {0.0, 1.0}


def test_rejects_multiple_answer_blocks_and_duplicate_keys():
    env = ModularCRTEnv()
    env.reset(seed=11)
    mod, residue = next(iter(env._residues.items()))
    assert env.step(f"<ANSWER>{env._target}</ANSWER><ANSWER>{env._target}</ANSWER>")[1] == 0.0
    dup = f'{{"residues":{{"{mod}":{residue},"{mod}":{residue}}},"crt":{env._crt}}}'
    assert _score(env, dup) == 0.0


def test_rejects_noncanonical_residue_keys():
    env = ModularCRTEnv()
    env.reset(seed=0)
    residues = {str(mod): residue for mod, residue in env._residues.items()}
    mod, residue = next(iter(env._residues.items()))
    residues[f"0{mod}"] = residue
    del residues[str(mod)]
    assert _score(env, json.dumps({"residues": residues, "crt": env._crt})) == 0.0
