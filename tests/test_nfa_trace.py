import json

from affine.envs.nfa_trace import NFATraceEnv


def _score(env, body: str) -> float:
    return env.step(f"<ANSWER>{body}</ANSWER>")[1]


def test_reset_is_deterministic():
    a, _ = NFATraceEnv().reset(seed=17)
    b, _ = NFATraceEnv().reset(seed=17)
    assert a == b


def test_different_seeds_generate_unique_tasks():
    tasks = {NFATraceEnv().reset(seed=i)[0] for i in range(10)}
    assert len(tasks) == 10


def test_known_correct_scores_one():
    env = NFATraceEnv()
    env.reset(seed=4)
    assert _score(env, env._target) == 1.0


def test_known_wrong_scores_zero():
    env = NFATraceEnv()
    env.reset(seed=4)
    wrong = json.dumps({"reachable": list(env._final), "accept": not env._accept})
    assert _score(env, wrong) == 0.0


def test_score_is_binary_and_stable():
    env = NFATraceEnv()
    env.reset(seed=9)
    wrong = json.dumps({"reachable": list(env._final), "accept": not env._accept})
    scores = [_score(env, env._target), _score(env, wrong), _score(env, env._target)]
    assert scores == [1.0, 0.0, 1.0]
    assert set(scores) <= {0.0, 1.0}


def test_rejects_multiple_answer_blocks_and_duplicate_keys():
    env = NFATraceEnv()
    env.reset(seed=11)
    assert env.step(f"<ANSWER>{env._target}</ANSWER><ANSWER>{env._target}</ANSWER>")[1] == 0.0
    dup = '{"reachable":[],"reachable":[],"accept":false}'
    assert _score(env, dup) == 0.0
