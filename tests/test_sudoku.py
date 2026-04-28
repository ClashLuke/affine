import json

from affine.envs.sudoku import SudokuEnv


def _score(env, body: str) -> float:
    return env.step(f"<ANSWER>{body}</ANSWER>")[1]


def _wrong(env) -> str:
    rows = env._grid[:]
    first = "1" if rows[0][0] != "1" else "2"
    rows[0] = first + rows[0][1:]
    return json.dumps({"grid": rows})


def test_reset_is_deterministic():
    a, _ = SudokuEnv().reset(seed=17)
    b, _ = SudokuEnv().reset(seed=17)
    assert a == b


def test_different_seeds_generate_unique_tasks():
    tasks = {SudokuEnv().reset(seed=i)[0] for i in range(10)}
    assert len(tasks) == 10


def test_known_correct_scores_one():
    env = SudokuEnv()
    env.reset(seed=4)
    assert _score(env, env._target) == 1.0
    assert env._branch_points >= env.options["min_branch_points"]


def test_known_wrong_scores_zero():
    env = SudokuEnv()
    env.reset(seed=4)
    assert _score(env, _wrong(env)) == 0.0


def test_score_is_binary_and_stable():
    env = SudokuEnv()
    env.reset(seed=9)
    scores = [_score(env, env._target), _score(env, _wrong(env)), _score(env, env._target)]
    assert scores == [1.0, 0.0, 1.0]
    assert set(scores) <= {0.0, 1.0}


def test_rejects_multiple_answer_blocks_and_duplicate_keys():
    env = SudokuEnv()
    env.reset(seed=11)
    assert env.step(f"<ANSWER>{env._target}</ANSWER><ANSWER>{env._target}</ANSWER>")[1] == 0.0
    dup = f'{{"grid":{json.dumps(env._grid)},"grid":{json.dumps(env._grid)}}}'
    assert _score(env, dup) == 0.0
