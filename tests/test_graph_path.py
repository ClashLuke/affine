import json

from affine.envs.graph_path import GraphPathEnv, _shortest


def _score(env, body: str) -> float:
    return env.step(f"<ANSWER>{body}</ANSWER>")[1]


def _wrong(env) -> str:
    return json.dumps({"distance": env._distance + 1, "path": list(env._path)})


def test_reset_is_deterministic():
    a, _ = GraphPathEnv().reset(seed=17)
    b, _ = GraphPathEnv().reset(seed=17)
    assert a == b


def test_different_seeds_generate_unique_tasks():
    tasks = {GraphPathEnv().reset(seed=i)[0] for i in range(10)}
    assert len(tasks) == 10


def test_known_correct_scores_one():
    env = GraphPathEnv()
    env.reset(seed=4)
    assert _score(env, env._target) == 1.0


def test_requested_edge_count_is_hard_cap():
    for seed in range(20):
        _prompt, info = GraphPathEnv(nodes=16, edges=4, min_path_len=5).reset(seed=seed)
        assert info["actual_edges"] == 4


def test_dense_graph_constraints_remain_generatable():
    env = GraphPathEnv()
    _prompt, info = env.reset(seed=0, options={"nodes": 16, "edges": 120, "min_path_len": 8})
    assert info["actual_edges"] == 120
    assert len(env._path) >= 8
    assert _score(env, env._target) == 1.0


def test_known_wrong_scores_zero():
    env = GraphPathEnv()
    env.reset(seed=4)
    assert _score(env, _wrong(env)) == 0.0


def test_score_is_binary_and_stable():
    env = GraphPathEnv()
    env.reset(seed=9)
    scores = [_score(env, env._target), _score(env, _wrong(env)), _score(env, env._target)]
    assert scores == [1.0, 0.0, 1.0]
    assert set(scores) <= {0.0, 1.0}


def test_rejects_multiple_answer_blocks_and_duplicate_keys():
    env = GraphPathEnv()
    env.reset(seed=11)
    assert env.step(f"<ANSWER>{env._target}</ANSWER><ANSWER>{env._target}</ANSWER>")[1] == 0.0
    dup = f'{{"distance":{env._distance},"distance":{env._distance},"path":{json.dumps(list(env._path))}}}'
    assert _score(env, dup) == 0.0


def test_shortest_path_keeps_later_lexicographic_tie():
    edges = {
        0: [(1, 5), (2, 1)],
        1: [(4, 5)],
        2: [(4, 9)],
        3: [],
        4: [],
    }
    assert _shortest(5, edges, 0, 4) == (10, (0, 1, 4))
