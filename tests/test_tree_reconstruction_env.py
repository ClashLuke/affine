import math

import pytest

from affine.envs.tree_reconstruction import HiddenTree, TreeReconstructionEnv


def test_hidden_tree_prufer_is_deterministic():
    a = HiddenTree(12, 123, "prufer")
    b = HiddenTree(12, 123, "prufer")
    c = HiddenTree(12, 124, "prufer")
    assert a.parent == b.parent
    assert a.parent != c.parent
    assert a.parent[0] == -1
    assert all(0 <= p < a.n for p in a.parent[1:])


def test_hidden_tree_lower_bound_matches_generator_sample_space():
    assert HiddenTree(10, 0, "prufer").lower_bound_bits() == pytest.approx(8 * math.log2(10))
    assert HiddenTree(10, 0, "recursive").lower_bound_bits() == pytest.approx(math.log2(math.factorial(9)))


def test_tree_queries_match_parent_structure():
    tree = HiddenTree(8, 42, "recursive")
    for child, parent in enumerate(tree.parent[1:], start=1):
        assert tree.query("ANCESTOR", [parent, child]).value is True
        assert tree.query("DEPTH", [child]).value == tree.depth[child]
        assert child in tree.query("CHILDREN", [parent]).value
        path = tree.query("PATH", [parent, child]).value
        assert path == [parent, child]


def test_tree_env_query_then_exact_submit():
    env = TreeReconstructionEnv(
        n=6, method="recursive", max_turns=4,
        allowed_queries=("ANCESTOR", "LCA", "DEPTH", "CHILDREN"),
    )
    prompt, reset = env.reset(seed=7)
    assert "hidden rooted tree" in prompt
    assert reset["env_id"] == "tree_reconstruction"
    assert reset["n"] == 6

    obs, reward, terminated, truncated, info = env.step("QUERY CHILDREN 0\nQUERY DEPTH 3")
    assert reward == 0.0
    assert not terminated and not truncated
    assert "CHILDREN 0:" in obs
    assert "DEPTH 3:" in obs
    assert info["query_count"] == 2

    parent = env._tree.parent
    _obs, reward, terminated, truncated, info = env.step("SUBMIT " + " ".join(map(str, parent[1:])))
    assert reward == 1.0
    assert terminated and not truncated
    assert info["success"] is True
    assert info["correct"] == 5


def test_tree_default_queries_do_not_allow_direct_child_dump():
    env = TreeReconstructionEnv(n=5)
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step("QUERY CHILDREN 0")
    assert "ERROR query type not allowed" in obs
    assert (reward, terminated, truncated) == (0.0, False, False)
    assert info["query_count"] == 0


def test_tree_env_reset_options_are_per_episode():
    env = TreeReconstructionEnv(n=5)
    _prompt, info = env.reset(seed=0, options={"n": 8, "max_turns": 3})
    assert info["n"] == 8
    assert info["max_turns"] == 3
    _prompt, info = env.reset(seed=0)
    assert info["n"] == 5
    assert info["max_turns"] == 32


def test_tree_env_wrong_submit_is_terminal_loss():
    env = TreeReconstructionEnv(n=5, method="recursive")
    env.reset(seed=0)
    _obs, reward, terminated, truncated, info = env.step("SUBMIT 0 0 0 0")
    assert terminated and not truncated
    assert reward < 1.0
    assert info["success"] is False


def test_tree_env_invalid_protocol_is_terminal_loss():
    env = TreeReconstructionEnv(n=5)
    env.reset(seed=0)
    _obs, reward, terminated, truncated, info = env.step("I do not know")
    assert (reward, terminated, truncated) == (0.0, True, False)
    assert info["error"] == "expected QUERY or SUBMIT"


def test_tree_env_rejects_bare_numeric_and_trailing_junk_submissions():
    env = TreeReconstructionEnv(n=5, method="recursive")
    env.reset(seed=0)
    parent = " ".join(map(str, env._tree.parent[1:]))
    _obs, reward, terminated, truncated, info = env.step(parent)
    assert (reward, terminated, truncated) == (0.0, True, False)
    assert info["error"] == "expected QUERY or SUBMIT"

    env.reset(seed=0)
    _obs, reward, terminated, truncated, info = env.step(f"SUBMIT {parent} junk")
    assert (reward, terminated, truncated) == (0.0, True, False)
    assert info["error"] == "malformed submission"


def test_tree_env_rejects_answer_wrapping():
    env = TreeReconstructionEnv(n=5, method="recursive")
    env.reset(seed=0)
    parent = " ".join(map(str, env._tree.parent[1:]))
    _obs, reward, terminated, truncated, info = env.step(f"<ANSWER>SUBMIT {parent}</ANSWER>")
    assert (reward, terminated, truncated) == (0.0, True, False)
    assert info["error"] == "expected QUERY or SUBMIT"


def test_tree_env_query_limit_truncates():
    env = TreeReconstructionEnv(n=5, max_queries=1)
    env.reset(seed=0)
    obs, _reward, terminated, truncated, _info = env.step("QUERY DEPTH 1")
    assert obs.startswith("DEPTH 1:")
    _obs, reward, terminated, truncated, info = env.step("QUERY DEPTH 2")
    assert (reward, terminated, truncated) == (0.0, False, True)
    assert info["error"] == "query limit reached"


def test_tree_env_query_limit_preserves_completed_batch_feedback():
    env = TreeReconstructionEnv(n=5, max_queries=1)
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step("QUERY DEPTH 1\nQUERY DEPTH 2")
    assert "DEPTH 1:" in obs
    assert "ERROR query limit reached" in obs
    assert (reward, terminated, truncated) == (0.0, False, True)
    assert info["query_count"] == 1


def test_tree_env_bad_submit_shape_is_loss_not_exception():
    env = TreeReconstructionEnv(n=5)
    env.reset(seed=0)
    _obs, reward, terminated, truncated, info = env.step("SUBMIT 0 1")
    assert (reward, terminated, truncated) == (0.0, True, False)
    assert "expected 4 parent values" in info["error"]


def test_hidden_tree_rejects_unknown_method():
    with pytest.raises(ValueError, match="unknown tree generation method"):
        HiddenTree(5, 0, "bad")


@pytest.mark.parametrize("options,msg", [
    ({"n": 1}, r"n must be in \[2,"),
    ({"method": "bad"}, "method must be"),
    ({"max_turns": 0}, r"max_turns must be in \[1,"),
    ({"max_queries": -1}, r"max_queries must be in \[0,"),
    ({"allowed_queries": ["BAD"]}, "unknown allowed_queries"),
])
def test_tree_env_rejects_invalid_options(options, msg):
    env = TreeReconstructionEnv()
    with pytest.raises(ValueError, match=msg):
        env.reset(seed=0, options=options)
