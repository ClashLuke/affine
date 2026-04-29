from affine.config import ENV_REGISTRY
from affine.envs import EnvFactory


def test_reset_info_does_not_expose_answers():
    forbidden = {
        "nfa": {"final_size", "accept", "reachable"},
        "graph": {"distance", "path", "path_len"},
        "modular": {"residues", "crt"},
        "sudoku": {"grid"},
        "boolean": {"count"},
        "tree": {"parent", "parents", "submitted_parent"},
        "python": {"target", "answer", "output"},
    }
    for name, spec in ENV_REGISTRY.items():
        env = EnvFactory(spec.entrypoint).make()
        env_options = {k: v for k, v in spec.params.items() if k in env.option_keys}
        _prompt, info = env.reset(seed=0, options=env_options)
        assert forbidden[name].isdisjoint(info), (name, info)
