import pytest

from affine.envs.python_interpreter import PythonInterpreterEnv


def test_python_interpreter_scores_exact_output_whitespace():
    env = PythonInterpreterEnv(lines=1, ops=("PRINT",))
    env.reset(seed=0)
    assert env._target == "\n\n"
    assert env.step(f"<ANSWER>{env._target}</ANSWER>")[1] == 1.0
    assert env.step("<ANSWER>\n</ANSWER>")[1] == 0.0


def test_python_interpreter_rejects_multiple_answer_blocks():
    env = PythonInterpreterEnv(lines=1, ops=("PRINT",))
    env.reset(seed=0)
    assert env.step("<ANSWER>\n</ANSWER><ANSWER>\n</ANSWER>")[1] == 0.0


@pytest.mark.parametrize("options,msg", [
    ({"lines": 0}, "lines"),
    ({"lines": True}, "lines must be an integer"),
    ({"max_digits": 0}, "max_digits"),
    ({"ops": ["NO_SUCH"]}, "unknown ops"),
    ({"ops": []}, "ops must be a non-empty list"),
])
def test_python_interpreter_rejects_invalid_options(options, msg):
    env = PythonInterpreterEnv()
    with pytest.raises(ValueError, match=msg):
        env.reset(seed=0, options=options)
