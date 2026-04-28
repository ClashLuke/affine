import json
import random
import re

from ._base import Env

static = ("In the following, we will test your ability to track a nondeterministic finite automaton.\n\n"
          "RULES:\n"
          "* You will not be able to use external tools during this test\n"
          "* The automaton has no epsilon transitions\n"
          "* After each input character, the active state set is the union of all listed destinations\n"
          "* Return the final active state set and whether any active state is accepting\n"
          "* Only outputs wrapped in XML-style <ANSWER></ANSWER> tags will be evaluated\n"
          "* You may write any text outside of the <ANSWER> tags\n\n"
          "Example:\n"
          "CHALLENGE:\n"
          "states: 0..3\n"
          "start: 0\n"
          "accepting: {2}\n"
          "input: ab\n"
          "transitions:\n"
          "0: a->{0,1} b->{1}\n"
          "1: a->{2} b->{3}\n"
          "2: a->{2} b->{}\n"
          "3: a->{} b->{2}\n"
          "RESPONSE:\n"
          "<ANSWER>{\"reachable\":[1,3],\"accept\":false}</ANSWER>\n\n"
          "Below, you will see the real task. Remember and follow the rules.\n\n"
          "CHALLENGE:\n")


def _run(start: int, word: str, transitions: dict[tuple[int, str], tuple[int, ...]]) -> tuple[int, ...]:
    active = {start}
    for ch in word:
        active = {dst for state in active for dst in transitions[state, ch]}
    return tuple(sorted(active))


def generate(states: int, length: int, alphabet: str, accept_count: int, rng: random.Random):
    found = False
    for _ in range(1000):
        transitions = {}
        for state in range(states):
            for ch in alphabet:
                deg = rng.choices((0, 1, 2, 3), (1, 5, 3, 1))[0]
                transitions[state, ch] = tuple(sorted(rng.sample(range(states), deg)))
        start = rng.randrange(states)
        word = ''.join(rng.choice(alphabet) for _ in range(length))
        final = _run(start, word, transitions)
        if 1 <= len(final) <= states - accept_count:
            found = True
            break
    if not found:
        raise ValueError("could not generate non-degenerate NFA trace")

    outside = [s for s in range(states) if s not in final]
    if rng.random() < 0.5:
        accepting = {rng.choice(final)}
        accepting.update(rng.sample(outside, accept_count - 1))
    else:
        accepting = set(rng.sample(outside, accept_count))

    rows = []
    for state in range(states):
        cols = []
        for ch in alphabet:
            dst = ','.join(map(str, transitions[state, ch]))
            cols.append(f"{ch}->{{{dst}}}")
        rows.append(f"{state}: {' '.join(cols)}")
    prompt = (
        f"states: 0..{states - 1}\n"
        f"start: {start}\n"
        f"accepting: {{{','.join(map(str, sorted(accepting)))}}}\n"
        f"input: {word}\n"
        "transitions:\n" + '\n'.join(rows)
    )
    accept = any(s in accepting for s in final)
    target = json.dumps({"reachable": list(final), "accept": accept}, separators=(",", ":"))
    return prompt, final, accept, target


def _tagged(action: str) -> str | None:
    matches = re.findall(r"<ANSWER>(.*?)</ANSWER>", action or "", re.IGNORECASE | re.DOTALL)
    return matches[0].strip() if len(matches) == 1 else None


def _json(body: str):
    def hook(pairs):
        out = {}
        for k, v in pairs:
            if k in out:
                raise ValueError(k)
            out[k] = v
        return out
    try:
        return json.loads(body, object_pairs_hook=hook)
    except (json.JSONDecodeError, ValueError):
        return None


def _parse(body: str):
    obj = _json(body)
    if not isinstance(obj, dict) or set(obj) != {"reachable", "accept"} or not isinstance(obj["accept"], bool):
        return None
    reachable = obj["reachable"]
    if not isinstance(reachable, list) or any(type(x) is not int for x in reachable):
        return None
    return tuple(sorted(reachable)), obj["accept"]


class NFATraceEnv(Env):
    __version__: str = "0.0.1"

    def __init__(self, states: int = 10, length: int = 16, alphabet: str = "abc", accept_count: int = 3):
        self._defaults = self.validate_options({
            "states": states, "length": length, "alphabet": alphabet, "accept_count": accept_count,
        })

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        opts = options or {}
        params = self.validate_options({**self._defaults, **opts})
        states = params["states"]
        length = params["length"]
        alphabet = params["alphabet"]
        accept_count = params["accept_count"]
        prompt, final, accept, self._target = generate(
            states, length, alphabet, accept_count, random.Random(0 if seed is None else seed)
        )
        self._final = final
        self._accept = accept
        return static + prompt, {
            "challenge_id": str(seed if seed is not None else 0),
            "env_id": "nfa_trace",
            "spec_version": self.__version__,
            **params,
        }

    def step(self, action: str):
        parsed = _parse(body) if (body := _tagged(action)) is not None else None
        ok = parsed == (self._final, self._accept)
        return None, float(ok), True, False, {"score": float(ok)}

    @staticmethod
    def validate_options(options: dict) -> dict:
        states = _int_param(options, "states", default=10, min_value=4, max_value=64)
        length = _int_param(options, "length", default=16, min_value=1, max_value=256)
        alphabet = str(options.get("alphabet", "abc"))
        if not alphabet or len(set(alphabet)) != len(alphabet) or any(ch.isspace() for ch in alphabet):
            raise ValueError("alphabet must be non-empty with unique non-whitespace symbols")
        if len(alphabet) > 8:
            raise ValueError(f"alphabet must have at most 8 symbols, got {len(alphabet)}")
        accept_count = _int_param(options, "accept_count", default=3, min_value=1, max_value=states - 1)
        return {"states": states, "length": length, "alphabet": alphabet, "accept_count": accept_count}


def _int_param(options: dict, key: str, *, default: int, min_value: int, max_value: int) -> int:
    value = options.get(key, default)
    if isinstance(value, bool):
        raise ValueError(f"{key} must be an integer, got {value!r}")
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be an integer, got {value!r}") from exc
    if out != value and not isinstance(value, str):
        raise ValueError(f"{key} must be an integer, got {value!r}")
    if not min_value <= out <= max_value:
        raise ValueError(f"{key} must be in [{min_value}, {max_value}], got {out}")
    return out


NFA = NFATraceEnv
