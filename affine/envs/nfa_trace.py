import json
import random

from ._base import ExactAnswerEnv, Spec, int_param, parse_json_obj

_SPEC = Spec(
    title="In the following, we will test your ability to track a nondeterministic finite automaton.",
    rules=(
        "You will not be able to use external tools during this test",
        "The automaton has no epsilon transitions",
        "After each input character, the active state set is the union of all listed destinations",
        "Return the final active state set and whether any active state is accepting",
        "Only outputs wrapped in XML-style <ANSWER></ANSWER> tags will be evaluated",
        "You may write any text outside of the <ANSWER> tags",
    ),
    example_challenge=(
        "states: 0..3\n"
        "start: 0\n"
        "accepting: {2}\n"
        "input: ab\n"
        "transitions:\n"
        "0: a->{0,1} b->{1}\n"
        "1: a->{2} b->{3}\n"
        "2: a->{2} b->{}\n"
        "3: a->{} b->{2}"
    ),
    example_answer='{"reachable":[1,3],"accept":false}',
)


def _run(start: int, word: str, transitions: dict[tuple[int, str], tuple[int, ...]]) -> tuple[int, ...]:
    active = {start}
    for ch in word:
        active = {dst for state in active for dst in transitions[state, ch]}
    return tuple(sorted(active))


def _generate_nfa(states: int, length: int, alphabet: str, accept_count: int, rng: random.Random):
    for _ in range(1000):
        transitions = {}
        for state in range(states):
            for ch in alphabet:
                deg = rng.choices((0, 1, 2, 3), (1, 5, 3, 1))[0]
                transitions[state, ch] = tuple(sorted(rng.sample(range(states), deg)))
        start = rng.randrange(states)
        word = "".join(rng.choice(alphabet) for _ in range(length))
        final = _run(start, word, transitions)
        if 1 <= len(final) <= states - accept_count:
            break
    else:
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
            dst = ",".join(map(str, transitions[state, ch]))
            cols.append(f"{ch}->{{{dst}}}")
        rows.append(f"{state}: {' '.join(cols)}")
    prompt = (
        f"states: 0..{states - 1}\n"
        f"start: {start}\n"
        f"accepting: {{{','.join(map(str, sorted(accepting)))}}}\n"
        f"input: {word}\n"
        "transitions:\n" + "\n".join(rows)
    )
    accept = any(s in accepting for s in final)
    return prompt, final, accept


class NFATraceEnv(ExactAnswerEnv):
    env_id = "nfa_trace"
    option_keys = frozenset({"states", "length", "alphabet", "accept_count"})
    spec = _SPEC

    def _generate(self, params, rng):
        prompt, self._final, self._accept = _generate_nfa(
            params["states"], params["length"], params["alphabet"], params["accept_count"], rng
        )
        self._target = json.dumps({"reachable": list(self._final), "accept": self._accept}, separators=(",", ":"))
        return prompt, {}

    def parse_answer(self, body: str):
        obj = parse_json_obj(body)
        if not isinstance(obj, dict) or set(obj) != {"reachable", "accept"} or not isinstance(obj["accept"], bool):
            return None
        reachable = obj["reachable"]
        if not isinstance(reachable, list) or any(type(x) is not int for x in reachable):
            return None
        return tuple(sorted(reachable)), obj["accept"]

    @classmethod
    def _validate(cls, options: dict) -> dict:
        states = int_param(options, "states", default=10, lo=4, hi=64)
        length = int_param(options, "length", default=16, lo=1, hi=256)
        alphabet = str(options.get("alphabet", "abc"))
        if not alphabet or len(set(alphabet)) != len(alphabet) or any(ch.isspace() for ch in alphabet):
            raise ValueError("alphabet must be non-empty with unique non-whitespace symbols")
        if len(alphabet) > 8:
            raise ValueError(f"alphabet must have at most 8 symbols, got {len(alphabet)}")
        accept_count = int_param(options, "accept_count", default=3, lo=1, hi=states - 1)
        return {"states": states, "length": length, "alphabet": alphabet, "accept_count": accept_count}
