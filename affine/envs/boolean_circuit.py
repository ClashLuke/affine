import json
import random

from ._base import ExactAnswerEnv, Spec, int_param, parse_json_obj

_SPEC = Spec(
    title="In the following, we will test your ability to count satisfying assignments of a Boolean circuit.",
    rules=(
        "You will not be able to use external tools during this test",
        "Variables range over all binary assignments independently",
        "Gates are evaluated in the order shown; each gate may use variables or earlier gates",
        "NOT has one input; AND, OR, and XOR have two inputs",
        "Return how many assignments make the final gate equal 1",
        "Only outputs wrapped in XML-style <ANSWER></ANSWER> tags will be evaluated",
        "The answer inside the tags must be a JSON object with key count",
    ),
    example_challenge=(
        "variables: x0, x1, x2\n"
        "g0 = x0 XOR x1\n"
        "g1 = g0 AND x2\n"
        "output: g1"
    ),
    example_answer='{"count":2}',
)


def _var_masks(n: int) -> tuple[int, ...]:
    masks = []
    for i in range(n):
        mask = 0
        for assignment in range(1 << n):
            if (assignment >> i) & 1:
                mask |= 1 << assignment
        masks.append(mask)
    return tuple(masks)


def _build(n_vars: int, gates: int, rng: random.Random):
    full = (1 << (1 << n_vars)) - 1
    masks = {f"x{i}": mask for i, mask in enumerate(_var_masks(n_vars))}
    depths = {name: 0 for name in masks}
    lines = [f"variables: {', '.join(f'x{i}' for i in range(n_vars))}"]
    names = list(masks)
    for i in range(gates):
        name = f"g{i}"
        op = rng.choices(("NOT", "AND", "OR", "XOR"), (1, 3, 3, 3))[0]
        if op == "NOT":
            a = rng.choice(names)
            masks[name] = full ^ masks[a]
            depths[name] = depths[a] + 1
            lines.append(f"{name} = NOT {a}")
        else:
            a, b = rng.sample(names, 2)
            if op == "AND":
                masks[name] = masks[a] & masks[b]
            elif op == "OR":
                masks[name] = masks[a] | masks[b]
            else:
                masks[name] = masks[a] ^ masks[b]
            depths[name] = max(depths[a], depths[b]) + 1
            lines.append(f"{name} = {a} {op} {b}")
        names.append(name)
    out = masks[f"g{gates - 1}"]
    return lines + [f"output: g{gates - 1}"], out.bit_count(), depths[f"g{gates - 1}"], out


def _fallback(n_vars: int, gates: int, min_influence: int, rng: random.Random):
    full = (1 << (1 << n_vars)) - 1
    masks = {f"x{i}": mask for i, mask in enumerate(_var_masks(n_vars))}
    lines = [f"variables: {', '.join(f'x{i}' for i in range(n_vars))}"]
    variables = rng.sample(range(n_vars), n_vars)

    def var(i: int) -> str:
        return f"x{variables[i]}"

    used = 0
    extra_count = rng.randint(0, min(2, max(0, min_influence - 1)))
    parity_count = min_influence - extra_count
    current_name = var(0)
    current_mask = masks[current_name]
    for i in range(1, parity_count):
        name = f"g{used}"
        current_mask ^= masks[var(i)]
        lines.append(f"{name} = {current_name} XOR {var(i)}")
        current_name = name
        used += 1
    for i in range(parity_count, min_influence):
        name = f"g{used}"
        op = rng.choice(("AND", "OR"))
        if op == "AND":
            current_mask &= masks[var(i)]
        else:
            current_mask |= masks[var(i)]
        lines.append(f"{name} = {current_name} {op} {var(i)}")
        current_name = name
        used += 1
    if used == 0:
        current_mask = full ^ current_mask
        current_name = "g0"
        lines.append(f"{current_name} = NOT {var(0)}")
        used = 1
    while used < gates:
        name = f"g{used}"
        current_mask = full ^ current_mask
        lines.append(f"{name} = NOT {current_name}")
        current_name = name
        used += 1
    return lines + [f"output: g{gates - 1}"], current_mask.bit_count(), gates, current_mask


def _influence(mask: int, n_vars: int) -> int:
    out = 0
    for i in range(n_vars):
        bit = 1 << i
        for assignment in range(1 << n_vars):
            if assignment & bit:
                continue
            if ((mask >> assignment) ^ (mask >> (assignment | bit))) & 1:
                out += 1
                break
    return out


def _generate_circuit(n_vars: int, gates: int, min_influence: int, rng: random.Random):
    total = 1 << n_vars
    attempts = 1000
    if gates <= min_influence or n_vars >= 11 or gates >= 48:
        attempts = 8
    elif n_vars >= 10 or gates >= 32:
        attempts = 64
    for _ in range(attempts):
        lines, count, depth, mask = _build(n_vars, gates, rng)
        influence = _influence(mask, n_vars)
        if total // 8 <= count <= total * 7 // 8 and depth >= 4 and influence >= min_influence:
            return lines, count, influence, depth
    lines, count, depth, mask = _fallback(n_vars, gates, min_influence, rng)
    return lines, count, _influence(mask, n_vars), depth


class BooleanCircuitEnv(ExactAnswerEnv):
    env_id = "boolean_circuit"
    option_keys = frozenset({"variables", "gates", "min_influence"})
    spec = _SPEC

    def _generate(self, params, rng):
        lines, self._count, self._influence, self._depth = _generate_circuit(
            params["variables"], params["gates"], params["min_influence"], rng
        )
        self._target = json.dumps({"count": self._count}, separators=(",", ":"))
        return "\n".join(lines), {"influence": self._influence, "depth": self._depth}

    def parse_answer(self, body: str):
        obj = parse_json_obj(body)
        if not isinstance(obj, dict) or set(obj) != {"count"} or type(obj["count"]) is not int:
            return None
        return obj["count"]

    @classmethod
    def _validate(cls, options: dict) -> dict:
        variables = int_param(options, "variables", default=9, lo=3, hi=12)
        gates = int_param(options, "gates", default=18, lo=6, hi=64)
        min_influence = int_param(options, "min_influence", default=7, lo=1, hi=min(variables - 1, gates + 1))
        return {"variables": variables, "gates": gates, "min_influence": min_influence}
