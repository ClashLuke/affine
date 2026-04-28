import json
import random
import re

from ._base import Env

static = ("In the following, we will test your ability to count satisfying assignments of a Boolean circuit.\n\n"
          "RULES:\n"
          "* You will not be able to use external tools during this test\n"
          "* Variables range over all binary assignments independently\n"
          "* Gates are evaluated in the order shown; each gate may use variables or earlier gates\n"
          "* NOT has one input; AND, OR, and XOR have two inputs\n"
          "* Return how many assignments make the final gate equal 1\n"
          "* Only outputs wrapped in XML-style <ANSWER></ANSWER> tags will be evaluated\n"
          "* The answer inside the tags must be a JSON object with key count\n\n"
          "Example:\n"
          "CHALLENGE:\n"
          "variables: x0, x1, x2\n"
          "g0 = x0 XOR x1\n"
          "g1 = g0 AND x2\n"
          "output: g1\n"
          "RESPONSE:\n"
          "<ANSWER>{\"count\":2}</ANSWER>\n\n"
          "Below, you will see the real task. Remember and follow the rules.\n\n"
          "CHALLENGE:\n")


def _var_masks(n: int) -> tuple[int, ...]:
    masks = []
    for i in range(n):
        mask = 0
        for assignment in range(1 << n):
            if (assignment >> i) & 1:
                mask |= 1 << assignment
        masks.append(mask)
    return tuple(masks)


def _depth(expr: str, depths: dict[str, int]) -> int:
    return depths.get(expr, 0)


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
            depths[name] = _depth(a, depths) + 1
            lines.append(f"{name} = NOT {a}")
        else:
            a, b = rng.sample(names, 2)
            if op == "AND":
                masks[name] = masks[a] & masks[b]
            elif op == "OR":
                masks[name] = masks[a] | masks[b]
            else:
                masks[name] = masks[a] ^ masks[b]
            depths[name] = max(_depth(a, depths), _depth(b, depths)) + 1
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


def generate(n_vars: int, gates: int, min_influence: int, rng: random.Random):
    total = 1 << n_vars
    influence = 0
    found = False
    attempts = 1000
    if gates <= min_influence or n_vars >= 11 or gates >= 48:
        attempts = 8
    elif n_vars >= 10 or gates >= 32:
        attempts = 64
    for _ in range(attempts):
        lines, count, depth, mask = _build(n_vars, gates, rng)
        influence = _influence(mask, n_vars)
        if total // 8 <= count <= total * 7 // 8 and depth >= 4 and influence >= min_influence:
            found = True
            break
    if not found:
        lines, count, depth, mask = _fallback(n_vars, gates, min_influence, rng)
        influence = _influence(mask, n_vars)
    target = json.dumps({"count": count}, separators=(",", ":"))
    return '\n'.join(lines), count, influence, depth, target


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
    if not isinstance(obj, dict) or set(obj) != {"count"} or type(obj["count"]) is not int:
        return None
    return obj["count"]


class BooleanCircuitEnv(Env):
    __version__: str = "0.0.1"

    def __init__(self, variables: int = 9, gates: int = 18, min_influence: int = 7):
        self._defaults = self.validate_options({
            "variables": variables, "gates": gates, "min_influence": min_influence,
        })
        self.variables = self._defaults["variables"]
        self.gates = self._defaults["gates"]
        self.min_influence = self._defaults["min_influence"]

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        opts = options or {}
        params = self.validate_options({**self._defaults, **opts})
        prompt, self._count, self._influence, self._depth, self._target = generate(
            params["variables"], params["gates"], params["min_influence"], random.Random(0 if seed is None else seed)
        )
        return static + prompt, {
            "challenge_id": str(seed if seed is not None else 0),
            "env_id": "boolean_circuit",
            "spec_version": self.__version__,
            **params,
            "influence": self._influence,
            "depth": self._depth,
        }

    def step(self, action: str):
        parsed = _parse(body) if (body := _tagged(action)) is not None else None
        ok = parsed == self._count
        return None, float(ok), True, False, {"score": float(ok)}

    @staticmethod
    def validate_options(options: dict) -> dict:
        variables = _int_param(options, "variables", default=9, min_value=3, max_value=12)
        gates = _int_param(options, "gates", default=18, min_value=6, max_value=64)
        min_influence = _int_param(options, "min_influence", default=7, min_value=1, max_value=min(variables - 1, gates + 1))
        return {"variables": variables, "gates": gates, "min_influence": min_influence}


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


BC = BooleanCircuitEnv
