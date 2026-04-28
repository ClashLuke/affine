import json
import math
import random

from ._base import ExactAnswerEnv, Spec, int_param, parse_json_obj

_SPEC = Spec(
    title="In the following, we will test your ability to execute modular arithmetic and combine residues.",
    rules=(
        "You will not be able to use external tools during this test",
        "All expressions are valid Python; % is the modulo operator and pow(b, e, m) is modular exponentiation",
        "inverse(y) is the multiplicative inverse of y modulo the modulus for that block",
        "Every inverse operand is guaranteed to be invertible",
        "All residues are canonical integers in [0, modulus)",
        "Return the final residue for each modulus and the unique CRT solution in [0, product)",
        "Only outputs wrapped in XML-style <ANSWER></ANSWER> tags will be evaluated",
        'The answer inside the tags must be JSON: {"residues":{"m":r,...},"crt":x}',
    ),
    example_challenge=(
        "mod 11: x0 = 7\n"
        "1. x = (3*x + 4) % 11\n"
        "2. x = inverse((5*x + 2) % 11) % 11\n"
        "mod 13: x0 = 9\n"
        "1. x = (2*x + 8) % 13\n"
        "2. x = (pow((x + 3) % 13, 4, 13) + 1) % 13"
    ),
    example_answer='{"residues":{"11":2,"13":4},"crt":134}',
)

_primes = (43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97)


def _crt(residues: dict[int, int]) -> int:
    total = math.prod(residues)
    out = 0
    for mod, residue in residues.items():
        partial = total // mod
        out += residue * partial * pow(partial, -1, mod)
    return out % total


def _block(mod: int, steps: int, rng: random.Random):
    x = rng.randrange(1, mod)
    lines = [f"mod {mod}: x0 = {x}"]
    for i in range(1, steps + 1):
        op = rng.choice(("affine", "inverse", "pow"))
        if op == "affine":
            a, b = rng.randrange(2, mod), rng.randrange(mod)
            x = (a * x + b) % mod
            lines.append(f"{i}. x = ({a}*x + {b}) % {mod}")
        elif op == "inverse":
            a = rng.randrange(1, mod)
            forbidden = (-a * x) % mod
            b = rng.randrange(mod - 1)
            if b >= forbidden:
                b += 1
            x = pow((a * x + b) % mod, -1, mod)
            lines.append(f"{i}. x = inverse(({a}*x + {b}) % {mod}) % {mod}")
        else:
            a, b, exp = rng.randrange(mod), rng.randrange(mod), rng.randint(3, 7)
            x = (pow((x + a) % mod, exp, mod) + b) % mod
            lines.append(f"{i}. x = (pow((x + {a}) % {mod}, {exp}, {mod}) + {b}) % {mod}")
    return lines, x


class ModularCRTEnv(ExactAnswerEnv):
    env_id = "modular_crt"
    option_keys = frozenset({"moduli", "steps"})
    spec = _SPEC

    def _generate(self, params, rng):
        mods = sorted(rng.sample(_primes, params["moduli"]))
        lines: list[str] = []
        residues: dict[int, int] = {}
        for mod in mods:
            block, residue = _block(mod, params["steps"], rng)
            lines.extend(block)
            residues[mod] = residue
        self._residues = residues
        self._crt = _crt(residues)
        self._target = json.dumps(
            {"residues": {str(mod): residues[mod] for mod in mods}, "crt": self._crt},
            separators=(",", ":"),
        )
        return "\n".join(lines), {"product": math.prod(mods)}

    def parse_answer(self, body: str):
        obj = parse_json_obj(body)
        if not isinstance(obj, dict) or set(obj) != {"residues", "crt"} or type(obj["crt"]) is not int:
            return None
        residues = obj["residues"]
        if not isinstance(residues, dict):
            return None
        out: dict[int, int] = {}
        for mod, residue in residues.items():
            if not isinstance(mod, str) or not mod.isdigit() or type(residue) is not int:
                return None
            parsed = int(mod)
            if str(parsed) != mod or parsed in out:
                return None
            out[parsed] = residue
        return out, obj["crt"]

    @classmethod
    def _validate(cls, options: dict) -> dict:
        return {
            "moduli": int_param(options, "moduli", default=3, lo=1, hi=len(_primes)),
            "steps": int_param(options, "steps", default=5, lo=1, hi=64),
        }
