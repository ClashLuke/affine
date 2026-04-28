import json
import math
import random
import re

from ._base import Env

static = ("In the following, we will test your ability to execute modular arithmetic and combine residues.\n\n"
          "RULES:\n"
          "* You will not be able to use external tools during this test\n"
          "* inverse(y) means the multiplicative inverse of y modulo the modulus for that block\n"
          "* Every inverse operand is guaranteed to be invertible\n"
          "* All residues are canonical integers in [0, modulus)\n"
          "* Return the final residue for each modulus and the unique CRT solution in [0, product)\n"
          "* Only outputs wrapped in XML-style <ANSWER></ANSWER> tags will be evaluated\n"
          "* The answer inside the tags must be JSON: {\"residues\":{\"m\":r,...},\"crt\":x}\n\n"
          "Example:\n"
          "CHALLENGE:\n"
          "mod 11: x0 = 7\n"
          "1. x = (3*x + 4) mod 11\n"
          "2. x = inverse((5*x + 2) mod 11) mod 11\n"
          "mod 13: x0 = 9\n"
          "1. x = (2*x + 8) mod 13\n"
          "2. x = (pow((x + 3) mod 13, 4, 13) + 1) mod 13\n"
          "RESPONSE:\n"
          "<ANSWER>{\"residues\":{\"11\":2,\"13\":4},\"crt\":134}</ANSWER>\n\n"
          "Below, you will see the real task. Remember and follow the rules.\n\n"
          "CHALLENGE:\n")

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
            lines.append(f"{i}. x = ({a}*x + {b}) mod {mod}")
        elif op == "inverse":
            a = rng.randrange(1, mod)
            b = rng.randrange(mod)
            while (a * x + b) % mod == 0:
                b = rng.randrange(mod)
            x = pow((a * x + b) % mod, -1, mod)
            lines.append(f"{i}. x = inverse(({a}*x + {b}) mod {mod}) mod {mod}")
        else:
            a, b, exp = rng.randrange(mod), rng.randrange(mod), rng.randint(3, 7)
            x = (pow((x + a) % mod, exp, mod) + b) % mod
            lines.append(f"{i}. x = (pow((x + {a}) mod {mod}, {exp}, {mod}) + {b}) mod {mod}")
    return lines, x


def generate(moduli: int, steps: int, rng: random.Random):
    mods = sorted(rng.sample(_primes, moduli))
    lines = []
    residues = {}
    for mod in mods:
        block, residue = _block(mod, steps, rng)
        lines.extend(block)
        residues[mod] = residue
    crt = _crt(residues)
    target = json.dumps({
        "residues": {str(mod): residues[mod] for mod in mods},
        "crt": crt,
    }, separators=(",", ":"))
    return '\n'.join(lines), residues, crt, target


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
    if not isinstance(obj, dict) or set(obj) != {"residues", "crt"} or type(obj["crt"]) is not int:
        return None
    residues = obj["residues"]
    if not isinstance(residues, dict):
        return None
    out = {}
    for mod, residue in residues.items():
        if not isinstance(mod, str) or not mod.isdigit() or type(residue) is not int:
            return None
        parsed = int(mod)
        if str(parsed) != mod or parsed in out:
            return None
        out[parsed] = residue
    return out, obj["crt"]


class ModularCRTEnv(Env):
    __version__: str = "0.0.1"

    def __init__(self, moduli: int = 3, steps: int = 5):
        self._defaults = self.validate_options({"moduli": moduli, "steps": steps})

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        opts = options or {}
        params = self.validate_options({**self._defaults, **opts})
        prompt, self._residues, self._crt, self._target = generate(
            params["moduli"], params["steps"], random.Random(0 if seed is None else seed)
        )
        return static + prompt, {
            "challenge_id": str(seed if seed is not None else 0),
            "env_id": "modular_crt",
            "spec_version": self.__version__,
            **params,
            "product": math.prod(self._residues),
        }

    def step(self, action: str):
        parsed = _parse(body) if (body := _tagged(action)) is not None else None
        ok = parsed == (self._residues, self._crt) and 0 <= self._crt < math.prod(self._residues)
        return None, float(ok), True, False, {"score": float(ok)}

    @staticmethod
    def validate_options(options: dict) -> dict:
        return {
            "moduli": _int_param(options, "moduli", default=3, min_value=1, max_value=len(_primes)),
            "steps": _int_param(options, "steps", default=5, min_value=1, max_value=64),
        }


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


CRT = ModularCRTEnv
