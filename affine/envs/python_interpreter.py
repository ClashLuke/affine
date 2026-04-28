import collections
import random

from ._base import ExactAnswerEnv, Spec, int_param

_all_ops = ("PRINT", "SORT", "APPEND", "ADD", "ASSIGN", "NOOP", "REVERSE", "POP",
            "ZIP", "EXTEND", "INSERT", "MIN", "MAX")

_SPEC = Spec(
    title="In the following, we will test your ability to understand and execute python code.",
    rules=(
        "You will not be able to use any external tools, such as a calculator or python during this test",
        "You will see the python code precisely once",
        "You may think and decypher the test for as long as you want",
        "Return the *exact* printout the python interpreter would otherwise return, including formatting and potential typos in the challenge",
        "Only outputs wrapped in XML-style <ANSWER></ANSWER> tags will be evaluated",
        "You may write any text outside of the <ANSWER> tags",
    ),
    example_challenge=(
        ">>> var1 = [62139]\n"
        ">>> var2 = [62598]\n"
        ">>> print(var1)\n"
        ">>> var4 = [40899]\n"
        ">>> var5 = sorted(var2)\n"
        ">>> var7 = 86093\n"
        ">>> var7 += 35501\n"
        ">>> print(var7)"
    ),
    example_answer="[62139]\n121594\n",
)


def codegen(op_count: int, allowed_ops, max_digits: int = 5, rng: random.Random | None = None):
    code: list[str] = []
    target: list[str] = []
    vars: dict[str, dict[str, object]] = collections.defaultdict(dict)
    rng = rng or random.Random()
    counter = 0

    def _randint():
        return rng.randint(0, 10 ** max_digits - 1)

    def _rand_key(name: str, default: bool = True):
        vals = list(vars[name].keys())
        if default:
            vals.append(None)
        return rng.choice(vals)

    def _print():
        flat = [(cat, name) for cat, items in vars.items() for name in items]
        if not flat:
            code.append("print()")
            target.append("\n")
            return
        cat, name = rng.choice(flat)
        code.append(f"print({name})")
        target.append(f"{vars[cat][name]}\n")

    def _sort():
        key = _rand_key("list")
        if key is None:
            return
        new = f"var{counter}"
        code.append(f"{new} = sorted({key})")
        vars["list"][new] = sorted(vars["list"][key])

    def _append():
        chosen_val_name = _rand_key("raw")
        chosen_list = _rand_key("list")
        if chosen_val_name is None:
            chosen_val = _randint()
            chosen_val_name = chosen_val
        else:
            chosen_val = vars["raw"][chosen_val_name]
        if chosen_list is None:
            new = f"var{counter}"
            vars["list"][new] = [chosen_val]
            code.append(f"{new} = [{chosen_val_name}]")
            return
        vars["list"][chosen_list].append(chosen_val)
        code.append(f"{chosen_list}.append({chosen_val_name})")

    def _add():
        chosen_val = _rand_key("raw")
        new = _randint()
        if chosen_val is None:
            chosen_val = f"var{counter}"
            code.append(f"{chosen_val} = {new}")
            vars["raw"][chosen_val] = new
            return
        vars["raw"][chosen_val] += new
        code.append(f"{chosen_val} += {new}")

    def _assign():
        chosen_val = _rand_key("raw")
        new = _randint()
        if chosen_val is None:
            chosen_val = f"var{counter}"
            code.append(f"{chosen_val} = {new}")
            vars["raw"][chosen_val] = new
            return
        vars["raw"][chosen_val] = new
        code.append(f"{chosen_val} = {new}")

    def _reverse():
        lst_key = _rand_key("list")
        if lst_key is None:
            return
        vars["list"][lst_key].reverse()
        code.append(f"{lst_key}.reverse()")

    def _pop():
        lst_key = _rand_key("list")
        if lst_key is None or not vars["list"][lst_key]:
            return
        vars["list"][lst_key].pop()
        code.append(f"{lst_key}.pop()")

    def _zip():
        a = _rand_key("list")
        b = _rand_key("list")
        if a is None or b is None:
            return
        new = f"var{counter}"
        merged = [x for z in zip(vars["list"][a], vars["list"][b]) for x in z]
        vars["list"][new] = merged
        code.append(f"{new} = [x for z in zip({a}, {b}) for x in z]")

    def _noop():
        if rng.random() < 0.5:
            code.append("")

    def _extend():
        lst1 = _rand_key("list")
        lst2 = _rand_key("list")
        if lst2 is None:
            lst2 = lst2_val = []
        else:
            lst2_val = vars["list"][lst2]
        if lst1 is None:
            new = f"var{counter}"
            vars["list"][new] = lst2_val[:]
            code.append(f"{new} = {lst2}[:]")
            return
        if lst2:
            vars["list"][lst1].extend(lst2_val)
        code.append(f"{lst1}.extend({lst2})")

    def _insert():
        lst = _rand_key("list")
        val = _randint()
        if lst is None:
            new = f"var{counter}"
            vars["list"][new] = [val]
            code.append(f"{new} = [{val}]")
            return
        idx = rng.randint(0, len(vars["list"][lst])) if vars["list"][lst] else 0
        vars["list"][lst].insert(idx, val)
        code.append(f"{lst}.insert({idx}, {val})")

    def _min():
        target_key = _rand_key("list")
        if target_key is None or not vars["list"][target_key]:
            return
        new = f"var{counter}"
        vars["raw"][new] = min(vars["list"][target_key])
        code.append(f"{new} = min({target_key})")

    def _max():
        target_key = _rand_key("list")
        if target_key is None or not vars["list"][target_key]:
            return
        new = f"var{counter}"
        vars["raw"][new] = max(vars["list"][target_key])
        code.append(f"{new} = max({target_key})")

    ops = {"PRINT": _print, "SORT": _sort, "APPEND": _append, "ADD": _add, "ASSIGN": _assign,
           "NOOP": _noop, "REVERSE": _reverse, "POP": _pop, "ZIP": _zip, "EXTEND": _extend,
           "INSERT": _insert, "MIN": _min, "MAX": _max}

    for _ in range(op_count):
        counter += 1
        ops[rng.choice(allowed_ops)]()
    _print()

    return "\n".join(f">>> {x}" for x in code), "".join(target)


class PythonInterpreterEnv(ExactAnswerEnv):
    env_id = "python_interpreter"
    option_keys = frozenset({"lines", "ops", "max_digits"})
    spec = _SPEC
    strip_answer = False

    def _generate(self, params, rng):
        code, self._target = codegen(params["lines"], list(params["ops"]), params["max_digits"], rng)
        return code, {}

    def parse_answer(self, body: str):
        return body

    @classmethod
    def _validate(cls, options: dict) -> dict:
        lines = int_param(options, "lines", default=64, lo=1, hi=256)
        max_digits = int_param(options, "max_digits", default=5, lo=1, hi=8)
        ops = options.get("ops", _all_ops)
        if isinstance(ops, str) or not isinstance(ops, (list, tuple)) or not ops:
            raise ValueError("ops must be a non-empty list")
        unknown = sorted({str(op) for op in ops} - set(_all_ops))
        if unknown:
            raise ValueError(f"unknown ops: {unknown}")
        return {"lines": lines, "ops": tuple(str(op) for op in ops), "max_digits": max_digits}
