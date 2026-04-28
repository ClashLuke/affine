import json
import random
import re

from ._base import Env

static = ("In the following, we will test your ability to solve a Sudoku by constraint propagation.\n\n"
          "RULES:\n"
          "* You will not be able to use external tools during this test\n"
          "* 0 denotes an empty cell\n"
          "* Fill the grid so every row, column, and 3x3 box contains digits 1 through 9 exactly once\n"
          "* The puzzle has a unique solution\n"
          "* Only outputs wrapped in XML-style <ANSWER></ANSWER> tags will be evaluated\n"
          "* The answer inside the tags must be JSON: {\"grid\":[\"123456789\",...]} with 9 row strings\n\n"
          "Example:\n"
          "CHALLENGE:\n"
          "530070000\n"
          "600195000\n"
          "098000060\n"
          "800060003\n"
          "400803001\n"
          "700020006\n"
          "060000280\n"
          "000419005\n"
          "000080079\n"
          "RESPONSE:\n"
          "<ANSWER>{\"grid\":[\"534678912\",\"672195348\",\"198342567\",\"859761423\",\"426853791\",\"713924856\",\"961537284\",\"287419635\",\"345286179\"]}</ANSWER>\n\n"
          "Below, you will see the real task. Remember and follow the rules.\n\n"
          "CHALLENGE:\n")

ALL = (1 << 9) - 1


def _box(r: int, c: int) -> int:
    return (r // 3) * 3 + c // 3


def _complete_grid(rng: random.Random):
    rows = [band * 3 + row for band in rng.sample(range(3), 3) for row in rng.sample(range(3), 3)]
    cols = [stack * 3 + col for stack in rng.sample(range(3), 3) for col in rng.sample(range(3), 3)]
    digits = rng.sample(range(1, 10), 9)
    return [[digits[(r * 3 + r // 3 + c) % 9] for c in cols] for r in rows]


def _solve(grid: list[list[int]], limit: int = 2):
    rows, cols, boxes, empties = [0] * 9, [0] * 9, [0] * 9, []
    for r in range(9):
        for c in range(9):
            v = grid[r][c]
            if v:
                bit = 1 << (v - 1)
                rows[r] |= bit
                cols[c] |= bit
                boxes[_box(r, c)] |= bit
            else:
                empties.append((r, c))
    solution, stats = None, {"nodes": 0, "branch_points": 0}

    def search():
        nonlocal solution
        if solution is not None and stats["count"] >= limit:
            return
        if not empties:
            stats["count"] += 1
            if solution is None:
                solution = [row[:] for row in grid]
            return
        best_i, best_mask = -1, 0
        for i, (r, c) in enumerate(empties):
            mask = ALL & ~(rows[r] | cols[c] | boxes[_box(r, c)])
            if mask == 0:
                return
            if best_i < 0 or mask.bit_count() < best_mask.bit_count():
                best_i, best_mask = i, mask
                if best_mask.bit_count() == 1:
                    break
        if best_mask.bit_count() > 1:
            stats["branch_points"] += 1
        r, c = empties.pop(best_i)
        b = _box(r, c)
        mask = best_mask
        while mask and stats["count"] < limit:
            bit = mask & -mask
            mask ^= bit
            v = bit.bit_length()
            grid[r][c] = v
            rows[r] |= bit
            cols[c] |= bit
            boxes[b] |= bit
            stats["nodes"] += 1
            search()
            boxes[b] ^= bit
            cols[c] ^= bit
            rows[r] ^= bit
            grid[r][c] = 0
        empties.insert(best_i, (r, c))

    stats["count"] = 0
    search()
    return stats["count"], stats, solution


def generate(clues: int, min_branch_points: int, rng: random.Random):
    last = None
    found = False
    for _ in range(256):
        solution = _complete_grid(rng)
        puzzle = [row[:] for row in solution]
        positions = [(r, c) for r in range(9) for c in range(9)]
        rng.shuffle(positions)
        left = 81
        for r, c in positions:
            if left <= clues:
                break
            old = puzzle[r][c]
            puzzle[r][c] = 0
            count, _, _ = _solve([row[:] for row in puzzle], 2)
            if count == 1:
                left -= 1
            else:
                puzzle[r][c] = old
        count, stats, solved = _solve([row[:] for row in puzzle], 2)
        last = puzzle, solved or solution, left, stats
        if count == 1 and left <= clues + 2 and stats["branch_points"] >= min_branch_points:
            found = True
            break
    if not found:
        raise ValueError("could not generate Sudoku with requested constraints")
    puzzle, solution, _, stats = last
    prompt = '\n'.join(''.join(str(x) for x in row) for row in puzzle)
    rows = [''.join(str(x) for x in row) for row in solution]
    return prompt, rows, stats["branch_points"], json.dumps({"grid": rows}, separators=(",", ":"))


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
    if not isinstance(obj, dict) or set(obj) != {"grid"}:
        return None
    grid = obj["grid"]
    if not isinstance(grid, list) or len(grid) != 9:
        return None
    if any(not isinstance(row, str) or re.fullmatch(r"[1-9]{9}", row) is None for row in grid):
        return None
    return grid


class SudokuEnv(Env):
    __version__: str = "0.0.1"

    def __init__(self, clues: int = 36, min_branch_points: int = 2):
        self._defaults = self.validate_options({"clues": clues, "min_branch_points": min_branch_points})
        self.clues = self._defaults["clues"]
        self.min_branch_points = self._defaults["min_branch_points"]

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        opts = options or {}
        params = self.validate_options({**self._defaults, **opts})
        prompt, self._grid, self._branch_points, self._target = generate(
            params["clues"], params["min_branch_points"], random.Random(0 if seed is None else seed)
        )
        return static + prompt, {
            "challenge_id": str(seed if seed is not None else 0),
            "env_id": "sudoku",
            "spec_version": self.__version__,
            **params,
            "branch_points": self._branch_points,
        }

    def step(self, action: str):
        parsed = _parse(body) if (body := _tagged(action)) is not None else None
        ok = parsed == self._grid
        return None, float(ok), True, False, {"score": float(ok)}

    @staticmethod
    def validate_options(options: dict) -> dict:
        clues = _int_param(options, "clues", default=36, min_value=24, max_value=45)
        min_branch_points = _int_param(options, "min_branch_points", default=2, min_value=0, max_value=4)
        max_clues_by_branch = (45, 38, 36, 35, 34)
        max_clues = max_clues_by_branch[min_branch_points]
        if clues > max_clues:
            raise ValueError(
                f"clues must be <= {max_clues} when min_branch_points={min_branch_points}, got {clues}"
            )
        return {"clues": clues, "min_branch_points": min_branch_points}


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


SDK = SudokuEnv
