import json
import random
import re

from ._base import ExactAnswerEnv, Spec, int_param, parse_json_obj

_SPEC = Spec(
    title="In the following, we will test your ability to solve a Sudoku by constraint propagation.",
    rules=(
        "You will not be able to use external tools during this test",
        "0 denotes an empty cell",
        "Fill the grid so every row, column, and 3x3 box contains digits 1 through 9 exactly once",
        "The puzzle has a unique solution",
        "Only outputs wrapped in XML-style <ANSWER></ANSWER> tags will be evaluated",
        'The answer inside the tags must be JSON: {"grid":["123456789",...]} with 9 row strings',
    ),
    example_challenge=(
        "530070000\n"
        "600195000\n"
        "098000060\n"
        "800060003\n"
        "400803001\n"
        "700020006\n"
        "060000280\n"
        "000419005\n"
        "000080079"
    ),
    example_answer=(
        '{"grid":["534678912","672195348","198342567","859761423","426853791",'
        '"713924856","961537284","287419635","345286179"]}'
    ),
)

ALL = (1 << 9) - 1
_ROW_RE = re.compile(r"[1-9]{9}")


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
                if rows[r] & bit or cols[c] & bit or boxes[_box(r, c)] & bit:
                    return 0, {"nodes": 0, "branch_points": 0, "count": 0}, None
                rows[r] |= bit
                cols[c] |= bit
                boxes[_box(r, c)] |= bit
            else:
                empties.append((r, c))
    solution, stats = None, {"nodes": 0, "branch_points": 0, "count": 0}

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

    search()
    return stats["count"], stats, solution


def _generate_sudoku(clues: int, min_branch_points: int, rng: random.Random):
    last = None
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
            puzzle, solution, _, stats = last
            prompt = "\n".join("".join(str(x) for x in row) for row in puzzle)
            rows = ["".join(str(x) for x in row) for row in solution]
            return prompt, rows, stats["branch_points"]
    raise ValueError("could not generate Sudoku with requested constraints")


class SudokuEnv(ExactAnswerEnv):
    env_id = "sudoku"
    option_keys = frozenset({"clues", "min_branch_points"})
    spec = _SPEC

    def __init__(self, clues: int = 36, min_branch_points: int = 2):
        self._defaults = self.validate_options({"clues": clues, "min_branch_points": min_branch_points})
        self.clues = self._defaults["clues"]
        self.min_branch_points = self._defaults["min_branch_points"]

    def _generate(self, params, rng):
        prompt, self._grid, self._branch_points = _generate_sudoku(
            params["clues"], params["min_branch_points"], rng
        )
        self._target = json.dumps({"grid": self._grid}, separators=(",", ":"))
        return prompt, {"branch_points": self._branch_points}

    def parse_answer(self, body: str):
        obj = parse_json_obj(body)
        if not isinstance(obj, dict) or set(obj) != {"grid"}:
            return None
        grid = obj["grid"]
        if not isinstance(grid, list) or len(grid) != 9:
            return None
        if any(not isinstance(row, str) or _ROW_RE.fullmatch(row) is None for row in grid):
            return None
        return grid

    @classmethod
    def validate_options(cls, options: dict) -> dict:
        clues = int_param(options, "clues", default=36, lo=24, hi=45)
        min_branch_points = int_param(options, "min_branch_points", default=2, lo=0, hi=4)
        max_clues = (45, 38, 36, 35, 34)[min_branch_points]
        if clues > max_clues:
            raise ValueError(
                f"clues must be <= {max_clues} when min_branch_points={min_branch_points}, got {clues}"
            )
        return {"clues": clues, "min_branch_points": min_branch_points}
