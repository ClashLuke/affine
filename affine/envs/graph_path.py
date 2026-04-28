import heapq
import json
import random
import re

from ._base import Env

static = ("In the following, we will test your ability to find a shortest path in a weighted directed graph.\n\n"
          "RULES:\n"
          "* You will not be able to use external tools during this test\n"
          "* All edge weights are positive integers\n"
          "* If more than one shortest path exists, return the lexicographically smallest node sequence\n"
          "* Lexicographic order compares integer node sequences from left to right\n"
          "* Only outputs wrapped in XML-style <ANSWER></ANSWER> tags will be evaluated\n"
          "* The answer inside the tags must be a JSON object with keys distance and path\n\n"
          "Example:\n"
          "CHALLENGE:\n"
          "nodes: 0..4\n"
          "source: 0\n"
          "target: 4\n"
          "edges:\n"
          "0 -> 1 (4)\n"
          "0 -> 2 (2)\n"
          "1 -> 4 (4)\n"
          "2 -> 3 (2)\n"
          "3 -> 4 (3)\n"
          "RESPONSE:\n"
          "<ANSWER>{\"distance\":7,\"path\":[0,2,3,4]}</ANSWER>\n\n"
          "Below, you will see the real task. Remember and follow the rules.\n\n"
          "CHALLENGE:\n")


def _shortest(n: int, edges: dict[int, list[tuple[int, int]]], source: int, target: int):
    heap = [(0, (source,), source)]
    best = {source: (0, (source,))}
    while heap:
        dist, path, node = heapq.heappop(heap)
        if best.get(node) != (dist, path):
            continue
        if node == target:
            return dist, path
        for nxt, weight in edges[node]:
            cand = (dist + weight, path + (nxt,))
            if cand < best.get(nxt, (10**18, ())):
                best[nxt] = cand
                heapq.heappush(heap, (cand[0], cand[1], nxt))
    return None


def generate(n: int, edge_count: int, min_path_len: int, rng: random.Random):
    source, target = 0, n - 1
    wanted = rng.randint(min_path_len, min(n, min_path_len + 3, edge_count + 1))
    found = False
    for _ in range(1000):
        order = [source] + rng.sample(range(1, n - 1), n - 2) + [target]
        chosen: dict[tuple[int, int], int] = {}
        planted = [source] + rng.sample(range(1, n - 1), wanted - 2) + [target]
        for a, b in zip(planted, planted[1:]):
            chosen[a, b] = rng.randint(2, 5)
        pairs = [(order[i], order[j]) for i in range(n) for j in range(i + 1, n)]
        rng.shuffle(pairs)
        planted_idx = {node: i for i, node in enumerate(planted)}
        for u, v in pairs:
            if len(chosen) >= edge_count:
                break
            if u in planted_idx and v in planted_idx and planted_idx[v] > planted_idx[u] + 1:
                segment = sum(chosen[planted[i], planted[i + 1]] for i in range(planted_idx[u], planted_idx[v]))
                chosen.setdefault((u, v), segment + rng.randint(1, 8))
            else:
                chosen.setdefault((u, v), 1000 + rng.randint(0, 1000))
        edges = {i: [] for i in range(n)}
        for (u, v), w in chosen.items():
            edges[u].append((v, w))
        for row in edges.values():
            row.sort()
        result = _shortest(n, edges, source, target)
        if result is not None and wanted <= len(result[1]) <= wanted + 2:
            found = True
            break
    if not found:
        raise ValueError("could not generate graph with requested shortest-path length")

    rows = [f"{u} -> {v} ({w})" for u, v, w in sorted((u, v, w) for (u, v), w in chosen.items())]
    prompt = (
        f"nodes: 0..{n - 1}\n"
        f"source: {source}\n"
        f"target: {target}\n"
        "edges:\n" + '\n'.join(rows)
    )
    dist, path = _shortest(n, edges, source, target)
    target_json = json.dumps({"distance": dist, "path": list(path)}, separators=(",", ":"))
    return prompt, dist, path, target_json, len(chosen)


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
    if not isinstance(obj, dict) or set(obj) != {"distance", "path"} or type(obj["distance"]) is not int:
        return None
    path = obj["path"]
    if not isinstance(path, list) or any(type(x) is not int for x in path):
        return None
    return obj["distance"], tuple(path)


class GraphPathEnv(Env):
    __version__: str = "0.0.1"

    def __init__(self, nodes: int = 16, edges: int = 46, min_path_len: int = 5):
        self._defaults = self.validate_options({"nodes": nodes, "edges": edges, "min_path_len": min_path_len})

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        opts = options or {}
        params = self.validate_options({**self._defaults, **opts})
        prompt, self._distance, self._path, self._target, edge_count = generate(
            params["nodes"], params["edges"], params["min_path_len"], random.Random(0 if seed is None else seed)
        )
        self._edge_count = edge_count
        return static + prompt, {
            "challenge_id": str(seed if seed is not None else 0),
            "env_id": "graph_path",
            "spec_version": self.__version__,
            **params,
            "actual_edges": edge_count,
        }

    def step(self, action: str):
        parsed = _parse(body) if (body := _tagged(action)) is not None else None
        ok = parsed == (self._distance, self._path)
        return None, float(ok), True, False, {"score": float(ok)}

    @staticmethod
    def validate_options(options: dict) -> dict:
        nodes = _int_param(options, "nodes", default=16, min_value=3, max_value=64)
        min_path_len = _int_param(options, "min_path_len", default=5, min_value=2, max_value=nodes)
        max_edges = nodes * (nodes - 1) // 2
        edges = _int_param(options, "edges", default=46, min_value=min_path_len - 1, max_value=max_edges)
        return {"nodes": nodes, "edges": edges, "min_path_len": min_path_len}


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


GP = GraphPathEnv
