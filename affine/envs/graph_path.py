import heapq
import json
import random

from ._base import ExactAnswerEnv, Spec, int_param, parse_json_obj

_SPEC = Spec(
    title="In the following, we will test your ability to find a shortest path in a weighted directed graph.",
    rules=(
        "You will not be able to use external tools during this test",
        "All edge weights are positive integers",
        "If more than one shortest path exists, return the lexicographically smallest node sequence",
        "Lexicographic order compares integer node sequences from left to right",
        "Only outputs wrapped in XML-style <ANSWER></ANSWER> tags will be evaluated",
        "The answer inside the tags must be a JSON object with keys distance and path",
    ),
    example_challenge=(
        "nodes: 0..4\n"
        "source: 0\n"
        "target: 4\n"
        "edges:\n"
        "0 -> 1 (4)\n"
        "0 -> 2 (2)\n"
        "1 -> 4 (4)\n"
        "2 -> 3 (2)\n"
        "3 -> 4 (3)"
    ),
    example_answer='{"distance":7,"path":[0,2,3,4]}',
)


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


def _generate_graph(n: int, edge_count: int, min_path_len: int, rng: random.Random):
    source, target = 0, n - 1
    wanted = rng.randint(min_path_len, min(n, min_path_len + 3, edge_count + 1))
    chosen: dict[tuple[int, int], int] = {}
    for _ in range(1000):
        order = [source] + rng.sample(range(1, n - 1), n - 2) + [target]
        chosen = {}
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
            break
    else:
        raise ValueError("could not generate graph with requested shortest-path length")

    rows = [f"{u} -> {v} ({w})" for u, v, w in sorted((u, v, w) for (u, v), w in chosen.items())]
    prompt = (
        f"nodes: 0..{n - 1}\n"
        f"source: {source}\n"
        f"target: {target}\n"
        "edges:\n" + "\n".join(rows)
    )
    dist, path = _shortest(n, edges, source, target)
    return prompt, dist, path, len(chosen)


class GraphPathEnv(ExactAnswerEnv):
    env_id = "graph_path"
    option_keys = frozenset({"nodes", "edges", "min_path_len"})
    spec = _SPEC

    def _generate(self, params, rng):
        prompt, self._distance, self._path, self._edge_count = _generate_graph(
            params["nodes"], params["edges"], params["min_path_len"], rng
        )
        self._target = json.dumps({"distance": self._distance, "path": list(self._path)}, separators=(",", ":"))
        return prompt, {"actual_edges": self._edge_count}

    def parse_answer(self, body: str):
        obj = parse_json_obj(body)
        if not isinstance(obj, dict) or set(obj) != {"distance", "path"} or type(obj["distance"]) is not int:
            return None
        path = obj["path"]
        if not isinstance(path, list) or any(type(x) is not int for x in path):
            return None
        return obj["distance"], tuple(path)

    @classmethod
    def _validate(cls, options: dict) -> dict:
        nodes = int_param(options, "nodes", default=16, lo=3, hi=64)
        min_path_len = int_param(options, "min_path_len", default=5, lo=2, hi=nodes)
        max_edges = nodes * (nodes - 1) // 2
        edges = int_param(options, "edges", default=46, lo=min_path_len - 1, hi=max_edges)
        return {"nodes": nodes, "edges": edges, "min_path_len": min_path_len}
