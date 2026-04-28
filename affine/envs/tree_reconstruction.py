from __future__ import annotations

import heapq
import math
import random
import re
from dataclasses import dataclass
from typing import Any

from ._base import Env


@dataclass(frozen=True)
class QueryResult:
    value: Any
    bits: float


class HiddenTree:
    def __init__(self, n: int, seed: int, method: str = "prufer"):
        if n < 2:
            raise ValueError(f"n must be at least 2, got {n}")
        self.n = n
        self.seed = seed
        self.method = method
        rng = random.Random(seed)
        if method == "prufer":
            self.parent = self._from_prufer([rng.randrange(n) for _ in range(n - 2)])
        elif method == "recursive":
            self.parent = [-1] + [rng.randrange(i) for i in range(1, n)]
        else:
            raise ValueError(f"unknown tree generation method: {method!r}")
        self._precompute()

    def _from_prufer(self, seq: list[int]) -> list[int]:
        n = len(seq) + 2
        degree = [1] * n
        for v in seq:
            degree[v] += 1
        leaves = [i for i, d in enumerate(degree) if d == 1]
        heapq.heapify(leaves)
        edges: list[tuple[int, int]] = []
        for v in seq:
            leaf = heapq.heappop(leaves)
            edges.append((leaf, v))
            degree[leaf] -= 1
            degree[v] -= 1
            if degree[v] == 1:
                heapq.heappush(leaves, v)
        edges.append((heapq.heappop(leaves), heapq.heappop(leaves)))
        return self._root(edges)

    def _root(self, edges: list[tuple[int, int]]) -> list[int]:
        adj = [[] for _ in range(len(edges) + 1)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        parent = [-1] * len(adj)
        queue = [0]
        for u in queue:
            for v in adj[u]:
                if v != parent[u]:
                    parent[v] = u
                    queue.append(v)
        return parent

    def _precompute(self) -> None:
        n = self.n
        self.children = [[] for _ in range(n)]
        for v, p in enumerate(self.parent[1:], start=1):
            self.children[p].append(v)

        self.depth = [0] * n
        self._tin = [0] * n
        self._tout = [0] * n
        t = 0
        stack = [(0, False)]
        while stack:
            u, done = stack.pop()
            if done:
                self._tout[u] = t
                t += 1
                continue
            self._tin[u] = t
            t += 1
            stack.append((u, True))
            for v in reversed(self.children[u]):
                self.depth[v] = self.depth[u] + 1
                stack.append((v, False))
        self.height = max(self.depth)

    def query(self, qtype: str, args: list[int]) -> QueryResult:
        qtype = qtype.upper()
        if qtype == "ANCESTOR":
            self._check_arity(qtype, args, 2)
            u, v = args
            self._check_nodes(u, v)
            return QueryResult(self._tin[u] <= self._tin[v] <= self._tout[u], 1.0)
        if qtype == "LCA":
            self._check_arity(qtype, args, 2)
            u, v = args
            self._check_nodes(u, v)
            return QueryResult(self._lca(u, v), math.ceil(math.log2(self.n)))
        if qtype == "DEPTH":
            self._check_arity(qtype, args, 1)
            (v,) = args
            self._check_nodes(v)
            return QueryResult(self.depth[v], max(1, math.ceil(math.log2(self.height + 1))))
        if qtype == "CHILDREN":
            self._check_arity(qtype, args, 1)
            (v,) = args
            self._check_nodes(v)
            k = len(self.children[v])
            bits = max(1.0, math.log2(self.n) + self._log2_comb(self.n - 1, k))
            return QueryResult(self.children[v].copy(), bits)
        if qtype == "PATH":
            self._check_arity(qtype, args, 2)
            u, v = args
            self._check_nodes(u, v)
            path = self._path(u, v)
            return QueryResult(path, max(1.0, self._log2_perm(self.n - 2, max(0, len(path) - 2))))
        raise ValueError(f"unknown query type: {qtype}")

    def score(self, predicted: list[int]) -> dict[str, Any]:
        if len(predicted) != self.n:
            raise ValueError(f"expected {self.n} parent values, got {len(predicted)}")
        wrong = [(i, predicted[i], self.parent[i]) for i in range(1, self.n) if predicted[i] != self.parent[i]]
        total = self.n - 1
        correct = total - len(wrong)
        return {"score": correct / total, "correct": correct, "total": total, "errors": wrong}

    def lower_bound_bits(self) -> float:
        if self.method == "recursive":
            return math.lgamma(self.n) / math.log(2)
        return (self.n - 2) * math.log2(self.n)

    def _lca(self, u: int, v: int) -> int:
        ancestors = set()
        while u != -1:
            ancestors.add(u)
            u = self.parent[u]
        while v not in ancestors:
            v = self.parent[v]
        return v

    def _path(self, u: int, v: int) -> list[int]:
        up = []
        seen = set()
        x = u
        while x != -1:
            up.append(x)
            seen.add(x)
            x = self.parent[x]
        down = []
        x = v
        while x not in seen:
            down.append(x)
            x = self.parent[x]
        return up[:up.index(x) + 1] + down[::-1]

    def _check_nodes(self, *nodes: int) -> None:
        for node in nodes:
            if not 0 <= node < self.n:
                raise ValueError(f"node {node} out of range [0, {self.n})")

    @staticmethod
    def _check_arity(qtype: str, args: list[int], n: int) -> None:
        if len(args) != n:
            raise ValueError(f"{qtype} expects {n} argument(s), got {len(args)}")

    @staticmethod
    def _log2_comb(n: int, k: int) -> float:
        if not 0 <= k <= n:
            return float("-inf")
        return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)) / math.log(2)

    @staticmethod
    def _log2_perm(n: int, k: int) -> float:
        if not 0 <= k <= n:
            return float("-inf")
        return (math.lgamma(n + 1) - math.lgamma(n - k + 1)) / math.log(2)


class TreeReconstructionEnv(Env):
    __version__ = "0.0.1"
    QUERY_RE = re.compile(r"^QUERY\s+(ANCESTOR|LCA|DEPTH|CHILDREN|PATH)\s+(-?\d+)(?:\s+(-?\d+))?$", re.I)
    SUBMIT_RE = re.compile(r"^SUBMIT\s+(?P<body>[-,\s\d\[\]]+)$", re.I)
    ANSWER_BODY_RE = re.compile(r"^[-,\s\d\[\]]+$")
    ANSWER_RE = re.compile(r"<ANSWER>(.*?)</ANSWER>", re.I | re.S)

    def __init__(
        self,
        n: int = 20,
        method: str = "prufer",
        max_queries: int = 64,
        max_turns: int = 32,
        allowed_queries: tuple[str, ...] = ("ANCESTOR", "LCA", "DEPTH"),
    ):
        self._defaults = self._validate_config(n, method, max_queries, max_turns, allowed_queries)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        opts = options or {}
        cfg = self._validate_config(
            opts.get("n", self._defaults["n"]),
            opts.get("method", self._defaults["method"]),
            opts.get("max_queries", self._defaults["max_queries"]),
            opts.get("max_turns", self._defaults["max_turns"]),
            opts.get("allowed_queries", self._defaults["allowed_queries"]),
        )
        self._n = cfg["n"]
        self._method = cfg["method"]
        self._max_queries = cfg["max_queries"]
        self._max_turns = cfg["max_turns"]
        self._allowed_queries = cfg["allowed_queries"]
        tree_seed = 0 if seed is None else int(seed)
        self._tree = HiddenTree(self._n, tree_seed, self._method)
        self._turns = 0
        self._queries = 0
        self._bits = 0.0
        self._done = False
        return self._prompt(), {
            "challenge_id": str(tree_seed),
            "env_id": "tree_reconstruction",
            "spec_version": self.__version__,
            "n": self._n,
            "method": self._method,
            "max_queries": self._max_queries,
            "max_turns": self._max_turns,
            "allowed_queries": list(self._allowed_queries),
            "info_lower_bound": self._tree.lower_bound_bits(),
        }

    def step(self, action: str):
        if self._done:
            raise RuntimeError("episode is already done; call reset()")
        self._turns += 1
        text, wrapped, tag_error = self._body(action or "")
        if tag_error:
            self._done = True
            return None, 0.0, True, False, self._info(score=0.0, success=False, error=tag_error)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        try:
            submitted = self._submission(lines, allow_answer=False)
        except ValueError as exc:
            self._done = True
            return None, 0.0, True, False, self._info(score=0.0, success=False, error=str(exc))
        if submitted is not None:
            return self._finish(submitted)

        queries = [self.QUERY_RE.fullmatch(line) for line in lines]
        if queries and all(queries):
            return self._step_queries(queries)
        if lines and any(match is not None for match in queries):
            self._done = True
            return None, 0.0, True, False, self._info(score=0.0, success=False, error="malformed query batch")

        try:
            submitted = self._submission(lines, allow_answer=wrapped)
        except ValueError as exc:
            self._done = True
            return None, 0.0, True, False, self._info(score=0.0, success=False, error=str(exc))
        if submitted is not None:
            return self._finish(submitted)

        self._done = True
        return None, 0.0, True, False, {
            "score": 0.0,
            "success": False,
            "error": "expected QUERY or SUBMIT",
            "query_count": self._queries,
            "turn_count": self._turns,
        }

    def _step_queries(self, queries):
        lines = []
        for match in queries:
            qtype = match.group(1).upper()
            args = [int(x) for x in match.groups()[1:] if x is not None]
            if qtype not in self._allowed_queries:
                lines.append(f"{qtype} {' '.join(map(str, args))}: ERROR query type not allowed")
                continue
            if self._queries >= self._max_queries:
                self._done = True
                lines.append("ERROR query limit reached")
                return "\n".join(lines), 0.0, False, True, self._info(error="query limit reached")
            try:
                result = self._tree.query(qtype, args)
            except ValueError as exc:
                lines.append(f"{qtype} {' '.join(map(str, args))}: ERROR {exc}")
                continue
            self._queries += 1
            self._bits += result.bits
            lines.append(f"{qtype} {' '.join(map(str, args))}: {self._format(result.value)}")

        if self._turns >= self._max_turns:
            self._done = True
            return "\n".join(lines), 0.0, False, True, self._info(error="turn limit reached")
        return "\n".join(lines), 0.0, False, False, self._info()

    def _submission(self, lines: list[str], *, allow_answer: bool) -> list[int] | None:
        if not lines:
            return None
        match = self.SUBMIT_RE.fullmatch(lines[0]) if len(lines) == 1 else None
        if match is None:
            if not allow_answer:
                return None
            if len(lines) != 1 or self.QUERY_RE.fullmatch(lines[0]) or lines[0].upper().startswith("QUERY "):
                return None
            body = lines[0]
            if not self.ANSWER_BODY_RE.fullmatch(body):
                raise ValueError("malformed submission")
        else:
            body = match.group("body")
        if not re.search(r"-?\d+", body):
            return None
        values = [int(x) for x in re.findall(r"-?\d+", body)]
        if len(values) == self._n - 1:
            return [-1] + values
        if len(values) == self._n:
            values[0] = -1
            return values
        raise ValueError(f"expected {self._n - 1} or {self._n} parent values, got {len(values)}")

    @classmethod
    def _body(cls, text: str) -> tuple[str, bool, str | None]:
        matches = list(cls.ANSWER_RE.finditer(text))
        if len(matches) > 1:
            return text, False, "multiple ANSWER blocks"
        if not matches:
            return text, False, None
        outside = (text[:matches[0].start()] + text[matches[0].end():]).strip()
        if outside:
            return text, False, "ANSWER block must not have surrounding text"
        return matches[0].group(1), True, None

    def _finish(self, predicted: list[int]):
        self._done = True
        result = self._tree.score(predicted)
        info = self._info(**result, submitted_parent=predicted, success=result["score"] == 1.0)
        return None, float(result["score"]), True, False, info

    def _info(self, **extra):
        return {
            "query_count": self._queries,
            "turn_count": self._turns,
            "total_bits": self._bits,
            "info_lower_bound": self._tree.lower_bound_bits(),
            **extra,
        }

    def _prompt(self) -> str:
        query_docs = {
            "ANCESTOR": "- QUERY ANCESTOR u v: YES if u is an ancestor of v, otherwise NO",
            "LCA": "- QUERY LCA u v: lowest common ancestor of u and v",
            "DEPTH": "- QUERY DEPTH v: depth of v, with root at depth 0",
            "CHILDREN": "- QUERY CHILDREN v: all children of v",
            "PATH": "- QUERY PATH u v: nodes on the path from u to v",
        }
        docs = "\n".join(query_docs[q] for q in self._allowed_queries)
        return f"""You are reconstructing a hidden rooted tree with {self._n} nodes labeled 0 to {self._n - 1}.
Node 0 is the root. Determine the parent of every node 1 through {self._n - 1}.

Available queries:
{docs}

Rules:
- In each response, either issue one or more QUERY lines or submit the final tree.
- Query budget: {self._max_queries}
- Turn budget: {self._max_turns}
- Submit as: SUBMIT p1 p2 ... p{self._n - 1}
- p_i is the parent of node i. Do not submit a parent for root node 0.

A complete reconstruction must identify every parent exactly."""

    @staticmethod
    def validate_options(options: dict) -> dict[str, Any]:
        return TreeReconstructionEnv._validate_config(
            options.get("n", 20),
            options.get("method", "prufer"),
            options.get("max_queries", 64),
            options.get("max_turns", 32),
            options.get("allowed_queries", ("ANCESTOR", "LCA", "DEPTH")),
        )

    @staticmethod
    def _validate_config(n, method, max_queries, max_turns, allowed_queries) -> dict[str, Any]:
        allowed = {"ANCESTOR", "LCA", "DEPTH", "CHILDREN", "PATH"}
        n = TreeReconstructionEnv._as_int("n", n)
        max_queries = TreeReconstructionEnv._as_int("max_queries", max_queries)
        max_turns = TreeReconstructionEnv._as_int("max_turns", max_turns)
        method = str(method)
        if isinstance(allowed_queries, str):
            raise ValueError("allowed_queries must be a non-empty sequence")
        try:
            queries = tuple(str(q).upper() for q in allowed_queries)
        except TypeError as exc:
            raise ValueError("allowed_queries must be a non-empty sequence") from exc
        if n < 2:
            raise ValueError(f"n must be at least 2, got {n}")
        if method not in {"prufer", "recursive"}:
            raise ValueError(f"method must be 'prufer' or 'recursive', got {method!r}")
        if max_queries < 0:
            raise ValueError(f"max_queries must be >= 0, got {max_queries}")
        if max_turns <= 0:
            raise ValueError(f"max_turns must be > 0, got {max_turns}")
        if not queries:
            raise ValueError("allowed_queries must not be empty")
        unknown = sorted(set(queries) - allowed)
        if unknown:
            raise ValueError(f"unknown allowed_queries: {unknown}")
        return {
            "n": n,
            "method": method,
            "max_queries": max_queries,
            "max_turns": max_turns,
            "allowed_queries": queries,
        }

    @staticmethod
    def _as_int(name: str, value) -> int:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be an integer, got {value!r}")
        try:
            out = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be an integer, got {value!r}") from exc
        if out != value and not isinstance(value, str):
            raise ValueError(f"{name} must be an integer, got {value!r}")
        return out

    @staticmethod
    def _format(value: Any) -> str:
        if isinstance(value, bool):
            return "YES" if value else "NO"
        if isinstance(value, list):
            return "[" + ", ".join(map(str, value)) + "]"
        return str(value)
