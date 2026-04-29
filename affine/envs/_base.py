from __future__ import annotations

import importlib
import json
import random
import re
from dataclasses import dataclass
from typing import Any

ANSWER_RE = re.compile(r"<ANSWER>(.*?)</ANSWER>", re.IGNORECASE | re.DOTALL)


def tagged(text: str, *, strip: bool = True) -> str | None:
    matches = ANSWER_RE.findall(text or "")
    if len(matches) != 1:
        return None
    return matches[0].strip() if strip else matches[0]


def parse_json_obj(body: str):
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


def int_param(opts: dict, key: str, *, default: int, lo: int, hi: int) -> int:
    v = opts.get(key, default)
    if isinstance(v, bool):
        raise ValueError(f"{key} must be an integer, got {v!r}")
    try:
        out = int(v)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be an integer, got {v!r}") from exc
    if out != v and not isinstance(v, str):
        raise ValueError(f"{key} must be an integer, got {v!r}")
    if not lo <= out <= hi:
        raise ValueError(f"{key} must be in [{lo}, {hi}], got {out}")
    return out


@dataclass(frozen=True)
class Spec:
    title: str
    rules: tuple[str, ...]
    example_challenge: str
    example_answer: str

    def render(self, challenge: str) -> str:
        rules = "\n".join(f"* {r}" for r in self.rules)
        return (
            f"{self.title}\n\nRULES:\n{rules}\n\n"
            f"Example:\nCHALLENGE:\n{self.example_challenge}\n"
            f"RESPONSE:\n<ANSWER>{self.example_answer}</ANSWER>\n\n"
            f"Below, you will see the real task. Remember and follow the rules.\n\n"
            f"CHALLENGE:\n{challenge}"
        )


def load_env_class(entrypoint: str):
    mod, sep, name = entrypoint.partition(":")
    if not (mod and sep and name):
        raise ValueError(f"env entrypoint must be 'module:Class', got {entrypoint!r}")
    return getattr(importlib.import_module(mod), name)


@dataclass(frozen=True)
class EnvFactory:
    entrypoint: str

    def __post_init__(self):
        object.__setattr__(self, "_cls", load_env_class(self.entrypoint))

    def make(self):
        return self._cls()


class Env:
    __version__ = "0.0.0"
    env_id: str = ""
    option_keys: frozenset[str] = frozenset()

    def __init__(self, **opts):
        self.options = self.validate_options(opts)

    @classmethod
    def validate_options(cls, opts: dict) -> dict:
        unknown = set(opts) - cls.option_keys
        if unknown:
            raise ValueError(f"unknown options: {sorted(unknown)}")
        return cls._validate(opts)

    @classmethod
    def _validate(cls, opts: dict) -> dict:
        raise NotImplementedError

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        raise NotImplementedError

    def step(self, action: Any):
        raise NotImplementedError

    def close(self) -> None:
        pass


class ExactAnswerEnv(Env):
    __version__ = "0.0.1"
    spec: Spec
    strip_answer: bool = True

    def _generate(self, params: dict, rng: random.Random) -> tuple[str, dict]:
        raise NotImplementedError

    def parse_answer(self, body: str):
        raise NotImplementedError

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        overrides = dict(options or {})
        unknown = set(overrides) - self.option_keys
        if unknown:
            raise ValueError(f"{type(self).__name__}: unknown reset options: {sorted(unknown)}")
        params = self.validate_options({**self.options, **overrides})
        challenge, info_extra = self._generate(params, random.Random(0 if seed is None else seed))
        self._answer = self.parse_answer(self._target)
        if self._answer is None:
            raise RuntimeError(f"{type(self).__name__}: canonical answer failed parse_answer round-trip")
        return self.spec.render(challenge), {
            "challenge_id": str(seed if seed is not None else 0),
            "env_id": self.env_id,
            "spec_version": self.__version__,
            **params,
            **info_extra,
        }

    def step(self, action: str):
        body = tagged(action, strip=self.strip_answer)
        parsed = self.parse_answer(body) if body is not None else None
        ok = parsed is not None and parsed == self._answer
        return None, float(ok), True, False, {"score": float(ok), "success": ok}
