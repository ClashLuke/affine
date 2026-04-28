from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EnvFactory:
    entrypoint: str

    def __post_init__(self):
        mod, sep, name = self.entrypoint.partition(":")
        if not (mod and sep and name):
            raise ValueError(f"env entrypoint must be 'module:Class', got {self.entrypoint!r}")
        cls = getattr(importlib.import_module(mod), name)
        object.__setattr__(self, "_cls", cls)

    def make(self):
        return self._cls()

    async def cleanup(self) -> None:
        pass


class Env:
    __version__ = "0.0.0"

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        raise NotImplementedError

    def step(self, action: Any):
        raise NotImplementedError

    def close(self) -> None:
        pass
