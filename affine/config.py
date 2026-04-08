from __future__ import annotations
import os
from dataclasses import dataclass, field


@dataclass
class EnvSpec:
    name: str
    image: str
    params: dict = field(default_factory=dict)
    env_vars: dict = field(default_factory=dict)
    mem_limit: str = "8g"


@dataclass
class Config:
    netuid: int = 120
    wallet_name: str = "default"
    hotkey_name: str = "default"
    network: str = "finney"
    subtensor_endpoint: str = "finney"
    subtensor_fallback: str = "wss://lite.sub.latent.to:443"
    max_tasks_per_env: int = 200
    tasks_per_batch: int = 4
    wilson_z: float = 1.96
    health_check_timeout: int = 300
    log_level: str = "INFO"
    environments: tuple[EnvSpec, ...] = ()

    @classmethod
    def from_env(cls) -> Config:
        return cls(
            netuid=int(os.getenv("NETUID", "120")),
            wallet_name=os.getenv("BT_WALLET_COLD", "default"),
            hotkey_name=os.getenv("BT_WALLET_HOT", "default"),
            network=os.getenv("BT_NETWORK", "finney"),
            subtensor_endpoint=os.getenv("SUBTENSOR_ENDPOINT", "finney"),
            subtensor_fallback=os.getenv("SUBTENSOR_FALLBACK", "wss://lite.sub.latent.to:443"),
            max_tasks_per_env=int(os.getenv("MAX_TASKS_PER_ENV", "200")),
            tasks_per_batch=int(os.getenv("TASKS_PER_BATCH", "4")),
            wilson_z=float(os.getenv("WILSON_Z", "1.96")),
            health_check_timeout=int(os.getenv("HEALTH_CHECK_TIMEOUT", "300")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            environments=_default_environments(),
        )


def _default_environments() -> tuple[EnvSpec, ...]:
    return (
        EnvSpec(
            name="affine:ded",
            image="affinefoundation/affine-env:v4",
            params={"task_type": "ded", "temperature": 0.0, "timeout": 600},
        ),
        EnvSpec(
            name="affine:abd",
            image="affinefoundation/affine-env:v4",
            params={"task_type": "abd", "temperature": 0.0, "timeout": 600},
        ),
        EnvSpec(
            name="game",
            image="affinefoundation/game:openspiel",
            params={"temperature": 0.0, "timeout": 7200},
        ),
        EnvSpec(
            name="distill",
            image="affinefoundation/distill:latest",
            params={"temperature": 0.0, "timeout": 600},
            mem_limit="2g",
        ),
    )
