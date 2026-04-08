from enum import Enum, auto
from math import sqrt


class Verdict(Enum):
    CHALLENGER_WINS = auto()
    CHAMPION_HOLDS = auto()
    UNDECIDED = auto()


def wilson_lower(wins: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 0.0
    p = wins / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = p + z2 / (2.0 * n)
    spread = z * sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    return (center - spread) / denom


def wilson_upper(wins: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 1.0
    p = wins / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = p + z2 / (2.0 * n)
    spread = z * sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    return (center + spread) / denom
