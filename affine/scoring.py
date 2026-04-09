from enum import Enum, auto
from math import exp, log, sqrt


class Verdict(Enum):
    CHALLENGER_WINS = auto()
    CHAMPION_HOLDS = auto()
    UNDECIDED = auto()


def bt_mle(w: int, l: int) -> tuple[float, float]:
    """Log-odds and variance for a two-player matchup.
    Pseudocounts avoid boundary infinities."""
    wa, la = w + 0.5, l + 0.5
    return log(wa / la), 1.0 / wa + 1.0 / la


def aggregate(deltas: list[float], variances: list[float]) -> tuple[float, float]:
    """Inverse-variance weighted combination. Returns (delta, var)."""
    if not deltas:
        return 0.0, float('inf')
    weights = [1.0 / v for v in variances]
    total = sum(weights)
    return sum(d * wi for d, wi in zip(deltas, weights)) / total, 1.0 / total


def check_duel(
    wins: dict[str, int],
    losses: dict[str, int],
    tasks: dict[str, int],
    max_tasks: int,
    k: float,
) -> tuple[Verdict, float]:
    """Global duel decision via BT meta-analysis. Returns (verdict, z_score)."""
    deltas, variances = [], []
    for name in wins:
        if wins[name] + losses[name] == 0:
            continue
        d, v = bt_mle(wins[name], losses[name])
        deltas.append(d)
        variances.append(v)

    if not deltas:
        return Verdict.UNDECIDED, 0.0

    delta, var = aggregate(deltas, variances)
    z = delta / sqrt(var)

    if z > k:
        return Verdict.CHALLENGER_WINS, z

    # Hopelessness: even winning all remaining tasks decisively can't reach k
    best_d, best_v = [], []
    for name in wins:
        remaining = max(0, max_tasks - tasks[name])
        bw, bl = wins[name] + remaining, losses[name]
        if bw + bl == 0:
            continue
        d, v = bt_mle(bw, bl)
        best_d.append(d)
        best_v.append(v)

    if best_d:
        bd, bv = aggregate(best_d, best_v)
        if bd / sqrt(bv) <= k:
            return Verdict.CHAMPION_HOLDS, z

    return Verdict.UNDECIDED, z


def compute_k(reign_blocks: int, k_init: float = 3.0, k_final: float = 1.0, halflife: int = 7200) -> float:
    """K decays exponentially from k_init to k_final over the champion's reign."""
    if halflife <= 0:
        return k_final
    return k_final + (k_init - k_final) * exp(-log(2) * reign_blocks / halflife)
