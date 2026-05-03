"""Pure statistical decisions. No I/O, no side effects.

Per-env stratified always-valid test for the matched-pair binomial. Each pair
gives (champion_pass, challenger_pass) in {0,1}^2; concordant pairs (both
pass / both fail) carry no signal. Discordant pairs are Bernoulli(p_e) per
env e, with p_e = P(challenger_only | discordant, env e). Under H_0_e (models
equivalent on env e) p_e = 1/2.

For each env, maintain a one-sided always-valid CS [L_e(t), U_e(t)] for p_e
via mixture e-processes at general boundary p_0 (Beta(1/2, 1/2) prior restricted
to one side of p_0). Aggregate at the parameter level with predeclared weights:
    L_mu = sum_e pi_e L_e        U_mu = sum_e pi_e U_e
Decision (anytime-valid):
    dethrone iff L_mu > p_star = 0.5 + delta_p
    futility iff U_mu <= p_star
Bonferroni-Ville bounds each direction at alpha/2.

The duel terminates on dethrone, futility, an outer budget cap (cursor reaches
cfg.max_pairs_per_duel — anytime-valid CSs hold at any stopping time), or an
infra-side abort (SLOT_DEAD, validator-stop). pair_log_e at p_0 = 1/2 is kept
for the symmetry sanity test; production decisions go through log_e_plus /
log_e_minus with the boundary-parametric form.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.special import betaincc
from scipy.stats import beta as _beta_dist


@dataclass(frozen=True)
class PairCounts:
    challenger_only: int = 0
    champion_only: int = 0
    both_pass: int = 0
    both_fail: int = 0

    @property
    def total(self) -> int:
        return self.challenger_only + self.champion_only + self.both_pass + self.both_fail

    @property
    def discordant(self) -> int:
        return self.challenger_only + self.champion_only

    def add(self, champion_pass: int, challenger_pass: int) -> PairCounts:
        if challenger_pass and not champion_pass:
            return PairCounts(self.challenger_only + 1, self.champion_only, self.both_pass, self.both_fail)
        if champion_pass and not challenger_pass:
            return PairCounts(self.challenger_only, self.champion_only + 1, self.both_pass, self.both_fail)
        if challenger_pass and champion_pass:
            return PairCounts(self.challenger_only, self.champion_only, self.both_pass + 1, self.both_fail)
        return PairCounts(self.challenger_only, self.champion_only, self.both_pass, self.both_fail + 1)


def pair_log_e(k: int, n: int) -> float:
    """log E_n^+ for the one-sided beta-(1/2, 1/2) mixture e-process at p_0 = 1/2."""
    if n == 0:
        return 0.0
    log_inner = (n * math.log(2.0)
                 + math.lgamma(k + 0.5) + math.lgamma(n - k + 0.5)
                 - math.lgamma(n + 1.0)
                 - math.log(math.pi))
    upper_tail = float(betaincc(k + 0.5, n - k + 0.5, 0.5))
    if upper_tail <= 0.0:
        return -math.inf
    return math.log(2.0) + log_inner + math.log(upper_tail)


# ---------------------------------------------------------------------------
# Per-env stratified design (mean-rule)
# ---------------------------------------------------------------------------
# Per env e, observe (k_e, n_e) where k_e = challenger_only_e, n_e = discordant_e.
# Maintain a one-sided always-valid CS [L_e(t), U_e(t)] for p_e via mixture
# e-processes at general boundary p_0 (Beta(1/2, 1/2) prior restricted to one
# side of p_0). Aggregate at the parameter level:
#     L_mu = sum_e pi_e * L_e
#     U_mu = sum_e pi_e * U_e
# Decision: dethrone iff L_mu > p_star, futility iff U_mu <= p_star, with
# weights pi_e fixed in advance and Bonferroni-split per-env alphas:
#     alpha^-_e = alpha_dethrone / E
#     alpha^+_e = alpha_futility / E

def log_e_plus(k: int, n: int, p0: float) -> float:
    """log E_n^+ for the one-sided mixture e-process testing H_0: p <= p_0,
    with Beta(1/2, 1/2) prior restricted to (p_0, 1).

    Under H_0 with respect to the natural filtration, {E_n^+}_n is a
    non-negative supermartingale with E_0 = 1. Computed in log space via
    scipy.stats.beta.logsf for tail stability.
    """
    if not (0.0 < p0 < 1.0):
        raise ValueError(f"p0 must be in (0, 1), got {p0}")
    if n == 0:
        return 0.0
    log_b_kn = math.lgamma(k + 0.5) + math.lgamma(n - k + 0.5) - math.lgamma(n + 1.0)
    log_upper_tail = float(_beta_dist.logsf(p0, k + 0.5, n - k + 0.5))
    log_upper_tail_prior = float(_beta_dist.logsf(p0, 0.5, 0.5))
    if not math.isfinite(log_upper_tail) or not math.isfinite(log_upper_tail_prior):
        return -math.inf
    return (log_b_kn + log_upper_tail
            - math.log(math.pi)
            - log_upper_tail_prior
            - k * math.log(p0)
            - (n - k) * math.log1p(-p0))


def log_e_minus(k: int, n: int, p0: float) -> float:
    """log E_n^- for the one-sided mixture e-process testing H_0: p >= p_0,
    with Beta(1/2, 1/2) prior restricted to (0, p_0).

    Symmetric construction to log_e_plus, computed via beta.logcdf.
    """
    if not (0.0 < p0 < 1.0):
        raise ValueError(f"p0 must be in (0, 1), got {p0}")
    if n == 0:
        return 0.0
    log_b_kn = math.lgamma(k + 0.5) + math.lgamma(n - k + 0.5) - math.lgamma(n + 1.0)
    log_lower_tail = float(_beta_dist.logcdf(p0, k + 0.5, n - k + 0.5))
    log_lower_tail_prior = float(_beta_dist.logcdf(p0, 0.5, 0.5))
    if not math.isfinite(log_lower_tail) or not math.isfinite(log_lower_tail_prior):
        return -math.inf
    return (log_b_kn + log_lower_tail
            - math.log(math.pi)
            - log_lower_tail_prior
            - k * math.log(p0)
            - (n - k) * math.log1p(-p0))


_CS_EPS = 1e-9
_CS_BISECT_STEPS = 40


def env_lower_cs(k: int, n: int, alpha_minus: float) -> float:
    """One-sided lower CS L_e: largest p_0 with log_e_plus(k, n, p_0) >= -log(alpha_minus).

    log_e_plus is decreasing in p_0, so the rejection set is [0, L_e]; bisect
    on (eps, 1-eps). Boundary cases: n=0 or k=0 -> L=0 (no positive evidence).
    """
    if not (0.0 < alpha_minus < 1.0):
        raise ValueError(f"alpha_minus must be in (0, 1), got {alpha_minus}")
    if n == 0 or k == 0:
        return 0.0
    threshold = -math.log(alpha_minus)
    if log_e_plus(k, n, 1.0 - _CS_EPS) >= threshold:
        return 1.0 - _CS_EPS
    if log_e_plus(k, n, _CS_EPS) < threshold:
        return 0.0
    lo, hi = _CS_EPS, 1.0 - _CS_EPS
    for _ in range(_CS_BISECT_STEPS):
        mid = 0.5 * (lo + hi)
        if log_e_plus(k, n, mid) >= threshold:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def env_upper_cs(k: int, n: int, alpha_plus: float) -> float:
    """One-sided upper CS U_e: smallest p_0 with log_e_minus(k, n, p_0) >= -log(alpha_plus).

    log_e_minus is increasing in p_0; rejection set is [U_e, 1]. n=0 or k=n -> U=1.
    """
    if not (0.0 < alpha_plus < 1.0):
        raise ValueError(f"alpha_plus must be in (0, 1), got {alpha_plus}")
    if n == 0 or k == n:
        return 1.0
    threshold = -math.log(alpha_plus)
    if log_e_minus(k, n, _CS_EPS) >= threshold:
        return _CS_EPS
    if log_e_minus(k, n, 1.0 - _CS_EPS) < threshold:
        return 1.0
    lo, hi = _CS_EPS, 1.0 - _CS_EPS
    for _ in range(_CS_BISECT_STEPS):
        mid = 0.5 * (lo + hi)
        if log_e_minus(k, n, mid) >= threshold:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


@dataclass(frozen=True)
class EnvCS:
    """Per-env one-sided CS bounds at the configured alpha levels."""
    k: int
    n: int
    L: float
    U: float


@dataclass(frozen=True)
class MeanDecision:
    """Mean-rule decision over a fixed pi_e weighting of envs.

    Coverage event (per Bonferroni split):
        P(L_e <= p_e for all e for all t) >= 1 - alpha_dethrone
        P(U_e >= p_e for all e for all t) >= 1 - alpha_futility
    Linearity gives L_mu <= mu <= U_mu uniformly in t under each event.
    """
    dethrone: bool
    futility: bool
    L_mu: float
    U_mu: float
    p_star: float
    alpha_dethrone: float
    alpha_futility: float
    env_cs: tuple[tuple[str, EnvCS], ...]


def decide_dethrone(
    per_env_counts: dict[str, PairCounts],
    weights: dict[str, float],
    p_star: float,
    alpha_dethrone: float,
    alpha_futility: float,
) -> MeanDecision:
    """Mean-rule dethrone decision.

    per_env_counts[e]: PairCounts; uses .challenger_only as k_e, .discordant as n_e.
    weights[e]: pi_e, fixed in advance, summing to 1.
    p_star = 1/2 + delta_p (declared in advance).
    alpha_dethrone, alpha_futility: per-direction error budgets, split Bonferroni
    across envs as alpha^-_e = alpha_dethrone / E and alpha^+_e = alpha_futility / E.
    """
    if not (0.0 < p_star < 1.0):
        raise ValueError(f"p_star must be in (0, 1), got {p_star}")
    if not (0.0 < alpha_dethrone < 1.0) or not (0.0 < alpha_futility < 1.0):
        raise ValueError("alpha_dethrone and alpha_futility must be in (0, 1)")
    if set(per_env_counts) != set(weights):
        raise ValueError("per_env_counts and weights must have the same env keys")
    weight_sum = math.fsum(weights.values())
    if not math.isclose(weight_sum, 1.0, abs_tol=1e-9):
        raise ValueError(f"weights must sum to 1, got {weight_sum}")
    for e, w in weights.items():
        if not (math.isfinite(w) and 0.0 <= w):
            raise ValueError(f"weight for env {e!r} must be finite and >= 0, got {w}")

    e_count = len(per_env_counts)
    am = alpha_dethrone / e_count
    ap = alpha_futility / e_count

    env_cs: list[tuple[str, EnvCS]] = []
    l_mu = 0.0
    u_mu = 0.0
    for e in sorted(per_env_counts):
        c = per_env_counts[e]
        k, n = c.challenger_only, c.discordant
        L = env_lower_cs(k, n, am)
        U = env_upper_cs(k, n, ap)
        env_cs.append((e, EnvCS(k=k, n=n, L=L, U=U)))
        l_mu += weights[e] * L
        u_mu += weights[e] * U

    return MeanDecision(
        dethrone=l_mu > p_star,
        futility=u_mu <= p_star,
        L_mu=l_mu,
        U_mu=u_mu,
        p_star=p_star,
        alpha_dethrone=alpha_dethrone,
        alpha_futility=alpha_futility,
        env_cs=tuple(env_cs),
    )


# ---------------------------------------------------------------------------
# Adaptive sampler (efficiency layer; not part of the validity proof)
# ---------------------------------------------------------------------------
# The CS combination above is anytime-valid under any predictable env-selection
# rule. The functions below are an efficiency heuristic: pick the env whose
# next sample most reduces uncertainty in the dethrone-or-futility decision.
# Selection uses only past data (counts and CSs already observed).

def stabilized_p(k: int, n: int) -> float:
    """Beta(1/2, 1/2)-posterior mean estimator for p_e.

    Returns (k + 1/2) / (n + 1). Defined for n = 0 (returns 1/2) and stable
    against degenerate k = 0 or k = n. This is a stabilized score, not the
    same object as the CS bounds.
    """
    return (k + 0.5) / (n + 1.0)


def _bernoulli_kl(p: float, q: float) -> float:
    """KL(Bern(p) || Bern(q)) in nats.

    Boundary handling: KL(0 || q) = -log(1 - q); KL(1 || q) = -log q;
    KL(p || 0) = +inf for p > 0; KL(p || 1) = +inf for p < 1.
    """
    if not (0.0 <= p <= 1.0) or not (0.0 < q < 1.0):
        if q <= 0.0 or q >= 1.0:
            return math.inf if (q <= 0.0 and p > 0.0) or (q >= 1.0 and p < 1.0) else 0.0
        raise ValueError(f"p must be in [0, 1], got {p}")
    if p == 0.0:
        return -math.log1p(-q)
    if p == 1.0:
        return -math.log(q)
    return p * math.log(p / q) + (1.0 - p) * math.log((1.0 - p) / (1.0 - q))


_UCB_C = 0.5


def env_score(
    k: int,
    n_disc: int,
    n_total: int,
    weight: float,
    p_star: float,
    L: float,
    U: float,
    score_lambda: float,
    cost: float = 1.0,
    t_total: int = 0,
) -> float:
    """Adaptive-sampling score for one env.

        score = (q_hat_e / cost_e) * pi_e * [
            lambda * 1{p_tilde_e > p_star} * KL(Bern(p_tilde_e) || Bern(p_star))
            + (1 - lambda) * (U_e - L_e)
            + c_explore * sqrt(log(t_total + 1) / (n_total_e + 1))
        ]

    First term is the dethrone-seeking signal (only counts envs whose stabilized
    estimate already favors dethrone). Second term reduces upper-bound uncertainty
    so that futility can fire when truly no challenger advantage exists. Third
    term is a UCB-style exploration bonus that revisits under-sampled envs;
    keeps the sampler from getting stuck on whichever env's q_hat estimate
    happened to be highest from one early sample. All weighted by the predeclared
    importance weight pi_e and by per-task informative yield q_hat_e divided by
    per-task cost.
    """
    if cost <= 0.0:
        raise ValueError(f"cost must be > 0, got {cost}")
    if t_total < 0:
        raise ValueError(f"t_total must be >= 0, got {t_total}")
    q_hat = (n_disc + 1.0) / (n_total + 2.0) if n_total > 0 else 0.5
    p_tilde = stabilized_p(k, n_disc)
    dethrone_term = _bernoulli_kl(p_tilde, p_star) if p_tilde > p_star else 0.0
    uncertainty_term = max(U - L, 0.0)
    explore_bonus = _UCB_C * math.sqrt(math.log(t_total + 1.0) / (n_total + 1))
    return (q_hat / cost) * weight * (
        score_lambda * dethrone_term + (1.0 - score_lambda) * uncertainty_term
        + explore_bonus
    )


def select_env(
    per_env_counts: dict[str, PairCounts],
    weights: dict[str, float],
    env_cs: dict[str, EnvCS],
    p_star: float,
    n_min: int,
    score_lambda: float,
    costs: dict[str, float] | None = None,
) -> str:
    """Pick the env to sample next.

    Cold-start: any env with total samples < n_min returns the lowest-n such env
    (deterministic round-robin by under-sampling). Once every env clears n_min,
    return argmax of env_score.

    Tie-breaking: stable lex order by env name. Selection depends only on past
    data, preserving the predictability requirement for anytime-valid inference.
    """
    if not (0.0 <= score_lambda <= 1.0):
        raise ValueError(f"score_lambda must be in [0, 1], got {score_lambda}")
    if n_min < 0:
        raise ValueError(f"n_min must be >= 0, got {n_min}")
    if set(per_env_counts) != set(weights) or set(per_env_counts) != set(env_cs):
        raise ValueError("per_env_counts, weights, env_cs must share the same env keys")
    cost_map = costs or {e: 1.0 for e in per_env_counts}

    cold = [e for e, c in per_env_counts.items() if c.total < n_min]
    if cold:
        return min(cold, key=lambda e: (per_env_counts[e].total, e))

    t_total = sum(c.total for c in per_env_counts.values())
    best_env = None
    best_score = -math.inf
    for e in sorted(per_env_counts):
        c = per_env_counts[e]
        cs = env_cs[e]
        s = env_score(
            k=c.challenger_only,
            n_disc=c.discordant,
            n_total=c.total,
            weight=weights[e],
            p_star=p_star,
            L=cs.L,
            U=cs.U,
            score_lambda=score_lambda,
            cost=cost_map[e],
            t_total=t_total,
        )
        if s > best_score:
            best_score = s
            best_env = e
    return best_env
