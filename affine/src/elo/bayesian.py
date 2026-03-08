"""
Bayesian Bradley-Terry Model (EXPERIMENTAL)

Not for authoritative ranking or weight decisions. Uses explicit priors,
which biases low-sample estimates. Use PairedBradleyTerryModel (pure MLE)
for canonical ratings and bootstrap for uncertainty.

Requires: numpyro, jax
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Set

HAS_NUMPYRO = False
jax = None
jnp = None
numpyro = None
dist = None
MCMC = None
NUTS = None

try:
    import jax as _jax
    import jax.numpy as _jnp
    import numpyro as _numpyro
    import numpyro.distributions as _dist
    from numpyro.infer import MCMC as _MCMC, NUTS as _NUTS

    jax = _jax
    jnp = _jnp
    numpyro = _numpyro
    dist = _dist
    MCMC = _MCMC
    NUTS = _NUTS
    HAS_NUMPYRO = True
except ImportError:
    pass

import numpy as np

from .config import EloConfig, DEFAULT_ELO_CONFIG
from .models import MatchResult, MatchOutcome, PairedMatchResult


@dataclass
class BayesianResult:
    """Result of Bayesian Bradley-Terry inference."""

    # Posterior samples for each player's skill (log-scale)
    skill_samples: Dict[str, np.ndarray]  # player_id -> array of posterior samples

    # Point estimates (posterior mean)
    skill_means: Dict[str, float]
    skill_stds: Dict[str, float]

    # ELO-scale ratings
    rating_means: Dict[str, Decimal]
    rating_stds: Dict[str, Decimal]

    # Credible intervals (95% by default)
    rating_ci_lower: Dict[str, Decimal]
    rating_ci_upper: Dict[str, Decimal]

    # First-mover advantage posterior
    first_mover_samples: Optional[np.ndarray] = None
    first_mover_mean: Optional[float] = None
    first_mover_std: Optional[float] = None
    first_mover_elo: Optional[float] = None
    first_mover_ci: Optional[Tuple[float, float]] = None

    # Draw probability (Davidson model)
    draw_nu_samples: Optional[np.ndarray] = None
    draw_nu_mean: Optional[float] = None

    # MCMC diagnostics
    num_samples: int = 0
    num_warmup: int = 0
    num_chains: int = 1
    r_hat: Optional[Dict[str, float]] = None  # Convergence diagnostic
    ess: Optional[Dict[str, float]] = None  # Effective sample size

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "skill_means": self.skill_means,
            "skill_stds": self.skill_stds,
            "rating_means": {k: float(v) for k, v in self.rating_means.items()},
            "rating_stds": {k: float(v) for k, v in self.rating_stds.items()},
            "rating_ci_lower": {k: float(v) for k, v in self.rating_ci_lower.items()},
            "rating_ci_upper": {k: float(v) for k, v in self.rating_ci_upper.items()},
            "first_mover_mean": self.first_mover_mean,
            "first_mover_std": self.first_mover_std,
            "first_mover_elo": self.first_mover_elo,
            "first_mover_ci": self.first_mover_ci,
            "draw_nu_mean": self.draw_nu_mean,
            "num_samples": self.num_samples,
            "num_warmup": self.num_warmup,
            "num_chains": self.num_chains,
            "r_hat": self.r_hat,
            "ess": self.ess,
        }



class BayesianBradleyTerry:
    """
    Bayesian Bradley-Terry model with MCMC inference.

    This model provides:
    1. Full posterior distributions over skill parameters
    2. Proper credible intervals (not just point estimates)
    3. Better small-sample behavior via informative priors
    4. Uncertainty that accounts for correlations between players
    5. Optional first-mover advantage estimation
    6. Optional Davidson extension for explicit draw modeling

    Uses Student-t(df=4) priors on skills for robustness to outliers.
    """

    def __init__(self, config: Optional[EloConfig] = None):
        """
        Initialize the Bayesian model.

        Args:
            config: ELO configuration
        """
        if not HAS_NUMPYRO:
            raise ImportError(
                "NumPyro/JAX required for Bayesian inference. "
                "Install with: pip install numpyro jax"
            )

        self.config = config or DEFAULT_ELO_CONFIG

        self._players: Set[str] = set()
        self._matches: List[Tuple[str, str, float, Optional[bool]]] = []  # (a, b, outcome, a_first)
        self._player_to_idx: Dict[str, int] = {}
        self._idx_to_player: Dict[int, str] = {}

        self._samples = None
        self._result: Optional[BayesianResult] = None

    def add_match(
        self,
        player_a: str,
        player_b: str,
        outcome: str,  # "a_wins", "b_wins", "draw"
        a_moves_first: Optional[bool] = True,
    ) -> None:
        """
        Add a match to the model.

        Args:
            player_a: ID of player A
            player_b: ID of player B
            outcome: Match outcome
            a_moves_first: Whether A moved first (None = unknown)
        """
        self._players.add(player_a)
        self._players.add(player_b)

        if outcome == "a_wins":
            y = 1.0
        elif outcome == "b_wins":
            y = 0.0
        else:
            y = 0.5

        self._matches.append((player_a, player_b, y, a_moves_first))

    def add_match_result(self, match: MatchResult) -> None:
        """Add a MatchResult to the model."""
        if len(match.participants) != 2:
            raise ValueError("Only 2-player matches supported")

        p_a = match.participants[0]
        p_b = match.participants[1]

        if p_a.outcome == MatchOutcome.WIN:
            outcome = "a_wins"
        elif p_b.outcome == MatchOutcome.WIN:
            outcome = "b_wins"
        else:
            outcome = "draw"

        self.add_match(p_a.miner_id, p_b.miner_id, outcome, match.is_first_mover)

    def add_paired_result(self, paired: PairedMatchResult) -> None:
        """Add a PairedMatchResult to the model."""
        self.add_match_result(paired.match_1)
        self.add_match_result(paired.match_2)

    def _build_index(self) -> None:
        """Build player index mappings."""
        sorted_players = sorted(self._players)
        self._player_to_idx = {p: i for i, p in enumerate(sorted_players)}
        self._idx_to_player = {i: p for i, p in enumerate(sorted_players)}

    def _bradley_terry_model(
        self,
        player_a_idx: jnp.ndarray,
        player_b_idx: jnp.ndarray,
        a_first: jnp.ndarray,
        outcomes: jnp.ndarray,
        n_players: int,
        estimate_first_mover: bool,
    ):
        """NumPyro model definition for Bradley-Terry with Student-t priors."""
        skills = numpyro.sample(
            "skills",
            dist.StudentT(
                df=4.0,
                loc=0.0,
                scale=self.config.BAYESIAN_PRIOR_SCALE
            ).expand([n_players])
        )

        if estimate_first_mover:
            alpha = numpyro.sample(
                "alpha",
                dist.Normal(0.0, 0.5)
            )
        else:
            alpha = 0.0

        skill_a = skills[player_a_idx]
        skill_b = skills[player_b_idx]
        logit = skill_a - skill_b + alpha * (2 * a_first - 1)
        p_a = jax.nn.sigmoid(logit)

        numpyro.sample(
            "obs",
            dist.Bernoulli(probs=p_a),
            obs=outcomes
        )

    def _davidson_model(
        self,
        player_a_idx: jnp.ndarray,
        player_b_idx: jnp.ndarray,
        a_first: jnp.ndarray,
        outcomes: jnp.ndarray,  # 0 = B wins, 1 = draw, 2 = A wins
        n_players: int,
        estimate_first_mover: bool,
    ):
        """
        Davidson model with explicit draw probability.

        P(draw) = ν√(p_a·p_b) / (p_a + p_b + ν√(p_a·p_b))
        """
        skills = numpyro.sample(
            "skills",
            dist.StudentT(df=4.0, loc=0.0, scale=self.config.BAYESIAN_PRIOR_SCALE)
            .expand([n_players])
        )

        nu = numpyro.sample("nu", dist.HalfNormal(1.0))

        if estimate_first_mover:
            alpha = numpyro.sample("alpha", dist.Normal(0.0, 0.5))
        else:
            alpha = 0.0

        skill_a = skills[player_a_idx]
        skill_b = skills[player_b_idx]

        logit = skill_a - skill_b + alpha * (2 * a_first - 1)
        p_a_raw = jax.nn.sigmoid(logit)
        p_b_raw = 1 - p_a_raw

        sqrt_prod = jnp.sqrt(p_a_raw * p_b_raw)
        denom = p_a_raw + p_b_raw + nu * sqrt_prod

        p_a_win = p_a_raw / denom
        p_draw = nu * sqrt_prod / denom
        p_b_win = p_b_raw / denom

        probs = jnp.stack([p_b_win, p_draw, p_a_win], axis=-1)

        numpyro.sample("obs", dist.Categorical(probs=probs), obs=outcomes)

    def fit(
        self,
        estimate_first_mover: Optional[bool] = None,
        use_davidson: bool = False,
        num_samples: Optional[int] = None,
        num_warmup: Optional[int] = None,
        num_chains: Optional[int] = None,
        confidence_level: Optional[float] = None,
        seed: int = 0,
    ) -> BayesianResult:
        """
        Fit the Bayesian Bradley-Terry model using MCMC.

        Args:
            estimate_first_mover: Whether to estimate first-mover advantage
            use_davidson: Whether to use Davidson model for explicit draws
            num_samples: Number of MCMC samples (default: from config)
            num_warmup: Number of warmup samples (default: from config)
            num_chains: Number of MCMC chains (default: from config)
            confidence_level: Confidence level for intervals (default: from config)
            seed: Random seed

        Returns:
            BayesianResult with posterior distributions
        """
        if len(self._matches) == 0:
            raise ValueError("No matches to fit")

        if len(self._players) < 2:
            raise ValueError("Need at least 2 players")

        if estimate_first_mover is None:
            estimate_first_mover = self.config.ESTIMATE_FIRST_MOVER_ADVANTAGE
        if num_samples is None:
            num_samples = self.config.BAYESIAN_NUM_SAMPLES
        if num_warmup is None:
            num_warmup = self.config.BAYESIAN_NUM_WARMUP
        if num_chains is None:
            num_chains = self.config.BAYESIAN_NUM_CHAINS
        if confidence_level is None:
            confidence_level = self.config.BOOTSTRAP_CONFIDENCE_LEVEL

        self._build_index()
        n_players = len(self._players)

        player_a_idx = []
        player_b_idx = []
        a_first = []
        outcomes = []

        for a, b, y, first in self._matches:
            player_a_idx.append(self._player_to_idx[a])
            player_b_idx.append(self._player_to_idx[b])
            # True→1.0, False→0.0, None→0.5 (alpha_sign=0 via 2*0.5-1=0)
            a_first.append(1.0 if first is True else (0.0 if first is False else 0.5))

            if use_davidson:
                if y == 1.0:
                    outcomes.append(2)
                elif y == 0.0:
                    outcomes.append(0)
                else:
                    outcomes.append(1)
            else:
                outcomes.append(y)

        player_a_idx = jnp.array(player_a_idx, dtype=jnp.int32)
        player_b_idx = jnp.array(player_b_idx, dtype=jnp.int32)
        a_first = jnp.array(a_first)
        outcomes = jnp.array(outcomes, dtype=jnp.int32 if use_davidson else jnp.float32)

        if use_davidson:
            model = self._davidson_model
        else:
            model = self._bradley_terry_model

        rng_key = jax.random.PRNGKey(seed)
        kernel = NUTS(model)
        mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)

        mcmc.run(
            rng_key,
            player_a_idx=player_a_idx,
            player_b_idx=player_b_idx,
            a_first=a_first,
            outcomes=outcomes,
            n_players=n_players,
            estimate_first_mover=estimate_first_mover,
        )

        samples = mcmc.get_samples()
        self._samples = samples

        skill_samples_array = np.array(samples["skills"])
        skill_samples_array = skill_samples_array - skill_samples_array.mean(axis=1, keepdims=True)

        skill_samples = {}
        skill_means = {}
        skill_stds = {}
        rating_means = {}
        rating_stds = {}
        rating_ci_lower = {}
        rating_ci_upper = {}

        scale_factor = self.config.SCALE / math.log(10)
        default_rating = float(self.config.DEFAULT_RATING)

        ci_lower_pct = (1 - confidence_level) / 2 * 100
        ci_upper_pct = (1 + confidence_level) / 2 * 100

        for i, player in enumerate(sorted(self._players)):
            player_samples = skill_samples_array[:, i]
            skill_samples[player] = player_samples
            skill_means[player] = float(np.mean(player_samples))
            skill_stds[player] = float(np.std(player_samples))

            elo_samples = scale_factor * player_samples + default_rating
            rating_means[player] = Decimal(str(round(np.mean(elo_samples), 2)))
            rating_stds[player] = Decimal(str(round(np.std(elo_samples), 2)))
            rating_ci_lower[player] = Decimal(str(round(np.percentile(elo_samples, ci_lower_pct), 2)))
            rating_ci_upper[player] = Decimal(str(round(np.percentile(elo_samples, ci_upper_pct), 2)))

        first_mover_samples = None
        first_mover_mean = None
        first_mover_std = None
        first_mover_elo = None
        first_mover_ci = None

        if estimate_first_mover and "alpha" in samples:
            first_mover_samples = np.array(samples["alpha"])
            first_mover_mean = float(np.mean(first_mover_samples))
            first_mover_std = float(np.std(first_mover_samples))
            first_mover_elo = scale_factor * first_mover_mean
            first_mover_ci = (
                float(np.percentile(first_mover_samples, ci_lower_pct)) * scale_factor,
                float(np.percentile(first_mover_samples, ci_upper_pct)) * scale_factor,
            )

        draw_nu_samples = None
        draw_nu_mean = None

        if use_davidson and "nu" in samples:
            draw_nu_samples = np.array(samples["nu"])
            draw_nu_mean = float(np.mean(draw_nu_samples))

        r_hat_dict = None
        ess_dict = None
        try:
            from numpyro.diagnostics import summary as numpyro_summary
            diag = numpyro_summary(samples, prob=confidence_level)
            r_hat_dict = {}
            ess_dict = {}
            for param_name, param_diag in diag.items():
                if param_name == "skills":
                    for i, player in enumerate(sorted(self._players)):
                        r_hat_dict[player] = float(param_diag["r_hat"][i])
                        ess_dict[player] = float(param_diag["n_eff"][i])
                elif param_name == "alpha":
                    r_hat_dict["alpha"] = float(param_diag["r_hat"])
                    ess_dict["alpha"] = float(param_diag["n_eff"])
                elif param_name == "nu":
                    r_hat_dict["nu"] = float(param_diag["r_hat"])
                    ess_dict["nu"] = float(param_diag["n_eff"])
        except Exception:
            pass

        self._result = BayesianResult(
            skill_samples=skill_samples,
            skill_means=skill_means,
            skill_stds=skill_stds,
            rating_means=rating_means,
            rating_stds=rating_stds,
            rating_ci_lower=rating_ci_lower,
            rating_ci_upper=rating_ci_upper,
            first_mover_samples=first_mover_samples,
            first_mover_mean=first_mover_mean,
            first_mover_std=first_mover_std,
            first_mover_elo=first_mover_elo,
            first_mover_ci=first_mover_ci,
            draw_nu_samples=draw_nu_samples,
            draw_nu_mean=draw_nu_mean,
            num_samples=num_samples,
            num_warmup=num_warmup,
            num_chains=num_chains,
            r_hat=r_hat_dict,
            ess=ess_dict,
        )

        return self._result

    def predict_win_probability(
        self,
        player_a: str,
        player_b: str,
        a_moves_first: bool = True,
        return_samples: bool = False,
    ) -> float | Tuple[float, np.ndarray]:
        """
        Predict win probability with uncertainty.

        When Davidson model was used, computes three-outcome probabilities
        using the fitted draw parameter.

        Args:
            player_a: ID of player A
            player_b: ID of player B
            a_moves_first: Whether A moves first
            return_samples: If True, return (mean, samples)

        Returns:
            Mean probability of A winning (and samples if requested)
        """
        if self._result is None:
            raise ValueError("Model not fitted")

        if player_a not in self._result.skill_samples or player_b not in self._result.skill_samples:
            raise ValueError("Unknown player(s)")

        skill_a = self._result.skill_samples[player_a]
        skill_b = self._result.skill_samples[player_b]

        if self._result.first_mover_samples is not None:
            alpha = self._result.first_mover_samples
        else:
            alpha = 0.0

        if a_moves_first:
            logit = skill_a - skill_b + alpha
        else:
            logit = skill_a - skill_b - alpha

        if self._result.draw_nu_samples is not None:
            nu = self._result.draw_nu_samples
            p_a_raw = 1.0 / (1.0 + np.exp(-logit))
            p_b_raw = 1.0 - p_a_raw
            sqrt_prod = np.sqrt(p_a_raw * p_b_raw)
            denom = p_a_raw + p_b_raw + nu * sqrt_prod
            p_samples = p_a_raw / denom
        else:
            p_samples = 1.0 / (1.0 + np.exp(-logit))

        p_mean = float(np.mean(p_samples))

        if return_samples:
            return p_mean, p_samples
        return p_mean

    def clear(self) -> None:
        """Clear all data."""
        self._players.clear()
        self._matches.clear()
        self._player_to_idx.clear()
        self._idx_to_player.clear()
        self._samples = None
        self._result = None

    @property
    def result(self) -> Optional[BayesianResult]:
        """Get the fitted result."""
        return self._result


def fit_bayesian_bradley_terry(
    matches: List[MatchResult],
    config: Optional[EloConfig] = None,
    estimate_first_mover: bool = True,
    use_davidson: bool = False,
    **kwargs,
) -> BayesianResult:
    """
    Convenience function to fit Bayesian Bradley-Terry.

    Args:
        matches: List of MatchResult objects
        config: ELO configuration
        estimate_first_mover: Whether to estimate first-mover advantage
        use_davidson: Whether to use Davidson model for draws
        **kwargs: Additional arguments to fit()

    Returns:
        BayesianResult with posterior distributions
    """
    if not HAS_NUMPYRO:
        raise ImportError("NumPyro required. Install with: pip install numpyro jax")

    model = BayesianBradleyTerry(config)
    for match in matches:
        model.add_match_result(match)
    return model.fit(
        estimate_first_mover=estimate_first_mover,
        use_davidson=use_davidson,
        **kwargs,
    )
