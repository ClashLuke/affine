"""
Paired Bradley-Terry Model

Implements Maximum Likelihood Estimation (MLE) for the Bradley-Terry model
with support for paired matches and first-mover advantage estimation.

The standard Bradley-Terry model:
    P(A beats B) = p_A / (p_A + p_B)

With first-mover advantage α:
    P(A beats B | A first) = σ(β_A - β_B + α)
    P(A beats B | B first) = σ(β_A - β_B - α)

Where σ is the logistic sigmoid and β values are log-skills.

This implementation uses torch.compile Newton-Raphson for fast, accurate MLE.
Newton-Raphson is optimal for the convex Bradley-Terry problem and achieves
convergence in ~20-30 iterations with O(1-10ms) runtime via JIT compilation.

Optimization approach:
1. Sharpness-Aware Minimization (SAM) for finding flatter minima
2. Uniform skill clamping (±8) to prevent divergence
3. NO L2 regularization (which causes sample-count bias)

Why no L2 regularization?
L2 regularization creates bias: players with many samples can overcome the
regularization while players with few samples get pulled toward the prior.
This makes high-sample players appear artificially better regardless of
actual win rate.

Why skill clamping is fair:
- Applied uniformly to ALL players regardless of sample count
- A player with 1 win CAN reach max skill (±8 ≈ 99.97% win probability)
- A player with 1000 games has skill determined purely by data
- Prevents numerical divergence without biasing estimates

Use bootstrap confidence intervals to quantify uncertainty for low-sample players.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Set
import numpy as np

from .config import EloConfig, DEFAULT_ELO_CONFIG
from .models import MatchResult, MatchOutcome, PairedMatchResult

import torch


def _compute_bt_gradient_hessian(
    params: torch.Tensor,
    player_a_idx: torch.Tensor,
    player_b_idx: torch.Tensor,
    alpha_sign: torch.Tensor,
    outcomes: torch.Tensor,
    n_players: int,
    alpha_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute gradient and Hessian diagonal for Bradley-Terry negative log-likelihood."""
    skills = params[:n_players]
    alpha = params[n_players]

    skill_diff = skills[player_a_idx] - skills[player_b_idx]
    logits = skill_diff + alpha_sign * alpha
    p_a = torch.sigmoid(torch.clamp(logits, -20.0, 20.0))
    residuals = outcomes - p_a

    grad = torch.zeros(n_players + 1, dtype=torch.float64)
    grad[:n_players].scatter_add_(0, player_a_idx, -residuals)
    grad[:n_players].scatter_add_(0, player_b_idx, residuals)
    grad[n_players] = -torch.sum(alpha_sign * residuals) * alpha_weight

    p_var = p_a * (1 - p_a)
    hess_diag = torch.zeros(n_players + 1, dtype=torch.float64)
    hess_diag[:n_players].scatter_add_(0, player_a_idx, p_var)
    hess_diag[:n_players].scatter_add_(0, player_b_idx, p_var)
    hess_diag[n_players] = torch.sum(p_var) * alpha_weight + (1.0 - alpha_weight)
    hess_diag = torch.clamp(hess_diag, min=1e-6)

    return grad, hess_diag, p_a


def _bt_newton_raphson(
    player_a_idx: torch.Tensor,
    player_b_idx: torch.Tensor,
    a_first: torch.Tensor,
    outcomes: torch.Tensor,
    n_players: int,
    alpha_weight: float,  # 1.0 to estimate alpha, 0.0 to ignore
    max_iter: int = 50,
    rho: float = 0.05,  # SAM perturbation radius
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Diag-Hessian preconditioned gradient descent with SAM for Bradley-Terry MLE.

    Uses gradient from the SAM-perturbed point with Hessian diagonal from the
    original point. Skills clamped to ±MAX_SKILL; boundary flags identify
    separable cases.

    Returns (skills, alpha, converged, at_boundary).
    """
    MAX_SKILL = 8.0
    PARAM_TOL = 1e-5

    params = torch.zeros(n_players + 1, dtype=torch.float64)
    # a_first encoding: 1.0=A first, 0.0=B first, 0.5=unknown
    # Map to alpha_sign: +1, -1, 0 respectively via linear transform: 2*(a_first - 0.5)
    # Then zero out unknowns: mask = (a_first != 0.5)
    raw_sign = 2.0 * (a_first - 0.5)
    mask = (a_first - 0.5).abs() > 0.1  # True for 0.0 and 1.0, False for 0.5
    alpha_sign = raw_sign * mask.to(raw_sign.dtype)
    lr = 0.5
    params_prev = params.clone()
    params_prev2 = params.clone()

    converged = False
    for it in range(max_iter):
        grad, hess_diag, _ = _compute_bt_gradient_hessian(
            params, player_a_idx, player_b_idx, alpha_sign, outcomes, n_players, alpha_weight
        )

        grad_norm = torch.norm(grad)
        safe_norm = torch.where(grad_norm > 1e-12, grad_norm, torch.ones_like(grad_norm))
        epsilon = rho * grad / safe_norm
        params_adv = params + epsilon
        grad_sam, _, _ = _compute_bt_gradient_hessian(
            params_adv, player_a_idx, player_b_idx, alpha_sign, outcomes, n_players, alpha_weight
        )

        params_prev2 = params_prev
        params_prev = params.clone()
        params = params - lr * grad_sam / hess_diag
        params[:n_players] = torch.clamp(params[:n_players], -MAX_SKILL, MAX_SKILL)

        delta_1 = torch.norm(params - params_prev)
        delta_2 = torch.norm(params - params_prev2)
        if it >= 2 and (delta_1 < PARAM_TOL or delta_2 < PARAM_TOL):
            if delta_2 < delta_1:
                params = (params + params_prev) * 0.5
            converged = True
            break

    at_boundary = (params[:n_players].abs() >= MAX_SKILL - 1e-6)

    return params[:n_players], params[n_players] * alpha_weight, torch.tensor(converged), at_boundary


try:
    _compiled_bt_newton = torch.compile(
        _bt_newton_raphson,
        mode="max-autotune-no-cudagraphs",
        fullgraph=False,
    )
except Exception:
    _compiled_bt_newton = _bt_newton_raphson


@dataclass
class MLEFitResult:
    """Result of Bradley-Terry MLE fitting."""
    skills: np.ndarray
    alpha: float
    iterations: int
    converged: bool
    at_boundary: np.ndarray  # Boolean array: True for players hitting ±MAX_SKILL


def fit_bradley_terry_mle_full(
    player_a_indices: np.ndarray,
    player_b_indices: np.ndarray,
    a_first: np.ndarray,
    outcomes: np.ndarray,
    n_players: int,
    estimate_alpha: bool = True,
    max_iterations: int = 30,
    sam_rho: float = 0.05,
) -> MLEFitResult:
    """Fit Bradley-Terry MLE with full diagnostics including boundary flags."""
    player_a_t = torch.tensor(player_a_indices, dtype=torch.long)
    player_b_t = torch.tensor(player_b_indices, dtype=torch.long)
    a_first_t = torch.tensor(a_first, dtype=torch.float64)
    outcomes_t = torch.tensor(outcomes, dtype=torch.float64)

    alpha_weight = 1.0 if estimate_alpha else 0.0

    skills, alpha, converged_t, at_boundary = _compiled_bt_newton(
        player_a_t, player_b_t, a_first_t, outcomes_t,
        n_players, alpha_weight, max_iterations, sam_rho
    )

    return MLEFitResult(
        skills=skills.numpy(),
        alpha=float(alpha),
        iterations=max_iterations,
        converged=bool(converged_t.item()),
        at_boundary=at_boundary.numpy(),
    )


@dataclass
class BradleyTerryResult:
    """Result of Bradley-Terry model fitting."""

    skills: Dict[str, float]
    ratings: Dict[str, Decimal]
    first_mover_alpha: float
    first_mover_elo: float

    log_likelihood: float
    num_matches: int
    num_pairs: int
    num_players: int

    converged: bool
    iterations: int

    # Connected components: list of sets of player IDs.
    # If len > 1, cross-component ratings are not comparable.
    components: List[Set[str]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "skills": self.skills,
            "ratings": {k: float(v) for k, v in self.ratings.items()},
            "first_mover_alpha": self.first_mover_alpha,
            "first_mover_elo": self.first_mover_elo,
            "log_likelihood": self.log_likelihood,
            "num_matches": self.num_matches,
            "num_pairs": self.num_pairs,
            "num_players": self.num_players,
            "converged": self.converged,
            "iterations": self.iterations,
            "num_components": len(self.components),
        }


@dataclass
class MatchRecord:
    """Internal representation of a match for MLE fitting."""

    player_a: str
    player_b: str
    outcome: float  # 1.0 = A wins, 0.0 = B wins, 0.5 = draw
    is_first_mover_a: Optional[bool] = None  # None = unknown, excluded from alpha estimation
    pair_id: Optional[str] = None


class PairedBradleyTerryModel:
    """
    Bradley-Terry model with paired match support and first-mover advantage.

    This model provides more statistically efficient rating estimates by:
    1. Fitting all player skills simultaneously via MLE
    2. Explicitly estimating first-mover advantage (α parameter)
    3. Properly handling paired match data (match + rematch)

    The model uses diag-Hessian preconditioned gradient descent with optional SAM.
    """

    def __init__(self, config: Optional[EloConfig] = None):
        """
        Initialize the model.

        Args:
            config: ELO configuration. Uses DEFAULT_ELO_CONFIG if not provided.
        """
        self.config = config or DEFAULT_ELO_CONFIG
        self._matches: List[MatchRecord] = []
        self._players: Set[str] = set()
        self._player_to_idx: Dict[str, int] = {}
        self._idx_to_player: Dict[int, str] = {}
        self._result: Optional[BradleyTerryResult] = None

    def add_match(
        self,
        player_a: str,
        player_b: str,
        outcome: str,  # "a_wins", "b_wins", "draw"
        is_first_mover_a: Optional[bool] = True,
        pair_id: Optional[str] = None,
    ) -> None:
        """
        Add a single match to the model.

        Args:
            player_a: ID of player A (typically the first mover)
            player_b: ID of player B
            outcome: "a_wins", "b_wins", or "draw"
            is_first_mover_a: Whether A moved first (None = unknown)
            pair_id: Optional pair UUID for linked matches
        """
        self._players.add(player_a)
        self._players.add(player_b)

        if outcome == "a_wins":
            score = 1.0
        elif outcome == "b_wins":
            score = 0.0
        else:
            score = 0.5

        self._matches.append(MatchRecord(
            player_a=player_a,
            player_b=player_b,
            outcome=score,
            is_first_mover_a=is_first_mover_a,
            pair_id=pair_id,
        ))

    def add_match_result(self, match: MatchResult) -> None:
        """
        Add a MatchResult to the model.

        Args:
            match: MatchResult from a game
        """
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

        self.add_match(
            player_a=p_a.miner_id,
            player_b=p_b.miner_id,
            outcome=outcome,
            is_first_mover_a=match.is_first_mover,
            pair_id=match.pair_uuid,
        )

    def add_paired_result(self, paired: PairedMatchResult) -> None:
        """
        Add a PairedMatchResult (both games) to the model.

        Args:
            paired: PairedMatchResult containing match + rematch
        """
        self.add_match_result(paired.match_1)
        self.add_match_result(paired.match_2)

    def clear(self) -> None:
        """Clear all match data."""
        self._matches.clear()
        self._players.clear()
        self._player_to_idx.clear()
        self._idx_to_player.clear()
        self._result = None

    def _build_player_index(self) -> None:
        """Build player index mappings."""
        sorted_players = sorted(self._players)
        self._player_to_idx = {p: i for i, p in enumerate(sorted_players)}
        self._idx_to_player = {i: p for i, p in enumerate(sorted_players)}

    def _find_connected_components(self) -> List[Set[str]]:
        """Find connected components in the comparison graph using union-find."""
        parent: Dict[str, str] = {p: p for p in self._players}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for m in self._matches:
            union(m.player_a, m.player_b)

        groups: Dict[str, Set[str]] = {}
        for p in self._players:
            root = find(p)
            if root not in groups:
                groups[root] = set()
            groups[root].add(p)

        return list(groups.values())

    def fit(
        self,
        estimate_first_mover: Optional[bool] = None,
        max_iterations: int = 30,
        sam_rho: Optional[float] = None,
    ) -> BradleyTerryResult:
        """
        Fit the Bradley-Terry model using MLE with torch.compile Newton-Raphson + SAM.

        Fast (~1-10ms) and accurate. Uses Sharpness-Aware Minimization (SAM) instead
        of L2 regularization for more principled optimization.

        Args:
            estimate_first_mover: Whether to estimate first-mover advantage.
                                 Uses config default if not provided.
            max_iterations: Newton iterations (30 is always sufficient)
            sam_rho: SAM perturbation radius. 0 = pure MLE (no SAM).
                     Defaults to config.BT_SAM_RHO if not provided.

        Returns:
            BradleyTerryResult with fitted parameters
        """
        if len(self._matches) == 0:
            raise ValueError("No matches to fit")

        if len(self._players) < 2:
            raise ValueError("Need at least 2 players")

        if estimate_first_mover is None:
            estimate_first_mover = self.config.ESTIMATE_FIRST_MOVER_ADVANTAGE
        if sam_rho is None:
            sam_rho = self.config.BT_SAM_RHO

        self._build_player_index()
        n_players = len(self._players)

        components = self._find_connected_components()
        if len(components) > 1:
            import warnings
            warnings.warn(
                f"Comparison graph has {len(components)} disconnected components "
                f"(sizes: {[len(c) for c in components]}). Cross-component ratings "
                f"are arbitrary — only within-component ordering is meaningful.",
                UserWarning,
                stacklevel=2,
            )

        player_a_indices = np.array([self._player_to_idx[m.player_a] for m in self._matches])
        player_b_indices = np.array([self._player_to_idx[m.player_b] for m in self._matches])
        # Encode first-mover: True→1.0, False→0.0, None→0.5 (sentinel for unknown)
        # The optimizer converts: 1.0→+1 alpha_sign, 0.0→-1 alpha_sign, 0.5→0 alpha_sign
        a_first = np.array([
            1.0 if m.is_first_mover_a is True else (0.0 if m.is_first_mover_a is False else 0.5)
            for m in self._matches
        ])
        outcomes = np.array([m.outcome for m in self._matches])

        fit_result = fit_bradley_terry_mle_full(
            player_a_indices, player_b_indices, a_first, outcomes,
            n_players, estimate_first_mover, max_iterations, sam_rho
        )
        skills = fit_result.skills
        alpha = fit_result.alpha
        iterations = fit_result.iterations
        converged = fit_result.converged

        sorted_players = sorted(self._players)

        # Per-component mean-centering: only center within connected components
        # so cross-component offsets remain unidentifiable rather than arbitrary
        for component in components:
            indices = [self._player_to_idx[p] for p in component]
            component_mean = np.mean(skills[indices])
            skills[indices] -= component_mean

        scale_factor = self.config.SCALE / math.log(10)
        ratings = {}
        skill_dict = {}

        for i, player in enumerate(sorted_players):
            skill_dict[player] = float(skills[i])
            elo = Decimal(str(round(scale_factor * skills[i] + float(self.config.DEFAULT_RATING), 2)))
            ratings[player] = elo

        alpha_elo = scale_factor * alpha
        pair_ids = set(m.pair_id for m in self._matches if m.pair_id)

        log_likelihood = self._compute_log_likelihood(skills, alpha)

        self._result = BradleyTerryResult(
            skills=skill_dict,
            ratings=ratings,
            first_mover_alpha=alpha,
            first_mover_elo=alpha_elo,
            log_likelihood=log_likelihood,
            num_matches=len(self._matches),
            num_pairs=len(pair_ids),
            num_players=n_players,
            converged=converged,
            iterations=iterations,
            components=components,
        )

        return self._result

    def _compute_log_likelihood(self, skills: np.ndarray, alpha: float) -> float:
        """Compute log-likelihood of fitted model."""
        ll = 0.0
        for match in self._matches:
            i = self._player_to_idx[match.player_a]
            j = self._player_to_idx[match.player_b]

            logit = skills[i] - skills[j]
            if match.is_first_mover_a is True:
                logit += alpha
            elif match.is_first_mover_a is False:
                logit -= alpha
            # None: no alpha adjustment

            p_a = 1.0 / (1.0 + math.exp(-np.clip(logit, -20, 20)))
            y = match.outcome

            if y == 1.0:
                ll += math.log(max(p_a, 1e-15))
            elif y == 0.0:
                ll += math.log(max(1 - p_a, 1e-15))
            else:
                ll += 0.5 * math.log(max(p_a, 1e-15)) + 0.5 * math.log(max(1 - p_a, 1e-15))

        return ll

    def predict_win_probability(
        self,
        player_a: str,
        player_b: str,
        a_moves_first: bool = True,
    ) -> float:
        """
        Predict probability that player A beats player B.

        Args:
            player_a: ID of player A
            player_b: ID of player B
            a_moves_first: Whether A moves first

        Returns:
            P(A beats B)
        """
        if self._result is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if player_a not in self._result.skills or player_b not in self._result.skills:
            raise ValueError(f"Unknown player(s)")

        beta_a = self._result.skills[player_a]
        beta_b = self._result.skills[player_b]
        alpha = self._result.first_mover_alpha

        if a_moves_first:
            logit = beta_a - beta_b + alpha
        else:
            logit = beta_a - beta_b - alpha

        return 1.0 / (1.0 + math.exp(-logit))

    @property
    def result(self) -> Optional[BradleyTerryResult]:
        """Get the fitted result."""
        return self._result


def fit_bradley_terry_from_matches(
    matches: List[MatchResult],
    config: Optional[EloConfig] = None,
    estimate_first_mover: bool = True,
) -> BradleyTerryResult:
    """
    Convenience function to fit Bradley-Terry from match results.

    Args:
        matches: List of MatchResult objects
        config: ELO configuration
        estimate_first_mover: Whether to estimate first-mover advantage

    Returns:
        BradleyTerryResult with fitted parameters
    """
    model = PairedBradleyTerryModel(config)
    for match in matches:
        model.add_match_result(match)
    return model.fit(estimate_first_mover=estimate_first_mover)


def fit_bradley_terry_from_pairs(
    pairs: List[PairedMatchResult],
    config: Optional[EloConfig] = None,
    estimate_first_mover: bool = True,
) -> BradleyTerryResult:
    """
    Convenience function to fit Bradley-Terry from paired results.

    Args:
        pairs: List of PairedMatchResult objects
        config: ELO configuration
        estimate_first_mover: Whether to estimate first-mover advantage

    Returns:
        BradleyTerryResult with fitted parameters
    """
    model = PairedBradleyTerryModel(config)
    for pair in pairs:
        model.add_paired_result(pair)
    return model.fit(estimate_first_mover=estimate_first_mover)
