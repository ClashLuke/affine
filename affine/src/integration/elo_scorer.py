"""
ELO-Integrated Scorer

A scorer that uses ELO ratings instead of or in addition to absolute scores.
This can be used as a drop-in replacement for the standard Scorer to test
ELO-based weight calculation.
"""

import time
from decimal import Decimal
from typing import Dict, Any, Optional, List

from affine.core.setup import logger
from affine.src.scorer.config import ScorerConfig
from affine.src.scorer.models import ScoringResult, MinerData
from affine.src.scorer.stage1_collector import Stage1Collector
from affine.src.scorer.stage4_weights import Stage4WeightNormalizer

from affine.src.elo.config import EloConfig
from affine.src.elo.match_engine import PairwiseMatchGenerator, apply_matches_to_ratings
from affine.src.elo.models import EloRating, SampleScore
from affine.src.integration.elo_validator import calculate_weights_from_ratings


class EloScorer:
    """
    Scorer that uses ELO ratings for weight calculation.

    Replaces the standard 4-stage pipeline with:
    1. Data Collection (same as Stage 1)
    2. ELO Match Generation & Rating Update
    3. ELO-to-Weight Conversion
    4. Weight Normalization (same as Stage 4)
    """

    def __init__(
        self,
        scorer_config: Optional[ScorerConfig] = None,
        elo_config: Optional[EloConfig] = None,
    ):
        """
        Initialize the ELO scorer.

        Args:
            scorer_config: Configuration for basic scoring
            elo_config: Configuration for ELO calculations
        """
        self.scorer_config = scorer_config or ScorerConfig()
        self.elo_config = elo_config or EloConfig()

        self.stage1 = Stage1Collector(self.scorer_config)
        self.pairwise_generator = PairwiseMatchGenerator(self.elo_config)
        self.stage4 = Stage4WeightNormalizer(self.scorer_config)
        self.ratings: Dict[str, EloRating] = {}
        self.total_matches: int = 0

    def convert_to_samples(
        self,
        scoring_data: Dict[str, Any],
        environments: List[str],
    ) -> List[SampleScore]:
        """
        Convert scoring API data to SampleScore objects.

        Args:
            scoring_data: Response from /samples/scoring API
            environments: List of environment names

        Returns:
            List of SampleScore objects
        """
        samples = []
        timestamp = int(time.time() * 1000)

        for hotkey, miner_data in scoring_data.items():
            revision = miner_data.get("model_revision", "unknown")

            for env in environments:
                env_data = miner_data.get("environments", {}).get(env, {})
                scores = env_data.get("scores", [])

                for i, score_entry in enumerate(scores):
                    if isinstance(score_entry, dict):
                        score_val = score_entry.get("score", 0)
                        task_id = score_entry.get("task_id", i)
                    else:
                        score_val = score_entry
                        task_id = i

                    if score_val < 0:
                        continue

                    samples.append(SampleScore(
                        miner_hotkey=hotkey,
                        model_revision=revision,
                        env=env,
                        task_id=task_id,
                        score=Decimal(str(score_val)),
                        timestamp=timestamp,
                    ))

        return samples

    def process_matches(self, samples: List[SampleScore]) -> int:
        """Generate pairwise matches and update ELO ratings."""
        matches = self.pairwise_generator.generate_matches_from_samples(
            samples, self.ratings,
        )
        apply_matches_to_ratings(matches, self.ratings, self.elo_config)
        self.total_matches += len(matches)
        return len(matches)

    def calculate_elo_weights(
        self,
        miners: Dict[int, MinerData],
    ) -> Dict[int, float]:
        """Calculate weights from ELO ratings. Returns {uid: normalized_weight}."""
        results = calculate_weights_from_ratings(self.ratings)

        hk_rev_to_uid = {
            f"{m.hotkey}#{m.model_revision}": uid
            for uid, m in miners.items()
        }
        weights = {uid: 0.0 for uid in miners}
        for r in results:
            uid = hk_rev_to_uid.get(f"{r.miner_hotkey}#{r.model_revision}")
            if uid is not None:
                weights[uid] = r.normalized_weight
        return weights

    def calculate_scores(
        self,
        scoring_data: Dict[str, Any],
        environments: List[str],
        env_configs: Dict[str, Any],
        block_number: int,
        print_summary: bool = True,
    ) -> ScoringResult:
        """
        Execute ELO-based scoring.

        Args:
            scoring_data: Response from /api/v1/samples/scoring
            environments: List of environment names
            env_configs: Environment configurations
            block_number: Current block number
            print_summary: Whether to print summary

        Returns:
            ScoringResult with ELO-based weights
        """
        if not self.elo_config.ELO_ENABLED:
            logger.warning("[ELO Scorer] ELO_ENABLED is False — skipping ELO processing")
            stage1_output = self.stage1.collect(scoring_data, environments, env_configs)
            return ScoringResult(
                block_number=block_number,
                calculated_at=int(time.time()),
                environments=environments,
                config=self.scorer_config.to_dict(),
                miners=stage1_output.miners,
                pareto_comparisons=[],
                subsets=[],
                final_weights={},
                total_miners=len(scoring_data),
                valid_miners=stage1_output.valid_count,
                invalid_miners=stage1_output.invalid_count,
            )

        start_time = time.time()
        logger.info(f"[ELO Scorer] Processing {len(scoring_data)} miners")

        stage1_output = self.stage1.collect(scoring_data, environments, env_configs)
        logger.info(f"[ELO Scorer] Stage 1: {stage1_output.valid_count} valid miners")

        valid_hotkeys = {m.hotkey for m in stage1_output.miners.values()}

        samples = [
            s for s in self.convert_to_samples(scoring_data, environments)
            if s.miner_hotkey in valid_hotkeys
        ]
        match_count = self.process_matches(samples)
        logger.info(f"[ELO Scorer] Processed {match_count} pairwise matches")
        logger.info(f"[ELO Scorer] Total ELO ratings: {len(self.ratings)}")

        elo_weights = self.calculate_elo_weights(stage1_output.miners)

        for uid, miner in stage1_output.miners.items():
            miner.cumulative_weight = elo_weights.get(uid, 0.0)

        stage4_output = self.stage4.normalize(stage1_output.miners)

        result = ScoringResult(
            block_number=block_number,
            calculated_at=int(time.time()),
            environments=environments,
            config=self.scorer_config.to_dict(),
            miners=stage1_output.miners,
            pareto_comparisons=[],
            subsets=[],
            final_weights=stage4_output.final_weights,
            total_miners=len(scoring_data),
            valid_miners=stage1_output.valid_count,
            invalid_miners=stage1_output.invalid_count,
        )

        elapsed_time = time.time() - start_time
        non_zero = len([w for w in result.final_weights.values() if w > 0])

        logger.info("=" * 80)
        logger.info(f"[ELO SCORING] Time: {elapsed_time:.2f}s, Active: {non_zero}/{len(scoring_data)}")
        logger.info(f"[ELO SCORING] Total matches: {self.total_matches}, Ratings: {len(self.ratings)}")
        logger.info("=" * 80)

        if print_summary:
            self._print_summary(result, environments)

        return result

    def _print_summary(self, result: ScoringResult, environments: List[str]):
        """Print ELO scoring summary."""
        logger.info("\nELO WEIGHT SUMMARY (Top 10):")
        logger.info("-" * 60)

        sorted_weights = sorted(
            result.final_weights.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        for uid, weight in sorted_weights[:10]:
            miner = result.miners.get(uid)
            if miner:
                env_elos = []
                for env in environments:
                    key = f"{miner.hotkey}#{miner.model_revision}#{env}"
                    rating = self.ratings.get(key)
                    if rating and rating.matches_played > 0:
                        env_elos.append(float(rating.rating))
                avg_elo = sum(env_elos) / len(env_elos) if env_elos else 1500.0
                logger.info(
                    f"UID {uid:3d}: weight={weight:.6f}, "
                    f"avg_elo={avg_elo:.1f}"
                )

    def get_leaderboard(self, env: Optional[str] = None, limit: int = 32) -> List[Dict[str, Any]]:
        """
        Get ELO leaderboard.

        Args:
            env: Optional environment filter
            limit: Maximum number of entries

        Returns:
            List of leaderboard entries
        """
        if env:
            ratings = [r for r in self.ratings.values() if r.env == env]
        else:
            miner_ratings: Dict[str, List[EloRating]] = {}
            for r in self.ratings.values():
                key = f"{r.miner_hotkey}#{r.model_revision}"
                if key not in miner_ratings:
                    miner_ratings[key] = []
                miner_ratings[key].append(r)

            ratings = []
            for key, env_ratings in miner_ratings.items():
                played = [r for r in env_ratings if r.matches_played > 0]
                if not played:
                    continue
                avg_rating = sum(float(r.rating) for r in played) / len(played)
                total_matches = sum(r.matches_played for r in played)
                total_wins = sum(r.wins for r in played)

                parts = key.split("#", 1)
                agg = EloRating(
                    miner_hotkey=parts[0],
                    model_revision=parts[1] if len(parts) > 1 else "",
                    env="aggregate",
                    rating=Decimal(str(avg_rating)),
                    matches_played=total_matches,
                    wins=total_wins,
                )
                ratings.append(agg)

        ratings.sort(key=lambda r: r.rating, reverse=True)

        return [
            {
                "rank": i + 1,
                "miner_hotkey": r.miner_hotkey,
                "model_revision": r.model_revision,
                "env": r.env,
                "rating": float(r.rating),
                "matches_played": r.matches_played,
                "win_rate": r.win_rate,
            }
            for i, r in enumerate(ratings[:limit])
        ]


def create_elo_scorer(
    scorer_config: Optional[ScorerConfig] = None,
    elo_config: Optional[EloConfig] = None,
) -> EloScorer:
    """Factory function to create an EloScorer."""
    return EloScorer(scorer_config, elo_config)
