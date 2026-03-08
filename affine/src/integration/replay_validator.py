"""
Replay Validator

Validates the ELO system using historical scoring data.
This allows testing without running live evaluations.

Key features:
- Loads historical scoring snapshots from database
- Replays scoring rounds to build ELO history
- Compares ELO rankings with actual weight history
- Validates ELO convergence and stability

Usage:
    python -m affine.src.integration.replay_validator --start-block 1000000 --end-block 1001000
"""

import asyncio
import json
import time
from decimal import Decimal
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field

from affine.core.setup import logger
from affine.database.dao.scores import ScoresDAO

from affine.src.elo.config import EloConfig
from affine.src.elo.match_engine import PairwiseMatchGenerator, apply_matches_to_ratings
from affine.src.elo.models import EloRating, SampleScore, MatchResult
from affine.src.integration.elo_validator import calculate_weights_from_ratings


@dataclass
class ReplayRound:
    """Results from replaying a single scoring round."""
    block_number: int
    timestamp: int

    # Historical data
    historical_weights: Dict[int, float] = field(default_factory=dict)
    historical_top_miners: List[int] = field(default_factory=list)

    # ELO results
    elo_weights: Dict[int, float] = field(default_factory=dict)
    elo_top_miners: List[int] = field(default_factory=list)

    # Match stats
    samples_processed: int = 0
    matches_generated: int = 0

    # Comparison
    rank_correlation: float = 0.0
    top_k_overlap: int = 0


@dataclass
class ReplayResult:
    """Complete replay validation result."""
    start_block: int
    end_block: int
    rounds_processed: int

    # Aggregated metrics
    avg_correlation: float = 0.0
    avg_top_k_overlap: float = 0.0
    total_matches: int = 0
    final_ratings_count: int = 0

    # Round details
    rounds: List[ReplayRound] = field(default_factory=list)

    # Final state
    final_elo_ratings: Dict[str, float] = field(default_factory=dict)
    final_leaderboard: List[Dict[str, Any]] = field(default_factory=list)


class ReplayValidator:
    """
    Replays historical scoring data to validate ELO system.
    """

    def __init__(
        self,
        elo_config: Optional[EloConfig] = None,
    ):
        self.elo_config = elo_config or EloConfig()
        self.pairwise_generator = PairwiseMatchGenerator(self.elo_config)
        self.ratings: Dict[str, EloRating] = {}
        self.all_matches: List[MatchResult] = []

    def extract_samples_from_scores(
        self,
        scores_data: List[Dict[str, Any]],
        block_number: int,
    ) -> List[SampleScore]:
        """
        Extract sample scores from historical scores data.

        Args:
            scores_data: List of score records from database
            block_number: Block number (used as timestamp proxy)

        Returns:
            List of SampleScore objects
        """
        samples = []
        timestamp = block_number * 12000  # Approximate ms per block

        for score_record in scores_data:
            hotkey = score_record.get("miner_hotkey", "")
            revision = score_record.get("model_revision", "unknown")
            scores_by_env = score_record.get("scores_by_env", {})

            for env, env_data in scores_by_env.items():
                if isinstance(env_data, dict):
                    score = env_data.get("score", 0)
                else:
                    score = env_data

                if score > 0:
                    # Create a sample for each env score
                    # Use block_number as task_id since we don't have original task_ids
                    samples.append(SampleScore(
                        miner_hotkey=hotkey,
                        model_revision=revision,
                        env=env,
                        task_id=block_number,
                        score=Decimal(str(score)),
                        timestamp=timestamp,
                    ))

        return samples

    def process_matches(self, samples: List[SampleScore]) -> List[MatchResult]:
        """Generate pairwise matches and update ratings."""
        matches = self.pairwise_generator.generate_matches_from_samples(
            samples, self.ratings,
        )
        apply_matches_to_ratings(matches, self.ratings, self.elo_config)
        self.all_matches.extend(matches)
        return matches

    def calculate_elo_weights(
        self,
        scores_data: List[Dict[str, Any]],
    ) -> Dict[int, float]:
        """Calculate weights from current ELO state. Returns {uid: normalized_weight}."""
        results = calculate_weights_from_ratings(self.ratings)

        hk_rev_to_uid = {}
        for record in scores_data:
            uid = record.get("uid")
            if uid is None:
                continue
            hk = record.get("miner_hotkey", "")
            rev = record.get("model_revision", "unknown")
            hk_rev_to_uid[f"{hk}#{rev}"] = uid

        weights = {}
        for r in results:
            uid = hk_rev_to_uid.get(f"{r.miner_hotkey}#{r.model_revision}")
            if uid is not None:
                weights[uid] = r.normalized_weight

        return weights

    def compare_rankings(
        self,
        historical: Dict[int, float],
        elo: Dict[int, float],
        top_k: int = 32,
    ) -> Tuple[float, int]:
        """
        Compare historical and ELO rankings.

        Returns:
            Tuple of (rank_correlation, top_k_overlap)
        """
        from scipy import stats

        common_uids = set(historical.keys()) & set(elo.keys())

        if len(common_uids) < 2:
            return 0.0, 0

        hist_ranks = {uid: i for i, (uid, _) in enumerate(
            sorted(historical.items(), key=lambda x: x[1], reverse=True)
        )}
        elo_ranks = {uid: i for i, (uid, _) in enumerate(
            sorted(elo.items(), key=lambda x: x[1], reverse=True)
        )}

        hist_list = [hist_ranks.get(uid, len(hist_ranks)) for uid in common_uids]
        elo_list = [elo_ranks.get(uid, len(elo_ranks)) for uid in common_uids]

        try:
            correlation, _ = stats.spearmanr(hist_list, elo_list)
        except Exception:
            correlation = 0.0

        hist_top = set(list(hist_ranks.keys())[:top_k])
        elo_top = set(list(elo_ranks.keys())[:top_k])
        overlap = len(hist_top & elo_top)

        return float(correlation), overlap

    async def replay_block(
        self,
        block_number: int,
        scores_dao: ScoresDAO,
    ) -> Optional[ReplayRound]:
        """
        Replay a single block's scoring data.

        Args:
            block_number: Block number to replay
            scores_dao: ScoresDAO for fetching historical data

        Returns:
            ReplayRound with results, or None if no data
        """
        # Fetch historical scores for this block
        scores_data = await scores_dao.get_scores_for_block(block_number)

        if not scores_data:
            return None

        # Extract historical weights
        historical_weights = {
            s["uid"]: s.get("overall_score", 0.0)
            for s in scores_data
            if s.get("uid") is not None
        }

        # Normalize historical weights
        total = sum(historical_weights.values())
        if total > 0:
            historical_weights = {uid: w / total for uid, w in historical_weights.items()}

        # Extract samples and process matches
        samples = self.extract_samples_from_scores(scores_data, block_number)
        matches = self.process_matches(samples)

        # Calculate ELO weights
        elo_weights = self.calculate_elo_weights(scores_data)

        # Compare
        correlation, overlap = self.compare_rankings(historical_weights, elo_weights)

        return ReplayRound(
            block_number=block_number,
            timestamp=int(time.time()),
            historical_weights=historical_weights,
            historical_top_miners=sorted(
                historical_weights.keys(),
                key=lambda uid: historical_weights.get(uid, 0),
                reverse=True,
            )[:32],
            elo_weights=elo_weights,
            elo_top_miners=sorted(
                elo_weights.keys(),
                key=lambda uid: elo_weights.get(uid, 0),
                reverse=True,
            )[:32],
            samples_processed=len(samples),
            matches_generated=len(matches),
            rank_correlation=correlation,
            top_k_overlap=overlap,
        )

    async def replay(
        self,
        start_block: int,
        end_block: int,
        step: int = 180,  # ~30 min worth of blocks
    ) -> ReplayResult:
        """
        Replay multiple blocks of historical data.

        Args:
            start_block: Starting block number
            end_block: Ending block number
            step: Block step size (default: 180 blocks ≈ 30 min)

        Returns:
            ReplayResult with complete analysis
        """
        logger.info(f"Starting replay: blocks {start_block} - {end_block}, step {step}")

        scores_dao = ScoresDAO()
        rounds = []

        block = start_block
        while block <= end_block:
            logger.info(f"Processing block {block}...")

            try:
                round_result = await self.replay_block(block, scores_dao)

                if round_result:
                    rounds.append(round_result)
                    logger.info(
                        f"  Samples: {round_result.samples_processed}, "
                        f"Matches: {round_result.matches_generated}, "
                        f"Correlation: {round_result.rank_correlation:.3f}"
                    )
                else:
                    logger.warning(f"  No data for block {block}")

            except Exception as e:
                logger.error(f"  Error processing block {block}: {e}")

            block += step

        # Calculate aggregates
        if rounds:
            avg_correlation = sum(r.rank_correlation for r in rounds) / len(rounds)
            avg_overlap = sum(r.top_k_overlap for r in rounds) / len(rounds)
        else:
            avg_correlation = 0.0
            avg_overlap = 0.0

        # Build final leaderboard
        final_leaderboard = sorted(
            [
                {
                    "hotkey": r.miner_hotkey[:16] + "...",
                    "revision": r.model_revision[:8],
                    "env": r.env,
                    "rating": float(r.rating),
                    "matches": r.matches_played,
                    "win_rate": r.win_rate,
                }
                for r in self.ratings.values()
            ],
            key=lambda x: x["rating"],
            reverse=True,
        )[:50]

        result = ReplayResult(
            start_block=start_block,
            end_block=end_block,
            rounds_processed=len(rounds),
            avg_correlation=avg_correlation,
            avg_top_k_overlap=avg_overlap,
            total_matches=len(self.all_matches),
            final_ratings_count=len(self.ratings),
            rounds=rounds,
            final_elo_ratings={
                k: float(v.rating) for k, v in self.ratings.items()
            },
            final_leaderboard=final_leaderboard,
        )

        self._print_summary(result)

        return result

    def _print_summary(self, result: ReplayResult):
        """Print replay summary."""
        logger.info("\n" + "=" * 80)
        logger.info("REPLAY VALIDATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Blocks: {result.start_block} - {result.end_block}")
        logger.info(f"Rounds processed: {result.rounds_processed}")
        logger.info(f"Total matches: {result.total_matches}")
        logger.info(f"Final ratings: {result.final_ratings_count}")
        logger.info("")
        logger.info("CORRELATION WITH HISTORICAL:")
        logger.info(f"  Average rank correlation: {result.avg_correlation:.4f}")
        logger.info(f"  Average top-32 overlap: {result.avg_top_k_overlap:.1f}")
        logger.info("")
        logger.info("FINAL ELO LEADERBOARD (Top 10):")
        for i, entry in enumerate(result.final_leaderboard[:10], 1):
            logger.info(
                f"  {i:2d}. {entry['hotkey']} | "
                f"ELO: {entry['rating']:.0f} | "
                f"Matches: {entry['matches']} | "
                f"Win: {entry['win_rate']:.1%}"
            )


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Replay Validator for ELO Testing")
    parser.add_argument("--start-block", type=int, required=True, help="Starting block number")
    parser.add_argument("--end-block", type=int, required=True, help="Ending block number")
    parser.add_argument("--step", type=int, default=180, help="Block step size")
    parser.add_argument("--output", type=str, help="Output JSON file for results")

    args = parser.parse_args()

    validator = ReplayValidator()
    result = await validator.replay(
        start_block=args.start_block,
        end_block=args.end_block,
        step=args.step,
    )

    if args.output:
        with open(args.output, "w") as f:
            json.dump({
                "start_block": result.start_block,
                "end_block": result.end_block,
                "rounds_processed": result.rounds_processed,
                "avg_correlation": result.avg_correlation,
                "avg_top_k_overlap": result.avg_top_k_overlap,
                "total_matches": result.total_matches,
                "final_ratings_count": result.final_ratings_count,
                "final_leaderboard": result.final_leaderboard,
            }, f, indent=2)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
