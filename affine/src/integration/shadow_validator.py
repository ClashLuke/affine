"""
Shadow Validator

A parallel validator that runs the complete scoring pipeline without setting weights.
Used for integration testing and validation of the ELO system before production deployment.

Key features:
- Fetches real miner data and evaluations via API
- Calculates both absolute scores AND ELO-based weights
- Compares results between scoring methods
- Does NOT set weights on chain (read-only)
- Produces detailed reports for analysis

Usage:
    python -m affine.src.integration.shadow_validator --netuid 120 --rounds 10
"""

import asyncio
import time
import json
from decimal import Decimal
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, field

from affine.core.setup import logger
from affine.utils.api_client import create_api_client
from affine.utils.subtensor import get_subtensor

# Scorer components
from affine.src.scorer.scorer import create_scorer

# ELO components
from affine.src.elo.config import EloConfig
from affine.src.elo.match_engine import PairwiseMatchGenerator, apply_matches_to_ratings
from affine.src.elo.models import EloRating, MatchResult, SampleScore
from affine.src.integration.elo_validator import calculate_weights_from_ratings


@dataclass
class EloState:
    """Current ELO ratings state."""
    ratings: Dict[str, EloRating] = field(default_factory=dict)
    matches: List[MatchResult] = field(default_factory=list)


@dataclass
class ValidationReport:
    """Report comparing absolute scoring vs ELO scoring."""
    timestamp: int
    block_number: int

    # Absolute scoring results
    absolute_weights: Dict[int, float] = field(default_factory=dict)
    absolute_top_miners: List[Tuple[int, float]] = field(default_factory=list)

    # ELO scoring results
    elo_weights: Dict[int, float] = field(default_factory=dict)
    elo_top_miners: List[Tuple[int, float]] = field(default_factory=list)
    elo_ratings: Dict[str, float] = field(default_factory=dict)

    # Comparison metrics
    rank_correlation: float = 0.0  # Spearman correlation between rankings
    weight_rmse: float = 0.0  # Root mean squared error of weights
    top_k_overlap: int = 0  # How many top-K miners are in both rankings

    # Match statistics
    matches_generated: int = 0
    total_elo_updates: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "block_number": self.block_number,
            "absolute_weights": {str(k): v for k, v in self.absolute_weights.items()},
            "absolute_top_miners": self.absolute_top_miners[:10],
            "elo_weights": {str(k): v for k, v in self.elo_weights.items()},
            "elo_top_miners": self.elo_top_miners[:10],
            "elo_ratings": self.elo_ratings,
            "rank_correlation": self.rank_correlation,
            "weight_rmse": self.weight_rmse,
            "top_k_overlap": self.top_k_overlap,
            "matches_generated": self.matches_generated,
            "total_elo_updates": self.total_elo_updates,
        }


class ShadowValidator:
    """
    Shadow validator for integration testing.

    Runs the complete scoring pipeline without setting weights on chain.
    Compares absolute scoring with ELO-based scoring.
    """

    def __init__(
        self,
        netuid: int = 120,
        dry_run: bool = True,
        save_reports: bool = True,
        report_dir: str = "shadow_reports",
    ):
        self.netuid = netuid
        self.dry_run = dry_run
        self.save_reports = save_reports
        self.report_dir = report_dir

        self.absolute_scorer = create_scorer()
        self.elo_config = EloConfig()
        self.pairwise_generator = PairwiseMatchGenerator(self.elo_config)

        # State
        self.elo_state = EloState()
        self.reports: List[ValidationReport] = []

        # API client
        self.api_client = None

    async def initialize(self):
        """Initialize connections and state."""
        logger.info(f"Initializing ShadowValidator (netuid={self.netuid}, dry_run={self.dry_run})")

        self.api_client = await create_api_client()

        if not self.dry_run:
            logger.warning("⚠️ DRY_RUN is disabled - this validator CAN set weights!")
        else:
            logger.info("✓ Running in dry-run mode (weights will NOT be set)")

    async def fetch_scoring_data(self) -> Tuple[Dict[str, Any], List[str], Dict[str, Any], int]:
        """
        Fetch scoring data from the API.

        Returns:
            Tuple of (scoring_data, environments, env_configs, block_number)
        """
        # Fetch scoring data
        # Note: base_url already includes /api/v1, so we don't prefix it
        scoring_response = await self.api_client.get("/samples/scoring")

        # Get current block
        subtensor = await get_subtensor()
        block_number = await subtensor.get_current_block()

        # The API returns data directly keyed by "hotkey#revision"
        # Each value has: uid, hotkey, model_revision, env (dict of environment data)
        scoring_data = scoring_response

        # Extract environments from the first miner's data
        environments = []
        env_configs = {}
        if scoring_data:
            first_miner = next(iter(scoring_data.values()))
            if "env" in first_miner:
                environments = list(first_miner["env"].keys())
                # Create simple env configs from environment names
                env_configs = {env: {"name": env} for env in environments}

        logger.info(f"Fetched data: {len(scoring_data)} miners, {len(environments)} environments, block {block_number}")

        return scoring_data, environments, env_configs, block_number

    def convert_to_sample_scores(
        self,
        scoring_data: Dict[str, Any],
        environments: List[str],
    ) -> List[SampleScore]:
        """
        Convert API scoring data to SampleScore objects for ELO processing.

        Args:
            scoring_data: Response from /samples/scoring (keyed by hotkey#revision)
            environments: List of environment names

        Returns:
            List of SampleScore objects
        """
        samples = []

        for key, miner_data in scoring_data.items():
            # Key format is "hotkey#revision"
            hotkey = miner_data.get("hotkey", key.split("#")[0] if "#" in key else key)
            revision = miner_data.get("model_revision", "unknown")

            # Environment data is under "env" key, each env has "samples"
            env_dict = miner_data.get("env", {})

            for env in environments:
                env_data = env_dict.get(env, {})
                score_entries = env_data.get("samples", [])

                for score_entry in score_entries:
                    if isinstance(score_entry, dict):
                        score = score_entry.get("score", 0)
                        task_id = score_entry.get("task_id", 0)
                        timestamp = score_entry.get("timestamp", int(time.time() * 1000))
                    else:
                        score = score_entry
                        task_id = 0
                        timestamp = int(time.time() * 1000)

                    if score < 0:
                        continue

                    samples.append(SampleScore(
                        miner_hotkey=hotkey,
                        model_revision=revision,
                        env=env,
                        task_id=task_id,
                        score=Decimal(str(score)),
                        timestamp=timestamp,
                    ))

        return samples

    def process_elo_matches(self, samples: List[SampleScore]) -> List[MatchResult]:
        """Generate pairwise matches from samples and update ELO ratings."""
        matches = self.pairwise_generator.generate_matches_from_samples(
            samples, self.elo_state.ratings,
        )
        logger.info(f"Generated {len(matches)} pairwise matches")
        apply_matches_to_ratings(matches, self.elo_state.ratings, self.elo_config)
        self.elo_state.matches.extend(matches)
        return matches

    def calculate_elo_weights(
        self,
        scoring_data: Dict[str, Any],
    ) -> Dict[int, float]:
        """Calculate weights from ELO ratings. Returns {uid: normalized_weight}."""
        results = calculate_weights_from_ratings(self.elo_state.ratings)

        # Build hotkey#revision -> uid mapping from scoring_data
        hk_rev_to_uid = {}
        for key, miner_data in scoring_data.items():
            uid = miner_data.get("uid")
            if uid is None:
                continue
            hotkey = miner_data.get("hotkey", key.split("#")[0] if "#" in key else key)
            revision = miner_data.get("model_revision", "unknown")
            hk_rev_to_uid[f"{hotkey}#{revision}"] = uid

        weights = {}
        for r in results:
            uid = hk_rev_to_uid.get(f"{r.miner_hotkey}#{r.model_revision}")
            if uid is not None:
                weights[uid] = r.normalized_weight

        return weights

    def compare_weights(
        self,
        absolute_weights: Dict[int, float],
        elo_weights: Dict[int, float],
        top_k: int = 32,
    ) -> Tuple[float, float, int]:
        """
        Compare absolute scoring weights with ELO weights.

        Args:
            absolute_weights: Weights from absolute scoring
            elo_weights: Weights from ELO scoring
            top_k: Number of top miners to compare

        Returns:
            Tuple of (rank_correlation, weight_rmse, top_k_overlap)
        """
        import numpy as np
        from scipy import stats

        # Get common UIDs
        common_uids = set(absolute_weights.keys()) & set(elo_weights.keys())

        if len(common_uids) < 2:
            return 0.0, 0.0, 0

        # Calculate rank correlation
        abs_ranks = {uid: i for i, (uid, _) in enumerate(
            sorted(absolute_weights.items(), key=lambda x: x[1], reverse=True)
        )}
        elo_ranks = {uid: i for i, (uid, _) in enumerate(
            sorted(elo_weights.items(), key=lambda x: x[1], reverse=True)
        )}

        abs_rank_list = [abs_ranks.get(uid, len(abs_ranks)) for uid in common_uids]
        elo_rank_list = [elo_ranks.get(uid, len(elo_ranks)) for uid in common_uids]

        try:
            correlation, _ = stats.spearmanr(abs_rank_list, elo_rank_list)
        except Exception:
            correlation = 0.0

        # Calculate RMSE
        abs_arr = np.array([absolute_weights.get(uid, 0) for uid in common_uids])
        elo_arr = np.array([elo_weights.get(uid, 0) for uid in common_uids])
        rmse = np.sqrt(np.mean((abs_arr - elo_arr) ** 2))

        # Calculate top-K overlap
        abs_top_k = set(list(abs_ranks.keys())[:top_k])
        elo_top_k = set(list(elo_ranks.keys())[:top_k])
        overlap = len(abs_top_k & elo_top_k)

        return float(correlation), float(rmse), overlap

    async def run_validation_round(self) -> ValidationReport:
        """
        Run a single validation round.

        Returns:
            ValidationReport with comparison metrics
        """
        logger.info("=" * 80)
        logger.info("Starting validation round")
        logger.info("=" * 80)

        # Fetch data
        scoring_data, environments, env_configs, block_number = await self.fetch_scoring_data()

        # Run absolute scoring (existing 4-stage pipeline)
        logger.info("Running absolute scoring...")
        absolute_result = self.absolute_scorer.calculate_scores(
            scoring_data=scoring_data,
            environments=environments,
            env_configs=env_configs,
            block_number=block_number,
            print_summary=False,
        )

        # Run ELO scoring
        logger.info("Running ELO scoring...")
        samples = self.convert_to_sample_scores(scoring_data, environments)
        matches = self.process_elo_matches(samples)
        elo_weights = self.calculate_elo_weights(scoring_data)

        # Compare results
        correlation, rmse, overlap = self.compare_weights(
            absolute_result.final_weights,
            elo_weights,
        )

        # Build report
        report = ValidationReport(
            timestamp=int(time.time()),
            block_number=block_number,
            absolute_weights=absolute_result.final_weights,
            absolute_top_miners=sorted(
                absolute_result.final_weights.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:32],
            elo_weights=elo_weights,
            elo_top_miners=sorted(
                elo_weights.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:32],
            elo_ratings={
                k: float(v.rating)
                for k, v in self.elo_state.ratings.items()
            },
            rank_correlation=correlation,
            weight_rmse=rmse,
            top_k_overlap=overlap,
            matches_generated=len(matches),
            total_elo_updates=len(self.elo_state.ratings),
        )

        self.reports.append(report)

        # Log summary
        logger.info("=" * 80)
        logger.info("VALIDATION ROUND COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Block: {block_number}")
        logger.info(f"Miners scored: {len(scoring_data)}")
        logger.info(f"Pairwise matches: {len(matches)}")
        logger.info(f"ELO ratings tracked: {len(self.elo_state.ratings)}")
        logger.info("")
        logger.info("COMPARISON METRICS:")
        logger.info(f"  Rank correlation: {correlation:.4f}")
        logger.info(f"  Weight RMSE: {rmse:.6f}")
        logger.info(f"  Top-32 overlap: {overlap}/32")
        logger.info("")
        logger.info("TOP 5 ABSOLUTE:")
        for uid, weight in report.absolute_top_miners[:5]:
            logger.info(f"  UID {uid:3d}: {weight:.6f}")
        logger.info("")
        logger.info("TOP 5 ELO:")
        for uid, weight in report.elo_top_miners[:5]:
            logger.info(f"  UID {uid:3d}: {weight:.6f}")

        # Save report
        if self.save_reports:
            await self._save_report(report)

        return report

    async def _save_report(self, report: ValidationReport):
        """Save a report to disk."""
        import os

        os.makedirs(self.report_dir, exist_ok=True)

        filename = f"{self.report_dir}/shadow_report_{report.block_number}_{report.timestamp}.json"

        with open(filename, "w") as f:
            json.dump(report.to_dict(), f, indent=2)

        logger.info(f"Report saved: {filename}")

    async def run(self, rounds: int = 10, interval_seconds: int = 300):
        """
        Run multiple validation rounds.

        Args:
            rounds: Number of validation rounds to run
            interval_seconds: Seconds between rounds
        """
        await self.initialize()

        logger.info(f"Starting shadow validation: {rounds} rounds, {interval_seconds}s interval")

        for i in range(rounds):
            logger.info(f"\n{'#' * 80}")
            logger.info(f"# ROUND {i + 1}/{rounds}")
            logger.info(f"{'#' * 80}")

            try:
                report = await self.run_validation_round()

                if i < rounds - 1:
                    logger.info(f"Waiting {interval_seconds}s before next round...")
                    await asyncio.sleep(interval_seconds)

            except Exception as e:
                logger.error(f"Error in validation round: {e}", exc_info=True)
                if i < rounds - 1:
                    await asyncio.sleep(interval_seconds)

        # Final summary
        self._print_final_summary()

    def _print_final_summary(self):
        """Print summary across all validation rounds."""
        if not self.reports:
            return

        logger.info("\n" + "=" * 80)
        logger.info("FINAL SHADOW VALIDATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total rounds: {len(self.reports)}")

        correlations = [r.rank_correlation for r in self.reports]
        rmses = [r.weight_rmse for r in self.reports]
        overlaps = [r.top_k_overlap for r in self.reports]

        logger.info(f"\nRank Correlation:")
        logger.info(f"  Mean: {sum(correlations)/len(correlations):.4f}")
        logger.info(f"  Min:  {min(correlations):.4f}")
        logger.info(f"  Max:  {max(correlations):.4f}")

        logger.info(f"\nWeight RMSE:")
        logger.info(f"  Mean: {sum(rmses)/len(rmses):.6f}")
        logger.info(f"  Min:  {min(rmses):.6f}")
        logger.info(f"  Max:  {max(rmses):.6f}")

        logger.info(f"\nTop-32 Overlap:")
        logger.info(f"  Mean: {sum(overlaps)/len(overlaps):.1f}")
        logger.info(f"  Min:  {min(overlaps)}")
        logger.info(f"  Max:  {max(overlaps)}")

        logger.info(f"\nTotal ELO ratings tracked: {len(self.elo_state.ratings)}")
        logger.info(f"Total matches processed: {len(self.elo_state.matches)}")


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Shadow Validator for ELO Integration Testing")
    parser.add_argument("--netuid", type=int, default=120, help="Network UID")
    parser.add_argument("--rounds", type=int, default=10, help="Number of validation rounds")
    parser.add_argument("--interval", type=int, default=300, help="Seconds between rounds")
    parser.add_argument("--report-dir", type=str, default="shadow_reports", help="Report output directory")
    parser.add_argument("--no-save", action="store_true", help="Don't save reports to disk")

    args = parser.parse_args()

    validator = ShadowValidator(
        netuid=args.netuid,
        dry_run=True,  # Always dry-run for safety
        save_reports=not args.no_save,
        report_dir=args.report_dir,
    )

    await validator.run(rounds=args.rounds, interval_seconds=args.interval)


if __name__ == "__main__":
    asyncio.run(main())
