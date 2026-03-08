import pytest
import sys
import os
import random
import math
from decimal import Decimal
from typing import Dict

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, "affine", "src"))

from elo.calculator import EloCalculator
from elo.config import EloConfig
from elo.models import EloRating, SampleScore
from elo.match_engine import PairwiseMatchGenerator


class TestEndToEndEloScoring:
    @pytest.fixture
    def elo_config(self):
        return EloConfig()

    @pytest.fixture
    def calculator(self, elo_config):
        return EloCalculator(elo_config)

    @pytest.fixture
    def match_generator(self, elo_config):
        return PairwiseMatchGenerator(elo_config)

    def test_single_task_pairwise_comparison(self, calculator, match_generator, elo_config):
        env = "affine:ded-v2"
        samples = [
            SampleScore(miner_hotkey="miner_a", model_revision="rev1", env=env, task_id=12345, score=Decimal("0.95"), timestamp=1700000000000),
            SampleScore(miner_hotkey="miner_b", model_revision="rev1", env=env, task_id=12345, score=Decimal("0.85"), timestamp=1700000000000),
            SampleScore(miner_hotkey="miner_c", model_revision="rev1", env=env, task_id=12345, score=Decimal("0.75"), timestamp=1700000000000),
        ]

        ratings = {}
        for s in samples:
            key = f"{s.miner_hotkey}#{s.model_revision}#{s.env}"
            ratings[key] = EloRating(miner_hotkey=s.miner_hotkey, model_revision=s.model_revision, env=s.env, rating=elo_config.DEFAULT_RATING)

        matches = match_generator.generate_matches_from_samples(samples, ratings)
        assert len(matches) == 3  # C(3,2)

        for match in matches:
            for p in match.participants:
                key = f"{p.miner_hotkey}#{p.model_revision}#{match.env}"
                if p.elo_after is not None:
                    ratings[key].rating = p.elo_after

        sorted_miners = sorted(ratings.items(), key=lambda x: x[1].rating, reverse=True)
        assert sorted_miners[0][0].startswith("miner_a")
        assert sorted_miners[2][0].startswith("miner_c")

    def test_multiple_rounds_convergence(self, calculator, match_generator, elo_config):
        random.seed(42)
        env = "affine:ded-v2"
        ratings = {
            "miner_a#rev1#affine:ded-v2": EloRating(miner_hotkey="miner_a", model_revision="rev1", env=env, rating=elo_config.DEFAULT_RATING),
            "miner_b#rev1#affine:ded-v2": EloRating(miner_hotkey="miner_b", model_revision="rev1", env=env, rating=elo_config.DEFAULT_RATING),
        }

        for round_num in range(50):
            score_a = Decimal(str(0.8 + random.uniform(-0.1, 0.1)))
            score_b = Decimal(str(0.5 + random.uniform(-0.1, 0.1)))
            samples = [
                SampleScore(miner_hotkey="miner_a", model_revision="rev1", env=env, task_id=round_num, score=score_a, timestamp=1700000000000 + round_num * 1000),
                SampleScore(miner_hotkey="miner_b", model_revision="rev1", env=env, task_id=round_num, score=score_b, timestamp=1700000000000 + round_num * 1000),
            ]
            matches = match_generator.generate_matches_from_samples(samples, ratings)
            for match in matches:
                for p in match.participants:
                    key = f"{p.miner_hotkey}#{p.model_revision}#{match.env}"
                    if p.elo_after is not None:
                        ratings[key].rating = p.elo_after
                        ratings[key].matches_played += 1

        rating_a = float(ratings["miner_a#rev1#affine:ded-v2"].rating)
        rating_b = float(ratings["miner_b#rev1#affine:ded-v2"].rating)
        assert rating_a > rating_b
        assert rating_a - rating_b > 100

    def test_multi_environment_scoring(self, match_generator, elo_config):
        envs = ["affine:ded-v2", "affine:tool-v1", "affine:chat-v2"]
        miners = ["miner_a", "miner_b", "miner_c"]

        ratings = {}
        for miner in miners:
            for env in envs:
                ratings[f"{miner}#rev1#{env}"] = EloRating(
                    miner_hotkey=miner, model_revision="rev1", env=env, rating=elo_config.DEFAULT_RATING,
                )

        scores = {"miner_a": "0.9", "miner_b": "0.7", "miner_c": "0.5"}
        for round_num in range(5):
            for env in envs:
                samples = [
                    SampleScore(miner_hotkey=m, model_revision="rev1", env=env, task_id=round_num,
                                score=Decimal(scores[m]), timestamp=1700000000000 + round_num * 1000)
                    for m in miners
                ]
                matches = match_generator.generate_matches_from_samples(samples, ratings)
                for match in matches:
                    for p in match.participants:
                        key = f"{p.miner_hotkey}#{p.model_revision}#{match.env}"
                        if p.elo_after is not None and p.elo_before is not None:
                            ratings[key].rating += p.elo_after - p.elo_before

        for env in envs:
            ra = float(ratings[f"miner_a#rev1#{env}"].rating)
            rb = float(ratings[f"miner_b#rev1#{env}"].rating)
            rc = float(ratings[f"miner_c#rev1#{env}"].rating)
            assert ra > rb > rc

    def test_arithmetic_mean_weight_calculation(self, match_generator, elo_config):
        envs = ["env1", "env2", "env3"]
        ratings = {
            "miner_a#rev1#env1": EloRating(miner_hotkey="miner_a", model_revision="rev1", env="env1", rating=Decimal("1600")),
            "miner_a#rev1#env2": EloRating(miner_hotkey="miner_a", model_revision="rev1", env="env2", rating=Decimal("1400")),
            "miner_a#rev1#env3": EloRating(miner_hotkey="miner_a", model_revision="rev1", env="env3", rating=Decimal("1500")),
            "miner_b#rev1#env1": EloRating(miner_hotkey="miner_b", model_revision="rev1", env="env1", rating=Decimal("1200")),
            "miner_b#rev1#env2": EloRating(miner_hotkey="miner_b", model_revision="rev1", env="env2", rating=Decimal("1300")),
            "miner_b#rev1#env3": EloRating(miner_hotkey="miner_b", model_revision="rev1", env="env3", rating=Decimal("1250")),
        }

        def calc_weight(hotkey):
            env_ratings = [float(ratings[f"{hotkey}#rev1#{e}"].rating) for e in envs]
            return max(0, sum(env_ratings) / len(env_ratings) - 1000)

        weight_a, weight_b = calc_weight("miner_a"), calc_weight("miner_b")
        assert weight_a > weight_b

        total = weight_a + weight_b
        norm_a, norm_b = weight_a / total, weight_b / total
        assert 0 < norm_a < 1
        assert 0 < norm_b < 1
        assert abs(norm_a + norm_b - 1.0) < 0.0001

    def test_phantom_1500_not_included(self, match_generator, elo_config):
        """A miner rated in 1 of 3 envs should get avg_elo from only that env, not diluted."""
        envs = ["env1", "env2", "env3"]
        # miner_a only has a rating in env1 (1800), no matches in env2/env3
        ratings = {
            "miner_a#rev1#env1": EloRating(
                miner_hotkey="miner_a", model_revision="rev1", env="env1",
                rating=Decimal("1800"), matches_played=20,
            ),
        }

        # Compute weight the way production should: only include envs with matches > 0
        env_elos = []
        for env in envs:
            key = f"miner_a#rev1#{env}"
            r = ratings.get(key)
            if r and r.matches_played > 0:
                env_elos.append(float(r.rating))

        assert len(env_elos) == 1  # only env1
        avg_elo = sum(env_elos) / len(env_elos)
        assert avg_elo == 1800.0  # not (1800+1500+1500)/3 = 1600

        raw_weight = max(0, avg_elo - 1000)
        assert raw_weight == 800.0

    def test_draw_handling(self, calculator, match_generator, elo_config):
        env = "affine:ded-v2"
        ratings = {
            "miner_a#rev1#affine:ded-v2": EloRating(miner_hotkey="miner_a", model_revision="rev1", env=env, rating=Decimal("1500")),
            "miner_b#rev1#affine:ded-v2": EloRating(miner_hotkey="miner_b", model_revision="rev1", env=env, rating=Decimal("1500")),
        }

        samples = [
            SampleScore(miner_hotkey="miner_a", model_revision="rev1", env=env, task_id=1, score=Decimal("0.800"), timestamp=1700000000000),
            SampleScore(miner_hotkey="miner_b", model_revision="rev1", env=env, task_id=1, score=Decimal("0.805"), timestamp=1700000000000),
        ]
        matches = match_generator.generate_matches_from_samples(samples, ratings)
        for match in matches:
            for p in match.participants:
                key = f"{p.miner_hotkey}#{p.model_revision}#{match.env}"
                if p.elo_after is not None:
                    ratings[key].rating = p.elo_after

        assert abs(float(ratings["miner_a#rev1#affine:ded-v2"].rating) - 1500) < 1
        assert abs(float(ratings["miner_b#rev1#affine:ded-v2"].rating) - 1500) < 1

    def test_head_to_head_rating_changes(self, calculator):
        new_a, new_b, delta_a, delta_b = calculator.update_ratings_head_to_head(
            Decimal("1500"), Decimal("1500"), 0, 0, "a_wins",
        )
        assert new_a > Decimal("1500")
        assert new_b < Decimal("1500")
        assert abs(delta_a + delta_b) < 0.01

    def test_upset_bonus(self, calculator):
        _, _, delta_a, delta_b = calculator.update_ratings_head_to_head(
            Decimal("1200"), Decimal("1800"), 100, 100, "a_wins",
        )
        assert delta_a > 20
        assert delta_b < -20


class TestShadowValidatorIntegration:
    def test_correlation_calculation(self):
        from scipy import stats
        assert stats.spearmanr([0, 1, 2], [0, 1, 2])[0] == 1.0
        assert stats.spearmanr([0, 1, 2], [2, 1, 0])[0] == -1.0

    def test_weight_rmse_calculation(self):
        absolute = {0: 0.5, 1: 0.3, 2: 0.2}
        elo = {0: 0.45, 1: 0.35, 2: 0.2}
        rmse = math.sqrt(sum((absolute[k] - elo[k]) ** 2 for k in absolute) / len(absolute))
        assert 0.04 < rmse < 0.05

    def test_top_k_overlap_calculation(self):
        absolute = {0: 0.5, 1: 0.3, 2: 0.2, 3: 0.15, 4: 0.1}
        elo = {0: 0.45, 1: 0.35, 2: 0.15, 3: 0.2, 4: 0.05}

        def top_k(weights, k):
            return set(sorted(weights, key=weights.get, reverse=True)[:k])

        assert len(top_k(absolute, 3) & top_k(elo, 3)) == 2


class TestReplayValidatorLogic:
    def test_sample_extraction_logic(self):
        scores_data = [
            {"uid": 1, "miner_hotkey": "5Ghotkey_a...", "model_revision": "rev1",
             "scores_by_env": {"affine:ded-v2": {"score": 0.9}, "affine:tool-v1": {"score": 0.85}}},
            {"uid": 2, "miner_hotkey": "5Ghotkey_b...", "model_revision": "rev1",
             "scores_by_env": {"affine:ded-v2": {"score": 0.7}, "affine:tool-v1": 0.65}},
        ]

        samples = []
        block_number = 1000000
        for record in scores_data:
            for env, env_data in record.get("scores_by_env", {}).items():
                score = env_data.get("score", 0) if isinstance(env_data, dict) else env_data
                if score > 0:
                    samples.append(SampleScore(
                        miner_hotkey=record["miner_hotkey"], model_revision=record["model_revision"],
                        env=env, task_id=block_number, score=Decimal(str(score)),
                        timestamp=block_number * 12000,
                    ))

        assert len(samples) == 4
        assert all(s.score > 0 for s in samples)

    def test_ranking_comparison_logic(self):
        from scipy import stats

        historical = {1: 0.5, 2: 0.3, 3: 0.15, 4: 0.05}
        elo = {1: 0.45, 2: 0.35, 3: 0.12, 4: 0.08}

        hist_ranks = {uid: i for i, (uid, _) in enumerate(sorted(historical.items(), key=lambda x: x[1], reverse=True))}
        elo_ranks = {uid: i for i, (uid, _) in enumerate(sorted(elo.items(), key=lambda x: x[1], reverse=True))}

        common_uids = set(historical.keys()) & set(elo.keys())
        correlation, _ = stats.spearmanr(
            [hist_ranks[u] for u in common_uids],
            [elo_ranks[u] for u in common_uids],
        )
        assert correlation > 0.9

        hist_top = set(list(hist_ranks.keys())[:3])
        elo_top = set(list(elo_ranks.keys())[:3])
        assert len(hist_top & elo_top) >= 2


class TestEloWeightIntegration:
    @pytest.fixture
    def elo_config(self):
        return EloConfig()

    @pytest.fixture
    def match_generator(self, elo_config):
        return PairwiseMatchGenerator(elo_config)

    def test_full_scoring_round_produces_valid_weights(self, elo_config, match_generator):
        miners = {
            "miner_a": {"true_skill": 0.9, "uid": 1},
            "miner_b": {"true_skill": 0.75, "uid": 2},
            "miner_c": {"true_skill": 0.6, "uid": 3},
            "miner_d": {"true_skill": 0.5, "uid": 4},
            "miner_e": {"true_skill": 0.4, "uid": 5},
        }
        envs = ["env1", "env2"]

        ratings = {}
        for miner in miners:
            for env in envs:
                ratings[f"{miner}#rev1#{env}"] = EloRating(
                    miner_hotkey=miner, model_revision="rev1", env=env, rating=elo_config.DEFAULT_RATING,
                )

        random.seed(42)
        for round_num in range(10):
            for env in envs:
                samples = [
                    SampleScore(
                        miner_hotkey=m, model_revision="rev1", env=env, task_id=round_num,
                        score=Decimal(str(max(0.1, min(1.0, data["true_skill"] + random.uniform(-0.1, 0.1))))),
                        timestamp=1700000000000 + round_num * 1000,
                    )
                    for m, data in miners.items()
                ]
                matches = match_generator.generate_matches_from_samples(samples, ratings)
                for match in matches:
                    for p in match.participants:
                        key = f"{p.miner_hotkey}#{p.model_revision}#{match.env}"
                        if p.elo_after is not None:
                            ratings[key].rating = p.elo_after
                            ratings[key].matches_played += 1

        weights: Dict[int, float] = {}
        for miner, data in miners.items():
            env_ratings = [float(ratings[f"{miner}#rev1#{e}"].rating) for e in envs]
            weights[data["uid"]] = max(0, sum(env_ratings) / len(env_ratings) - 1000)

        total = sum(weights.values())
        weights = {uid: w / total for uid, w in weights.items()}

        assert len(weights) == 5
        assert abs(sum(weights.values()) - 1.0) < 0.0001

        sorted_by_weight = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        assert 1 in {sorted_by_weight[0][0], sorted_by_weight[1][0]}  # miner_a in top 2
        assert 5 in {sorted_by_weight[3][0], sorted_by_weight[4][0]}  # miner_e in bottom 2

    def test_weights_change_appropriately_over_time(self, elo_config, match_generator):
        ratings = {
            "miner_a#rev1#env1": EloRating(miner_hotkey="miner_a", model_revision="rev1", env="env1", rating=elo_config.DEFAULT_RATING),
            "miner_b#rev1#env1": EloRating(miner_hotkey="miner_b", model_revision="rev1", env="env1", rating=elo_config.DEFAULT_RATING),
        }

        def run_rounds(start, end, score_a, score_b):
            for rn in range(start, end):
                samples = [
                    SampleScore(miner_hotkey="miner_a", model_revision="rev1", env="env1", task_id=rn, score=Decimal(score_a), timestamp=1700000000000 + rn * 1000),
                    SampleScore(miner_hotkey="miner_b", model_revision="rev1", env="env1", task_id=rn, score=Decimal(score_b), timestamp=1700000000000 + rn * 1000),
                ]
                for match in match_generator.generate_matches_from_samples(samples, ratings):
                    for p in match.participants:
                        key = f"{p.miner_hotkey}#rev1#env1"
                        if p.elo_after:
                            ratings[key].rating = p.elo_after

        run_rounds(0, 10, "0.9", "0.5")
        phase1_a = float(ratings["miner_a#rev1#env1"].rating)
        phase1_b = float(ratings["miner_b#rev1#env1"].rating)
        assert phase1_a > phase1_b

        run_rounds(10, 20, "0.6", "0.9")
        phase2_a = float(ratings["miner_a#rev1#env1"].rating)
        phase2_b = float(ratings["miner_b#rev1#env1"].rating)
        assert phase2_b > phase1_b
        assert phase2_a < phase1_a
