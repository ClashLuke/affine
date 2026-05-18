import asyncio
from dataclasses import dataclass

import affine.duel as duel_mod
from affine.chain import Miner
from affine.config import Config, EnvSpec
from affine.decide import DuelOutcome
from affine.duel import run_duel, task_keys
from affine.loop import static_chain
from affine.store import Champion, Store, artifact_id


@dataclass
class FakeSlot:
    model: str
    revision: str = "rev"


@dataclass
class FakeWrapper:
    env_id: str


def _champion() -> Champion:
    return Champion(
        artifact_id=artifact_id("champ", "rev"),
        model="champ",
        revision="rev",
        uid=1,
        hotkey="hk-champ",
        reign_start=0,
        payable=True,
    )


def test_task_keys_are_paired_and_do_not_use_miner_uid():
    a = task_keys("hotkey", "seed", "env", 7)
    b = task_keys("hotkey", "seed", "env", 7)
    c = task_keys("hotkey", "seed", "env", 8)
    assert a == b
    assert a != c
    assert 0 <= a[0] < (1 << 63)
    assert 0 <= a[1] < (1 << 31)


def test_task_keys_include_validator_hotkey():
    a = task_keys("hotkey-a", "seed", "env", 7)
    b = task_keys("hotkey-b", "seed", "env", 7)
    assert a != b


def test_store_persists_validator_hotkey(tmp_path):
    cfg = Config(db_path=str(tmp_path / "affine.sqlite3"), environments=(EnvSpec("a", "unused", {"timeout": 1}),))
    store = Store(cfg.db_path)
    try:
        champ = _champion()
        store.set_champion(champ)
        duel = store.create_duel(
            champion=champ,
            challenger_uid=2,
            challenger_hotkey="hk-chal",
            challenger_model="chal",
            challenger_revision="rev",
            validator_hotkey="validator",
            schedule_seed="abc",
            alpha=0.1,
            delta_dethrone=0.0,
            delta_hold=0.0,
            pi={"a": 1.0},
            versions_hash="vh",
            started_block=0,
        )
        row = store.duel(duel.id)
        assert row["validator_hotkey"] == "validator"
    finally:
        store.close()


async def test_run_duel_records_paired_samples_and_dethrones(tmp_path, monkeypatch):
    async def fake_run_one(wrapper, params, timeout, slot, seed, task_id):
        return (slot.model == "chal"), 0.01, 1

    monkeypatch.setattr(duel_mod, "run_one", fake_run_one)
    cfg = Config(
        db_path=str(tmp_path / "affine.sqlite3"),
        alpha=0.1,
        delta_dethrone=0.0,
        delta_hold=0.0,
        rounds_max=30,
        slot_dead_run=3,
        environments=(
            EnvSpec("a", "unused", {"timeout": 1}),
            EnvSpec("b", "unused", {"timeout": 1}),
            EnvSpec("c", "unused", {"timeout": 1}),
        ),
    )
    envs = {spec.name: (FakeWrapper(spec.name), spec) for spec in cfg.environments}
    pi = {name: 1 / 3 for name in envs}
    store = Store(cfg.db_path)
    try:
        champ = _champion()
        store.set_champion(champ)
        verdict = await run_duel(
            store,
            static_chain([]),
            cfg,
            envs,
            pi,
            "vh",
            champ,
            FakeSlot("champ"),
            Miner(uid=2, hotkey="hk-chal", model="chal", revision="rev", block=1),
            FakeSlot("chal"),
            asyncio.Event(),
        )
        assert verdict.outcome is DuelOutcome.DETHRONE
        assert verdict.reason == "dethrone"
        samples = store.samples_for_duel(1)
        assert {s.env_id for s in samples[:3]} == {"a", "b", "c"}
        assert all(s.champ_correct == 0 and s.chal_correct == 1 for s in samples)
    finally:
        store.close()


async def test_run_duel_aborts_slot_dead_without_samples(tmp_path, monkeypatch):
    async def fake_run_one(wrapper, params, timeout, slot, seed, task_id):
        if slot.model == "chal":
            return None, 0.01, 0
        return True, 0.01, 1

    monkeypatch.setattr(duel_mod, "run_one", fake_run_one)
    cfg = Config(
        db_path=str(tmp_path / "affine.sqlite3"),
        rounds_max=5,
        slot_dead_run=2,
        environments=(EnvSpec("a", "unused", {"timeout": 1}),),
    )
    envs = {"a": (FakeWrapper("a"), cfg.environments[0])}
    store = Store(cfg.db_path)
    try:
        champ = _champion()
        store.set_champion(champ)
        verdict = await run_duel(
            store,
            static_chain([]),
            cfg,
            envs,
            {"a": 1.0},
            "vh",
            champ,
            FakeSlot("champ"),
            Miner(uid=2, hotkey="hk-chal", model="chal", revision="rev", block=1),
            FakeSlot("chal"),
            asyncio.Event(),
        )
        assert verdict.outcome is DuelOutcome.NO_DETHRONE
        assert verdict.reason == "challenger_slot_dead"
        assert store.samples_for_duel(1) == []
    finally:
        store.close()
