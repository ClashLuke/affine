from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "affine.analyze", *args],
        capture_output=True, text=True, check=False,
    )


def test_analyze_sqlite_counts_only_delivered_pairs(tmp_path):
    from affine.store import BackupRecord, Champion, PairSample, Store, artifact_id

    db = tmp_path / "affine.sqlite3"
    store = Store(db)
    art = artifact_id("champ", "r")
    backup = BackupRecord(art, "champ", "r", "p", "p/manifest.json", "sha", "verified")
    champ = Champion(art, "champ", "r", 0, "hk0", 0, backup.manifest_key, backup.prefix, True)
    store.set_champion(champ, backup)
    duel = store.create_duel(
        champion=champ,
        challenger_uid=1,
        challenger_hotkey="hk1",
        challenger_model="chal",
        challenger_revision="r",
        schedule_seed="seed",
        pairs_per_env=2,
        min_discordant=1,
        alpha=0.05,
        started_block=0,
    )
    store.add_samples([
        PairSample(duel.id, "E", 1, 0, 0, 0, 1, 0.0, 1.0, 0, 1, 0, 1),
        PairSample(duel.id, "E", 2, 1, 0, 1, 0, 1.0, 1.0, 1, 1, 1, 1),
    ])
    store.close()

    out = _run([str(db)])
    assert out.returncode == 0, out.stderr
    assert "0-1" in out.stdout
    assert "1-0" not in out.stdout


def test_analyze_missing_file(tmp_path):
    out = _run([str(tmp_path / "nope.sqlite3")])
    assert out.returncode == 1
    assert "no evidence" in out.stdout
