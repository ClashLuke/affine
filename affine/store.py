from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


def now() -> int:
    return int(time.time())


def artifact_id(model: str, pinned_revision: str) -> str:
    return hashlib.sha256(f"{model}\0{pinned_revision}".encode()).hexdigest()[:24]


@dataclass(frozen=True)
class Champion:
    artifact_id: str
    model: str
    revision: str
    uid: int | None
    hotkey: str | None
    reign_start: int
    payable: bool


@dataclass(frozen=True)
class BackupRecord:
    artifact_id: str
    model: str
    revision: str
    manifest_key: str
    status: str


@dataclass(frozen=True)
class DuelRecord:
    id: int
    champion_artifact_id: str
    challenger_artifact_id: str
    challenger_uid: int
    challenger_hotkey: str
    challenger_model: str
    challenger_revision: str
    validator_hotkey: str
    schedule_seed: str
    alpha: float
    delta_dethrone: float
    delta_hold: float
    pi_json: str
    versions_hash: str
    status: str


@dataclass(frozen=True)
class Sample:
    duel_id: int
    iter_idx: int
    env_id: str
    task_id: int
    seed: int
    champ_correct: int
    chal_correct: int
    champ_latency_s: float
    chal_latency_s: float
    champ_tokens: int
    chal_tokens: int
    observed_at: int


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self._needs_clean_cut():
            legacy = self.path.with_name(f"{self.path.name}.legacy.{now()}")
            self.path.replace(legacy)
            for suffix in ("-wal", "-shm"):
                sidecar = self.path.with_name(f"{self.path.name}{suffix}")
                if sidecar.exists():
                    sidecar.replace(legacy.with_name(f"{legacy.name}{suffix}"))
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def close(self) -> None:
        self.db.close()

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield self.db
        except BaseException:
            self.db.rollback()
            raise
        else:
            self.db.commit()

    def _needs_clean_cut(self) -> bool:
        if not self.path.exists():
            return False
        try:
            db = sqlite3.connect(self.path)
            rows = db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            tables = {r[0] for r in rows}
            if "cell_observations" in tables or "env_state" in tables:
                return True
            if "duels" in tables:
                cols = {r[1] for r in db.execute("PRAGMA table_info(duels)").fetchall()}
                required = {"delta_dethrone", "delta_hold", "versions_hash", "validator_hotkey"}
                return (
                    not required.issubset(cols)
                    or "log_capital_at_zero" in cols
                    or "attempted_artifacts" in tables
                    or "publications" in tables
                )
            return False
        except sqlite3.DatabaseError:
            return False
        finally:
            try:
                db.close()
            except Exception:
                pass

    def _init_schema(self) -> None:
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS champion (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                artifact_id TEXT NOT NULL,
                model TEXT NOT NULL,
                revision TEXT NOT NULL,
                uid INTEGER,
                hotkey TEXT,
                reign_start INTEGER NOT NULL,
                payable INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS duels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                champion_artifact_id TEXT NOT NULL,
                challenger_artifact_id TEXT NOT NULL,
                challenger_uid INTEGER NOT NULL,
                challenger_hotkey TEXT NOT NULL,
                challenger_model TEXT NOT NULL,
                challenger_revision TEXT NOT NULL,
                validator_hotkey TEXT NOT NULL,
                schedule_seed TEXT NOT NULL,
                alpha REAL NOT NULL,
                delta_dethrone REAL NOT NULL,
                delta_hold REAL NOT NULL,
                pi_json TEXT NOT NULL,
                versions_hash TEXT NOT NULL,
                status TEXT NOT NULL CHECK (
                    status IN ('running', 'dethrone', 'hold', 'inconclusive',
                               'challenger_slot_dead', 'cancelled')
                ),
                rounds_collected INTEGER,
                delta_hat REAL,
                ci_low REAL,
                ci_hi REAL,
                started_block INTEGER NOT NULL,
                finished_block INTEGER,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS samples (
                duel_id INTEGER NOT NULL,
                iter_idx INTEGER NOT NULL,
                env_id TEXT NOT NULL,
                task_id INTEGER NOT NULL,
                seed INTEGER NOT NULL,
                champ_correct INTEGER NOT NULL CHECK (champ_correct IN (0,1)),
                chal_correct INTEGER NOT NULL CHECK (chal_correct IN (0,1)),
                champ_latency_s REAL NOT NULL,
                chal_latency_s REAL NOT NULL,
                champ_tokens INTEGER NOT NULL,
                chal_tokens INTEGER NOT NULL,
                observed_at INTEGER NOT NULL,
                PRIMARY KEY (duel_id, iter_idx, env_id),
                FOREIGN KEY (duel_id) REFERENCES duels(id)
            );
            CREATE INDEX IF NOT EXISTS samples_duel_env ON samples (duel_id, env_id);
            CREATE TABLE IF NOT EXISTS backups (
                artifact_id TEXT NOT NULL,
                model TEXT NOT NULL,
                revision TEXT NOT NULL,
                manifest_key TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                verified_at INTEGER
            );
            """
        )
        self.db.execute("DELETE FROM backups WHERE status='staging'")
        self.db.commit()

    # -- champion / backup --

    def champion(self) -> Champion | None:
        row = self.db.execute("SELECT * FROM champion WHERE id = 1").fetchone()
        return _champion(row) if row else None

    def set_champion(self, champ: Champion, backup: BackupRecord | None = None) -> None:
        ts = now()
        with self.tx() as db:
            if backup is not None:
                db.execute(
                    """
                    INSERT INTO backups(artifact_id, model, revision, manifest_key,
                                        status, created_at, verified_at)
                    VALUES (?, ?, ?, ?, 'current', ?, ?)
                    ON CONFLICT(manifest_key) DO UPDATE SET status='current', verified_at=excluded.verified_at
                    """,
                    (backup.artifact_id, backup.model, backup.revision, backup.manifest_key, ts, ts),
                )
            db.execute(
                "UPDATE backups SET status='retiring' WHERE status='current' AND artifact_id <> ?",
                (champ.artifact_id,),
            )
            db.execute(
                """
                INSERT INTO champion(id, artifact_id, model, revision, uid, hotkey,
                                     reign_start, payable, updated_at)
                VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    artifact_id=excluded.artifact_id,
                    model=excluded.model,
                    revision=excluded.revision,
                    uid=excluded.uid,
                    hotkey=excluded.hotkey,
                    reign_start=excluded.reign_start,
                    payable=excluded.payable,
                    updated_at=excluded.updated_at
                """,
                (
                    champ.artifact_id,
                    champ.model,
                    champ.revision,
                    champ.uid,
                    champ.hotkey,
                    champ.reign_start,
                    int(champ.payable),
                    ts,
                ),
            )

    def update_backup_manifest(self, artifact_id: str, manifest_key: str, model: str, revision: str) -> None:
        ts = now()
        with self.tx() as db:
            db.execute(
                """
                INSERT INTO backups(artifact_id, model, revision, manifest_key,
                                    status, created_at, verified_at)
                VALUES (?, ?, ?, ?, 'current', ?, ?)
                ON CONFLICT(manifest_key) DO UPDATE SET
                    status='current',
                    verified_at=excluded.verified_at
                """,
                (artifact_id, model, revision, manifest_key, ts, ts),
            )
            db.execute("DELETE FROM backups WHERE artifact_id=? AND manifest_key<>?", (artifact_id, manifest_key))
            db.execute("UPDATE backups SET status='retiring' WHERE status='current' AND artifact_id<>?", (artifact_id,))

    def demote_champion(self, artifact: str) -> bool:
        with self.tx() as db:
            cur = db.execute(
                """
                UPDATE champion
                SET uid=NULL, hotkey=NULL, payable=0, updated_at=?
                WHERE id=1 AND artifact_id=?
                """,
                (now(), artifact),
            )
            return cur.rowcount > 0

    def latest_backup_for(self, artifact_id: str) -> BackupRecord | None:
        row = self.db.execute(
            """
            SELECT * FROM backups
            WHERE artifact_id = ?
            ORDER BY
                CASE status WHEN 'current' THEN 0 WHEN 'retiring' THEN 1 ELSE 2 END,
                COALESCE(verified_at, created_at) DESC
            LIMIT 1
            """,
            (artifact_id,),
        ).fetchone()
        return _backup(row) if row else None

    def retiring_backups(self) -> list[BackupRecord]:
        rows = self.db.execute("SELECT * FROM backups WHERE status = 'retiring'").fetchall()
        return [_backup(r) for r in rows]

    def mark_backup_deleted(self, manifest_key: str) -> None:
        with self.tx() as db:
            db.execute("DELETE FROM backups WHERE manifest_key = ?", (manifest_key,))

    # -- duels / samples --

    def create_duel(
        self,
        *,
        champion: Champion,
        challenger_uid: int,
        challenger_hotkey: str,
        challenger_model: str,
        challenger_revision: str,
        validator_hotkey: str,
        schedule_seed: str,
        alpha: float,
        delta_dethrone: float,
        delta_hold: float,
        pi: dict[str, float],
        versions_hash: str,
        started_block: int,
    ) -> DuelRecord:
        ts = now()
        challenger_art = artifact_id(challenger_model, challenger_revision)
        pi_json = json.dumps(pi, sort_keys=True, separators=(",", ":"))
        with self.tx() as db:
            cur = db.execute(
                """
                INSERT INTO duels(champion_artifact_id, challenger_artifact_id, challenger_uid,
                                  challenger_hotkey, challenger_model, challenger_revision,
                                  validator_hotkey, schedule_seed, alpha, delta_dethrone, delta_hold, pi_json,
                                  versions_hash, status, started_block, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'running', ?, ?, ?)
                """,
                (
                    champion.artifact_id,
                    challenger_art,
                    challenger_uid,
                    challenger_hotkey,
                    challenger_model,
                    challenger_revision,
                    validator_hotkey,
                    schedule_seed,
                    alpha,
                    delta_dethrone,
                    delta_hold,
                    pi_json,
                    versions_hash,
                    started_block,
                    ts,
                    ts,
                ),
            )
            duel_id = int(cur.lastrowid)
        return DuelRecord(
            duel_id,
            champion.artifact_id,
            challenger_art,
            challenger_uid,
            challenger_hotkey,
            challenger_model,
            challenger_revision,
            validator_hotkey,
            schedule_seed,
            alpha,
            delta_dethrone,
            delta_hold,
            pi_json,
            versions_hash,
            "running",
        )

    def finish_duel(
        self,
        duel_id: int,
        status: str,
        rounds_collected: int,
        delta_hat: float | None,
        ci_low: float | None,
        ci_hi: float | None,
        finished_block: int | None,
    ) -> None:
        with self.tx() as db:
            db.execute(
                """
                UPDATE duels
                SET status = ?, rounds_collected = ?, delta_hat = ?, ci_low = ?, ci_hi = ?,
                    finished_block = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    status,
                    rounds_collected,
                    delta_hat,
                    ci_low,
                    ci_hi,
                    finished_block,
                    now(),
                    duel_id,
                ),
            )

    def abort_running_duels(self) -> None:
        with self.tx() as db:
            db.execute("UPDATE duels SET status='cancelled', updated_at=? WHERE status='running'", (now(),))

    def record_sample(
        self,
        duel_id: int,
        iter_idx: int,
        env_id: str,
        task_id: int,
        seed: int,
        champ_correct: int,
        chal_correct: int,
        champ_latency_s: float,
        chal_latency_s: float,
        champ_tokens: int,
        chal_tokens: int,
    ) -> None:
        with self.tx() as db:
            db.execute(
                """
                INSERT INTO samples(
                    duel_id, iter_idx, env_id, task_id, seed, champ_correct, chal_correct,
                    champ_latency_s, chal_latency_s, champ_tokens, chal_tokens, observed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    duel_id,
                    iter_idx,
                    env_id,
                    task_id,
                    seed,
                    int(champ_correct),
                    int(chal_correct),
                    float(champ_latency_s),
                    float(chal_latency_s),
                    int(champ_tokens),
                    int(chal_tokens),
                    now(),
                ),
            )

    def samples_for_duel(self, duel_id: int) -> list[Sample]:
        rows = self.db.execute(
            "SELECT * FROM samples WHERE duel_id = ? ORDER BY iter_idx, env_id", (duel_id,)
        ).fetchall()
        return [_sample(r) for r in rows]

    def duel(self, duel_id: int) -> sqlite3.Row | None:
        return self.db.execute("SELECT * FROM duels WHERE id = ?", (duel_id,)).fetchone()

    def attempted_artifact_ids(self, versions_hash: str) -> set[str]:
        rows = self.db.execute(
            """
            SELECT challenger_artifact_id AS artifact_id FROM duels
            WHERE versions_hash = ? AND status NOT IN ('running', 'dethrone')
            UNION
            SELECT champion_artifact_id AS artifact_id FROM duels
            WHERE versions_hash = ? AND status = 'dethrone'
            """,
            (versions_hash, versions_hash),
        ).fetchall()
        return {r["artifact_id"] for r in rows}


def _champion(row: sqlite3.Row) -> Champion:
    return Champion(
        artifact_id=row["artifact_id"],
        model=row["model"],
        revision=row["revision"],
        uid=row["uid"],
        hotkey=row["hotkey"],
        reign_start=int(row["reign_start"]),
        payable=bool(row["payable"]),
    )


def _backup(row: sqlite3.Row) -> BackupRecord:
    return BackupRecord(
        artifact_id=row["artifact_id"],
        model=row["model"],
        revision=row["revision"],
        manifest_key=row["manifest_key"],
        status=row["status"],
    )


def _sample(row: sqlite3.Row) -> Sample:
    return Sample(
        duel_id=int(row["duel_id"]),
        iter_idx=int(row["iter_idx"]),
        env_id=str(row["env_id"]),
        task_id=int(row["task_id"]),
        seed=int(row["seed"]),
        champ_correct=int(row["champ_correct"]),
        chal_correct=int(row["chal_correct"]),
        champ_latency_s=float(row["champ_latency_s"]),
        chal_latency_s=float(row["chal_latency_s"]),
        champ_tokens=int(row["champ_tokens"]),
        chal_tokens=int(row["chal_tokens"]),
        observed_at=int(row["observed_at"]),
    )
