from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


def now() -> int:
    return int(time.time())


def artifact_id(model: str, pinned_revision: str) -> str:
    """Content-addressed artifact identity. A champion without this is not a champion."""
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
    schedule_seed: str
    alpha: float
    delta_theta: float
    pi_json: str
    status: str


@dataclass(frozen=True)
class CellObservation:
    """One (miner, env, task_id) Bernoulli outcome.

    Replaces the matched-pair `PairSample`. Each cell is an independent
    observation; the IRT model joins them at fit time without requiring two
    miners to share a task_id.
    """
    observation_id: str
    miner_artifact_id: str
    env_id: str
    env_version: str
    task_id: int
    task_spec_hash: str
    grader_hash: str
    serving_hash: str
    raw_outcome: int
    outcome: int
    gated: int
    latency_s: float
    tokens: int
    observed_at: int
    collection_context: str
    sampler_policy_hash: str


@dataclass(frozen=True)
class Cell:
    """Canonical-view row: (miner, env, task) → outcome with timing.

    The IRT fit consumes these; the underlying observation_id and versioning
    metadata stay in cell_observations for audit.
    """
    miner_artifact_id: str
    env_id: str
    task_id: int
    outcome: int
    latency_s: float
    tokens: int
    observed_at: int


ArtKey = tuple[str, str]


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
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
            CREATE TABLE IF NOT EXISTS backups (
                artifact_id TEXT NOT NULL,
                model TEXT NOT NULL,
                revision TEXT NOT NULL,
                manifest_key TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                verified_at INTEGER
            );
            CREATE TABLE IF NOT EXISTS duels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                champion_artifact_id TEXT NOT NULL,
                challenger_artifact_id TEXT NOT NULL,
                challenger_uid INTEGER NOT NULL,
                challenger_hotkey TEXT NOT NULL,
                challenger_model TEXT NOT NULL,
                challenger_revision TEXT NOT NULL,
                schedule_seed TEXT NOT NULL,
                alpha REAL NOT NULL,
                delta_theta REAL NOT NULL,
                pi_json TEXT NOT NULL,
                evaluation_version TEXT NOT NULL,
                status TEXT NOT NULL,
                cells_collected INTEGER,
                delta_theta_observed REAL,
                se_theta REAL,
                rating_diff_diagnostic REAL,
                decision_statistic TEXT NOT NULL DEFAULT 'theta',
                calibration_snapshot_hash TEXT,
                started_block INTEGER NOT NULL,
                finished_block INTEGER,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS cell_observations (
                observation_id TEXT PRIMARY KEY,
                miner_artifact_id TEXT NOT NULL,
                env_id TEXT NOT NULL,
                env_version TEXT NOT NULL,
                task_id INTEGER NOT NULL,
                task_spec_hash TEXT NOT NULL,
                grader_hash TEXT NOT NULL,
                serving_hash TEXT NOT NULL,
                raw_outcome INTEGER NOT NULL CHECK (raw_outcome IN (0,1)),
                outcome INTEGER NOT NULL CHECK (outcome IN (0,1)),
                gated INTEGER NOT NULL CHECK (gated IN (0,1)),
                latency_s REAL NOT NULL CHECK (latency_s >= 0),
                tokens INTEGER NOT NULL CHECK (tokens >= 0),
                observed_at INTEGER NOT NULL,
                collection_context TEXT NOT NULL,
                sampler_policy_hash TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS cells_artifact_env
                ON cell_observations (miner_artifact_id, env_id, env_version);
            CREATE INDEX IF NOT EXISTS cells_observed_at
                ON cell_observations (observed_at);
            CREATE TABLE IF NOT EXISTS env_state (
                env_id TEXT PRIMARY KEY,
                state TEXT NOT NULL,
                env_version TEXT NOT NULL,
                task_spec_hash TEXT NOT NULL,
                grader_hash TEXT NOT NULL,
                serving_hash TEXT NOT NULL,
                updated_at INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS attempted_artifacts (
                miner_artifact_id TEXT NOT NULL,
                evaluation_version TEXT NOT NULL,
                model TEXT NOT NULL,            -- audit columns; identity is miner_artifact_id
                revision TEXT NOT NULL,
                archive_snapshot_hash TEXT,     -- D15 of eval-target.md (drift replay)
                first_observed_at INTEGER NOT NULL,
                PRIMARY KEY (miner_artifact_id, evaluation_version)
            );
            CREATE TABLE IF NOT EXISTS publications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                artifact_id TEXT NOT NULL,
                action TEXT NOT NULL,
                uid INTEGER,
                hotkey TEXT,
                status TEXT NOT NULL,
                dry_run INTEGER NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            );
            """
        )
        # Legacy schema cleanup: drop columns that don't exist in the new design.
        for table, col in (
            ("champion", "backup_manifest"), ("champion", "backup_prefix"),
            ("backups", "prefix"), ("backups", "manifest_sha256"),
            ("duels", "min_discordant"), ("duels", "pairs_per_env"),
            ("duels", "precision"), ("duels", "p_value"),
            ("duels", "log_e_final"), ("duels", "cs_lower"), ("duels", "cs_upper"),
        ):
            cols = {r["name"] for r in self.db.execute(f"PRAGMA table_info({table})").fetchall()}
            if col in cols:
                self.db.execute(f"ALTER TABLE {table} DROP COLUMN {col}")
        # Migrate old `samples` table into cells if present (rename + emit cells).
        tables = {r["name"] for r in self.db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        if "samples" in tables and "legacy_samples" not in tables:
            self.db.execute("ALTER TABLE samples RENAME TO legacy_samples")
        # Ensure duels has the new columns even on legacy DBs.
        duel_cols = {r["name"] for r in self.db.execute("PRAGMA table_info(duels)").fetchall()}
        if "delta_theta" not in duel_cols:
            self.db.execute("ALTER TABLE duels ADD COLUMN delta_theta REAL NOT NULL DEFAULT 0.4")
        if "pi_json" not in duel_cols:
            self.db.execute("ALTER TABLE duels ADD COLUMN pi_json TEXT NOT NULL DEFAULT '{}'")
        if "evaluation_version" not in duel_cols:
            self.db.execute(
                "ALTER TABLE duels ADD COLUMN evaluation_version TEXT NOT NULL DEFAULT ''"
            )
        if "cells_collected" not in duel_cols:
            self.db.execute("ALTER TABLE duels ADD COLUMN cells_collected INTEGER")
        if "delta_theta_observed" not in duel_cols:
            self.db.execute("ALTER TABLE duels ADD COLUMN delta_theta_observed REAL")
        if "se_theta" not in duel_cols:
            self.db.execute("ALTER TABLE duels ADD COLUMN se_theta REAL")
        if "rating_diff_diagnostic" not in duel_cols:
            self.db.execute("ALTER TABLE duels ADD COLUMN rating_diff_diagnostic REAL")
        if "decision_statistic" not in duel_cols:
            self.db.execute("ALTER TABLE duels ADD COLUMN decision_statistic TEXT NOT NULL DEFAULT 'theta'")
        if "calibration_snapshot_hash" not in duel_cols:
            self.db.execute("ALTER TABLE duels ADD COLUMN calibration_snapshot_hash TEXT")
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
                    (backup.artifact_id, backup.model, backup.revision,
                     backup.manifest_key, ts, ts),
                )
            db.execute(
                "UPDATE backups SET status='retiring' WHERE status='current' AND artifact_id <> ?",
                (champ.artifact_id,),
            )
            db.execute(
                "UPDATE publications SET status='superseded', updated_at=? "
                "WHERE artifact_id<>? AND status IN ('confirmed','dry_run')",
                (ts, champ.artifact_id),
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
                (champ.artifact_id, champ.model, champ.revision, champ.uid, champ.hotkey,
                 champ.reign_start, int(champ.payable), ts),
            )

    def update_backup_manifest(self, artifact_id: str, manifest_key: str,
                               model: str, revision: str) -> None:
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
            db.execute(
                "DELETE FROM backups WHERE artifact_id=? AND manifest_key<>?",
                (artifact_id, manifest_key),
            )
            db.execute(
                "UPDATE backups SET status='retiring' "
                "WHERE status='current' AND artifact_id<>?",
                (artifact_id,),
            )

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

    # -- duel records --

    def create_duel(
        self,
        *,
        champion: Champion,
        challenger_uid: int,
        challenger_hotkey: str,
        challenger_model: str,
        challenger_revision: str,
        schedule_seed: str,
        alpha: float,
        delta_theta: float,
        pi: dict[str, float],
        evaluation_version: str,
        started_block: int,
    ) -> DuelRecord:
        ts = now()
        challenger_art = artifact_id(challenger_model, challenger_revision)
        pi_json = json.dumps(pi, sort_keys=True)
        with self.tx() as db:
            cur = db.execute(
                """
                INSERT INTO duels(champion_artifact_id, challenger_artifact_id, challenger_uid,
                                  challenger_hotkey, challenger_model, challenger_revision,
                                  schedule_seed, alpha, delta_theta, pi_json, evaluation_version,
                                  status, started_block, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'running', ?, ?, ?)
                """,
                (champion.artifact_id, challenger_art, challenger_uid, challenger_hotkey,
                 challenger_model, challenger_revision, schedule_seed, alpha, delta_theta,
                 pi_json, evaluation_version, started_block, ts, ts),
            )
            duel_id = int(cur.lastrowid)
        return DuelRecord(duel_id, champion.artifact_id, challenger_art, challenger_uid,
                          challenger_hotkey, challenger_model, challenger_revision,
                          schedule_seed, alpha, delta_theta, pi_json, "running")

    def finish_duel(
        self,
        duel_id: int,
        status: str,
        cells_collected: int,
        delta_theta_observed: float | None,
        se_theta: float | None,
        rating_diff_diagnostic: float | None,
        calibration_snapshot_hash: str | None,
        finished_block: int | None,
    ) -> None:
        with self.tx() as db:
            db.execute(
                """
                UPDATE duels
                SET status = ?, cells_collected = ?, delta_theta_observed = ?, se_theta = ?,
                    rating_diff_diagnostic = ?, calibration_snapshot_hash = ?,
                    finished_block = ?, updated_at = ?
                WHERE id = ?
                """,
                (status, cells_collected, delta_theta_observed, se_theta,
                 rating_diff_diagnostic, calibration_snapshot_hash,
                 finished_block, now(), duel_id),
            )

    def abort_running_duels(self) -> None:
        with self.tx() as db:
            db.execute("UPDATE duels SET status='aborted_crash', updated_at=? WHERE status='running'", (now(),))

    # -- cell observations --

    def add_observation(self, obs: CellObservation) -> None:
        with self.tx() as db:
            db.execute(
                """
                INSERT INTO cell_observations(
                    observation_id, miner_artifact_id, env_id, env_version, task_id,
                    task_spec_hash, grader_hash, serving_hash,
                    raw_outcome, outcome, gated, latency_s, tokens, observed_at,
                    collection_context, sampler_policy_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (obs.observation_id, obs.miner_artifact_id, obs.env_id, obs.env_version,
                 obs.task_id, obs.task_spec_hash, obs.grader_hash, obs.serving_hash,
                 obs.raw_outcome, obs.outcome, obs.gated, obs.latency_s, obs.tokens,
                 obs.observed_at, obs.collection_context, obs.sampler_policy_hash),
            )

    def cells_view(
        self,
        env_set: set[str],
        env_version: dict[str, str],
        task_spec_hash: dict[str, str],
        grader_hash: dict[str, str],
        serving_hash: dict[str, str],
        *,
        rule: str = "first",
        exclude_artifacts: set[str] | None = None,
    ) -> list[Cell]:
        """Materialize the canonical view as a list of `Cell` rows.

        `rule` is "first" (default; first-observation per (miner, env, task))
        or "latest". Filters by current `(env_version, task_spec_hash,
        grader_hash, serving_hash)` per env *before* aggregation, so old-version
        observations can't shadow current ones (D1).
        """
        if rule not in {"first", "latest"}:
            raise ValueError(f"unknown view rule: {rule!r}")
        if not env_set:
            return []
        order = "ASC" if rule == "first" else "DESC"
        placeholders = ",".join("?" * len(env_set))
        envs = sorted(env_set)
        sql = f"""
            SELECT miner_artifact_id, env_id, task_id, outcome, latency_s, tokens, observed_at
            FROM (
                SELECT *,
                       ROW_NUMBER() OVER (
                           PARTITION BY miner_artifact_id, env_id, task_id
                           ORDER BY observed_at {order}, observation_id {order}
                       ) AS rn
                FROM cell_observations
                WHERE env_id IN ({placeholders})
            )
            WHERE rn = 1
        """
        rows = self.db.execute(sql, envs).fetchall()
        cur_env_version = env_version
        cur_task = task_spec_hash
        cur_grader = grader_hash
        cur_serving = serving_hash
        excl = exclude_artifacts or set()
        out: list[Cell] = []
        # Re-filter by per-env current versioning (the canonical view's full
        # filter is on the underlying observation rows; we re-check here so
        # the view stays consistent even if the SELECT retrieved a stale row
        # for a (miner, env, task) the current version doesn't recognize).
        # In practice cell_observations stores env_version etc. per row, and
        # we want only observations whose versioning matches current. Pull
        # the full row and filter:
        rows = self.db.execute(
            f"""
            SELECT miner_artifact_id, env_id, env_version, task_id,
                   task_spec_hash, grader_hash, serving_hash,
                   outcome, latency_s, tokens, observed_at, observation_id
            FROM (
                SELECT *,
                       ROW_NUMBER() OVER (
                           PARTITION BY miner_artifact_id, env_id, task_id
                           ORDER BY observed_at {order}, observation_id {order}
                       ) AS rn
                FROM cell_observations
                WHERE env_id IN ({placeholders})
            )
            WHERE rn = 1
            """,
            envs,
        ).fetchall()
        for r in rows:
            env = r["env_id"]
            if r["miner_artifact_id"] in excl:
                continue
            if cur_env_version.get(env) != r["env_version"]:
                continue
            if cur_task.get(env) != r["task_spec_hash"]:
                continue
            if cur_grader.get(env) != r["grader_hash"]:
                continue
            if cur_serving.get(env) != r["serving_hash"]:
                continue
            out.append(Cell(
                miner_artifact_id=r["miner_artifact_id"],
                env_id=env,
                task_id=int(r["task_id"]),
                outcome=int(r["outcome"]),
                latency_s=float(r["latency_s"]),
                tokens=int(r["tokens"]),
                observed_at=int(r["observed_at"]),
            ))
        return out

    # -- env lifecycle (D10, simplified for MVP) --

    def env_state(self, env_id: str) -> dict | None:
        row = self.db.execute(
            "SELECT * FROM env_state WHERE env_id = ?", (env_id,)
        ).fetchone()
        if row is None:
            return None
        return {k: row[k] for k in row.keys()}

    def upsert_env_state(
        self,
        env_id: str,
        state: str,
        env_version: str,
        task_spec_hash: str,
        grader_hash: str,
        serving_hash: str,
    ) -> None:
        ts = now()
        with self.tx() as db:
            db.execute(
                """
                INSERT INTO env_state(env_id, state, env_version, task_spec_hash,
                                      grader_hash, serving_hash, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(env_id) DO UPDATE SET
                    state=excluded.state,
                    env_version=excluded.env_version,
                    task_spec_hash=excluded.task_spec_hash,
                    grader_hash=excluded.grader_hash,
                    serving_hash=excluded.serving_hash,
                    updated_at=excluded.updated_at
                """,
                (env_id, state, env_version, task_spec_hash, grader_hash, serving_hash, ts),
            )

    def all_env_states(self) -> dict[str, dict]:
        rows = self.db.execute("SELECT * FROM env_state").fetchall()
        return {r["env_id"]: {k: r[k] for k in r.keys()} for r in rows}

    # -- cummax: persistent attempted set --

    def attempted_artifacts(self, evaluation_version: str) -> set[str]:
        """Return set of `miner_artifact_id`s already attempted under this
        evaluation_version. Identity is the full content hash (D9 of
        eval-target.md), not (model, revision) — different tokenizer /
        adapter / decoding configs are different artifacts."""
        rows = self.db.execute(
            "SELECT miner_artifact_id FROM attempted_artifacts WHERE evaluation_version = ?",
            (evaluation_version,),
        ).fetchall()
        return {r["miner_artifact_id"] for r in rows}

    def mark_attempted(
        self,
        miner_artifact_id: str,
        evaluation_version: str,
        *,
        model: str,
        revision: str,
        archive_snapshot_hash: str | None = None,
    ) -> None:
        with self.tx() as db:
            db.execute(
                """
                INSERT OR IGNORE INTO attempted_artifacts(
                    miner_artifact_id, evaluation_version, model, revision,
                    archive_snapshot_hash, first_observed_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (miner_artifact_id, evaluation_version, model, revision,
                 archive_snapshot_hash, now()),
            )

    # -- publications (unchanged) --

    def publication_intent(self, artifact: str, action: str, uid: int | None,
                           hotkey: str | None, dry_run: bool) -> int:
        ts = now()
        with self.tx() as db:
            row = db.execute(
                """
                SELECT id FROM publications
                WHERE artifact_id=? AND action=?
                  AND uid IS ?
                  AND hotkey IS ?
                  AND dry_run=?
                  AND status IN ('intent','submitted','failed','confirmed','dry_run')
                ORDER BY id DESC LIMIT 1
                """,
                (artifact, action, uid, hotkey, int(dry_run)),
            ).fetchone()
            if row:
                latest = db.execute(
                    """
                    SELECT id FROM publications
                    WHERE artifact_id=? AND dry_run=?
                      AND status IN ('intent','submitted','failed','confirmed','dry_run')
                    ORDER BY id DESC LIMIT 1
                    """,
                    (artifact, int(dry_run)),
                ).fetchone()
                if latest and int(latest["id"]) == int(row["id"]):
                    return int(row["id"])
            cur = db.execute(
                """
                INSERT INTO publications(artifact_id, action, uid, hotkey, status, dry_run,
                                         created_at, updated_at)
                VALUES (?, ?, ?, ?, 'intent', ?, ?, ?)
                """,
                (artifact, action, uid, hotkey, int(dry_run), ts, ts),
            )
            return int(cur.lastrowid)

    def publication_status(self, pub_id: int) -> str | None:
        row = self.db.execute("SELECT status FROM publications WHERE id=?", (pub_id,)).fetchone()
        return str(row["status"]) if row else None

    def mark_publication(self, pub_id: int, status: str) -> None:
        with self.tx() as db:
            db.execute(
                "UPDATE publications SET status=?, attempts=attempts+1, updated_at=? WHERE id=?",
                (status, now(), pub_id),
            )


def make_observation_id() -> str:
    return uuid.uuid4().hex


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
