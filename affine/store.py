from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .paired import PairCounts


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
    backup_manifest: str
    backup_prefix: str
    payable: bool


@dataclass(frozen=True)
class BackupRecord:
    artifact_id: str
    model: str
    revision: str
    prefix: str
    manifest_key: str
    manifest_sha256: str
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
    pairs_per_env: int
    min_discordant: int
    alpha: float
    status: str


@dataclass(frozen=True)
class PairSample:
    duel_id: int
    env: str
    task_id: int
    iter_idx: int
    block: int
    champion_pass: int
    challenger_pass: int
    champion_latency: float
    challenger_latency: float
    champion_delivered: int
    challenger_delivered: int
    champion_tokens: int
    challenger_tokens: int


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
                backup_manifest TEXT NOT NULL,
                backup_prefix TEXT NOT NULL,
                payable INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS pending_champion (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                artifact_id TEXT NOT NULL,
                model TEXT NOT NULL,
                revision TEXT NOT NULL,
                uid INTEGER,
                hotkey TEXT,
                reign_start INTEGER NOT NULL,
                backup_manifest TEXT NOT NULL,
                backup_prefix TEXT NOT NULL,
                payable INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS backups (
                artifact_id TEXT NOT NULL,
                model TEXT NOT NULL,
                revision TEXT NOT NULL,
                prefix TEXT NOT NULL,
                manifest_key TEXT PRIMARY KEY,
                manifest_sha256 TEXT NOT NULL,
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
                pairs_per_env INTEGER NOT NULL,
                min_discordant INTEGER NOT NULL,
                alpha REAL NOT NULL,
                status TEXT NOT NULL,
                counts_json TEXT,
                p_value REAL,
                started_block INTEGER NOT NULL,
                finished_block INTEGER,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                duel_id INTEGER NOT NULL REFERENCES duels(id) ON DELETE CASCADE,
                env TEXT NOT NULL,
                task_id INTEGER NOT NULL,
                iter_idx INTEGER NOT NULL,
                block INTEGER NOT NULL,
                champion_pass INTEGER NOT NULL,
                challenger_pass INTEGER NOT NULL,
                champion_latency REAL NOT NULL,
                challenger_latency REAL NOT NULL,
                champion_delivered INTEGER NOT NULL,
                challenger_delivered INTEGER NOT NULL,
                champion_tokens INTEGER NOT NULL,
                challenger_tokens INTEGER NOT NULL,
                UNIQUE(duel_id, env, iter_idx)
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

    def champion(self) -> Champion | None:
        row = self.db.execute("SELECT * FROM champion WHERE id = 1").fetchone()
        return _champion(row) if row else None

    def pending_champion(self) -> Champion | None:
        row = self.db.execute("SELECT * FROM pending_champion WHERE id = 1").fetchone()
        return _champion(row) if row else None

    def set_champion(self, champ: Champion, backup: BackupRecord) -> None:
        ts = now()
        with self.tx() as db:
            db.execute(
                """
                INSERT INTO backups(artifact_id, model, revision, prefix, manifest_key,
                                    manifest_sha256, status, created_at, verified_at)
                VALUES (?, ?, ?, ?, ?, ?, 'current', ?, ?)
                ON CONFLICT(manifest_key) DO UPDATE SET status='current', verified_at=excluded.verified_at
                """,
                (backup.artifact_id, backup.model, backup.revision, backup.prefix,
                 backup.manifest_key, backup.manifest_sha256, ts, ts),
            )
            db.execute("UPDATE backups SET status='retiring' WHERE status='current' AND manifest_key <> ?",
                       (backup.manifest_key,))
            db.execute(
                """
                INSERT INTO champion(id, artifact_id, model, revision, uid, hotkey,
                                     reign_start, backup_manifest, backup_prefix, payable, updated_at)
                VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    artifact_id=excluded.artifact_id,
                    model=excluded.model,
                    revision=excluded.revision,
                    uid=excluded.uid,
                    hotkey=excluded.hotkey,
                    reign_start=excluded.reign_start,
                    backup_manifest=excluded.backup_manifest,
                    backup_prefix=excluded.backup_prefix,
                    payable=excluded.payable,
                    updated_at=excluded.updated_at
                """,
                (champ.artifact_id, champ.model, champ.revision, champ.uid, champ.hotkey,
                 champ.reign_start, champ.backup_manifest, champ.backup_prefix,
                 int(champ.payable), ts),
            )

    def set_pending_champion(self, champ: Champion, backup: BackupRecord) -> None:
        ts = now()
        with self.tx() as db:
            db.execute(
                """
                INSERT INTO backups(artifact_id, model, revision, prefix, manifest_key,
                                    manifest_sha256, status, created_at, verified_at)
                VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?)
                ON CONFLICT(manifest_key) DO UPDATE SET status='pending', verified_at=excluded.verified_at
                """,
                (backup.artifact_id, backup.model, backup.revision, backup.prefix,
                 backup.manifest_key, backup.manifest_sha256, ts, ts),
            )
            db.execute(
                """
                INSERT INTO pending_champion(id, artifact_id, model, revision, uid, hotkey,
                                             reign_start, backup_manifest, backup_prefix, payable, updated_at)
                VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    artifact_id=excluded.artifact_id,
                    model=excluded.model,
                    revision=excluded.revision,
                    uid=excluded.uid,
                    hotkey=excluded.hotkey,
                    reign_start=excluded.reign_start,
                    backup_manifest=excluded.backup_manifest,
                    backup_prefix=excluded.backup_prefix,
                    payable=excluded.payable,
                    updated_at=excluded.updated_at
                """,
                (champ.artifact_id, champ.model, champ.revision, champ.uid, champ.hotkey,
                 champ.reign_start, champ.backup_manifest, champ.backup_prefix,
                 int(champ.payable), ts),
            )

    def finalize_pending_champion(self) -> Champion | None:
        row = self.db.execute("SELECT * FROM pending_champion WHERE id = 1").fetchone()
        if row is None:
            return None
        champ = _champion(row)
        ts = now()
        with self.tx() as db:
            db.execute("UPDATE backups SET status='retiring' WHERE status='current' AND manifest_key <> ?",
                       (champ.backup_manifest,))
            db.execute("UPDATE backups SET status='current', verified_at=? WHERE manifest_key=?",
                       (ts, champ.backup_manifest))
            db.execute(
                """
                INSERT INTO champion(id, artifact_id, model, revision, uid, hotkey,
                                     reign_start, backup_manifest, backup_prefix, payable, updated_at)
                VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    artifact_id=excluded.artifact_id,
                    model=excluded.model,
                    revision=excluded.revision,
                    uid=excluded.uid,
                    hotkey=excluded.hotkey,
                    reign_start=excluded.reign_start,
                    backup_manifest=excluded.backup_manifest,
                    backup_prefix=excluded.backup_prefix,
                    payable=excluded.payable,
                    updated_at=excluded.updated_at
                """,
                (champ.artifact_id, champ.model, champ.revision, champ.uid, champ.hotkey,
                 champ.reign_start, champ.backup_manifest, champ.backup_prefix,
                 int(champ.payable), ts),
            )
            db.execute("DELETE FROM pending_champion WHERE id=1")
        return champ

    def clear_pending_champion(self) -> BackupRecord | None:
        row = self.db.execute(
            """
            SELECT b.*
            FROM pending_champion p
            JOIN backups b ON b.manifest_key = p.backup_manifest
            WHERE p.id = 1
            """
        ).fetchone()
        backup = _backup(row) if row else None
        with self.tx() as db:
            db.execute(
                """
                UPDATE backups
                SET status='staging'
                WHERE manifest_key=(SELECT backup_manifest FROM pending_champion WHERE id=1)
                  AND status='pending'
                """
            )
            db.execute("DELETE FROM pending_champion WHERE id=1")
        return backup

    def record_backup(self, backup: BackupRecord, status: str) -> None:
        ts = now()
        with self.tx() as db:
            db.execute(
                """
                INSERT INTO backups(artifact_id, model, revision, prefix, manifest_key,
                                    manifest_sha256, status, created_at, verified_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(manifest_key) DO UPDATE SET
                    status=excluded.status,
                    verified_at=excluded.verified_at
                """,
                (backup.artifact_id, backup.model, backup.revision, backup.prefix,
                 backup.manifest_key, backup.manifest_sha256, status, ts, ts),
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

    def retiring_backups(self) -> list[BackupRecord]:
        rows = self.db.execute("SELECT * FROM backups WHERE status = 'retiring'").fetchall()
        return [_backup(r) for r in rows]

    def staging_backups(self) -> list[BackupRecord]:
        rows = self.db.execute("SELECT * FROM backups WHERE status = 'staging'").fetchall()
        return [_backup(r) for r in rows]

    def mark_backup_deleted(self, manifest_key: str) -> None:
        with self.tx() as db:
            db.execute("DELETE FROM backups WHERE manifest_key = ?", (manifest_key,))

    def create_duel(
        self,
        *,
        champion: Champion,
        challenger_uid: int,
        challenger_hotkey: str,
        challenger_model: str,
        challenger_revision: str,
        schedule_seed: str,
        pairs_per_env: int,
        min_discordant: int,
        alpha: float,
        started_block: int,
    ) -> DuelRecord:
        ts = now()
        challenger_art = artifact_id(challenger_model, challenger_revision)
        with self.tx() as db:
            cur = db.execute(
                """
                INSERT INTO duels(champion_artifact_id, challenger_artifact_id, challenger_uid,
                                  challenger_hotkey, challenger_model, challenger_revision,
                                  schedule_seed, pairs_per_env, min_discordant, alpha,
                                  status, started_block, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'running', ?, ?, ?)
                """,
                (champion.artifact_id, challenger_art, challenger_uid, challenger_hotkey,
                 challenger_model, challenger_revision, schedule_seed, pairs_per_env,
                 min_discordant, alpha, started_block, ts, ts),
            )
            duel_id = int(cur.lastrowid)
        return DuelRecord(duel_id, champion.artifact_id, challenger_art, challenger_uid,
                          challenger_hotkey, challenger_model, challenger_revision,
                          schedule_seed, pairs_per_env, min_discordant, alpha, "running")

    def add_samples(self, samples: list[PairSample]) -> None:
        if not samples:
            return
        ts = now()
        with self.tx() as db:
            db.executemany(
                """
                INSERT INTO samples(duel_id, env, task_id, iter_idx, block,
                    champion_pass, challenger_pass, champion_latency, challenger_latency,
                    champion_delivered, challenger_delivered, champion_tokens, challenger_tokens)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [(s.duel_id, s.env, s.task_id, s.iter_idx, s.block, s.champion_pass,
                  s.challenger_pass, s.champion_latency, s.challenger_latency,
                  s.champion_delivered, s.challenger_delivered, s.champion_tokens,
                  s.challenger_tokens) for s in samples],
            )
            db.execute("UPDATE duels SET updated_at = ? WHERE id = ?", (ts, samples[0].duel_id))

    def counts(self, duel_id: int) -> PairCounts:
        rows = self.db.execute(
            """
            SELECT champion_pass, challenger_pass
            FROM samples
            WHERE duel_id = ? AND champion_delivered = 1 AND challenger_delivered = 1
            """,
            (duel_id,),
        ).fetchall()
        counts = PairCounts()
        for r in rows:
            counts = counts.add(int(r["champion_pass"]), int(r["challenger_pass"]))
        return counts

    def finish_duel(self, duel_id: int, status: str, counts: PairCounts, p_value: float,
                    finished_block: int | None) -> None:
        payload = json.dumps(counts.__dict__, sort_keys=True)
        with self.tx() as db:
            db.execute(
                """
                UPDATE duels
                SET status = ?, counts_json = ?, p_value = ?, finished_block = ?, updated_at = ?
                WHERE id = ?
                """,
                (status, payload, p_value, finished_block, now(), duel_id),
            )

    def abort_running_duels(self) -> None:
        with self.tx() as db:
            db.execute("UPDATE duels SET status='aborted_crash', updated_at=? WHERE status='running'", (now(),))

    def publication_intent(self, artifact: str, action: str, uid: int | None,
                           hotkey: str | None, dry_run: bool) -> int:
        ts = now()
        with self.tx() as db:
            row = db.execute(
                """
                SELECT id FROM publications
                WHERE artifact_id=? AND action=?
                  AND ((uid IS NULL AND ? IS NULL) OR uid = ?)
                  AND ((hotkey IS NULL AND ? IS NULL) OR hotkey = ?)
                  AND dry_run=?
                  AND status IN ('intent','submitted','failed','confirmed','dry_run')
                ORDER BY id DESC LIMIT 1
                """,
                (artifact, action, uid, uid, hotkey, hotkey, int(dry_run)),
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


def _champion(row: sqlite3.Row) -> Champion:
    return Champion(
        artifact_id=row["artifact_id"],
        model=row["model"],
        revision=row["revision"],
        uid=row["uid"],
        hotkey=row["hotkey"],
        reign_start=int(row["reign_start"]),
        backup_manifest=row["backup_manifest"],
        backup_prefix=row["backup_prefix"],
        payable=bool(row["payable"]),
    )


def _backup(row: sqlite3.Row) -> BackupRecord:
    return BackupRecord(
        artifact_id=row["artifact_id"],
        model=row["model"],
        revision=row["revision"],
        prefix=row["prefix"],
        manifest_key=row["manifest_key"],
        manifest_sha256=row["manifest_sha256"],
        status=row["status"],
    )
