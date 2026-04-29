from affine.paired import PairCounts
from affine.store import BackupRecord, Champion, PairSample, Store, artifact_id


def test_store_round_trips_champion_backup_duel_and_samples(tmp_path):
    store = Store(tmp_path / "affine.sqlite3")
    art = artifact_id("Qwen/Qwen3-32B", "rev")
    backup = BackupRecord(art, "Qwen/Qwen3-32B", "rev", "prefix", "prefix/manifest.json", "sha", "verified")
    champion = Champion(art, "Qwen/Qwen3-32B", "rev", None, None, 10,
                        backup.manifest_key, backup.prefix, False)

    store.set_champion(champion, backup)
    assert store.champion() == champion

    duel = store.create_duel(
        champion=champion,
        challenger_uid=7,
        challenger_hotkey="hk7",
        challenger_model="m",
        challenger_revision="r",
        schedule_seed="seed",
        pairs_per_env=2,
        min_discordant=1,
        alpha=0.05,
        started_block=11,
    )
    store.add_samples([
        PairSample(duel.id, "E", 1, 0, 11, 0, 1, 1.0, 1.0, 1, 1, 3, 4),
        PairSample(duel.id, "E", 2, 1, 11, 1, 1, 1.0, 1.0, 1, 1, 3, 4),
    ])
    assert store.counts(duel.id) == PairCounts(challenger_only=1, both_pass=1)

    store.finish_duel(duel.id, "hold", store.counts(duel.id), 0.5, 12)
    store.close()


def test_store_counts_only_both_delivered_pairs(tmp_path):
    store = Store(tmp_path / "affine.sqlite3")
    art = artifact_id("champ", "sha")
    backup = BackupRecord(art, "champ", "sha", "p", "p/manifest.json", "sha", "verified")
    champion = Champion(art, "champ", "sha", 1, "hk1", 0, backup.manifest_key, backup.prefix, True)
    store.set_champion(champion, backup)
    duel = store.create_duel(
        champion=champion,
        challenger_uid=2,
        challenger_hotkey="hk2",
        challenger_model="chal",
        challenger_revision="sha2",
        schedule_seed="seed",
        pairs_per_env=3,
        min_discordant=1,
        alpha=0.05,
        started_block=1,
    )
    store.add_samples([
        PairSample(duel.id, "E", 1, 0, 1, 0, 1, 0.0, 1.0, 0, 1, 0, 4),
        PairSample(duel.id, "E", 2, 1, 1, 1, 0, 1.0, 0.0, 1, 0, 3, 0),
        PairSample(duel.id, "E", 3, 2, 1, 0, 1, 1.0, 1.0, 1, 1, 3, 4),
    ])
    assert store.counts(duel.id) == PairCounts(challenger_only=1)
    store.close()


def test_store_demotes_champion_payment_identity(tmp_path):
    store = Store(tmp_path / "affine.sqlite3")
    art = artifact_id("champ", "sha")
    backup = BackupRecord(art, "champ", "sha", "p", "p/manifest.json", "sha", "verified")
    store.set_champion(Champion(art, "champ", "sha", 7, "hk7", 0,
                                backup.manifest_key, backup.prefix, True), backup)
    assert store.demote_champion(art) is True
    champ = store.champion()
    assert champ.uid is None
    assert champ.hotkey is None
    assert champ.payable is False
    store.close()


def test_store_records_staging_backups(tmp_path):
    store = Store(tmp_path / "affine.sqlite3")
    backup = BackupRecord("art", "m", "r", "p", "p/manifest.json", "sha", "verified")
    store.record_backup(backup, "staging")
    assert store.staging_backups() == [BackupRecord("art", "m", "r", "p", "p/manifest.json", "sha", "staging")]
    store.mark_backup_deleted(backup.manifest_key)
    assert store.staging_backups() == []
    store.close()


def test_publication_intent_separates_dry_run_and_real_targets(tmp_path):
    store = Store(tmp_path / "affine.sqlite3")
    dry = store.publication_intent("art", "set_weights", 1, "hk1", True)
    store.mark_publication(dry, "dry_run")
    real = store.publication_intent("art", "set_weights", 1, "hk1", False)
    other_uid = store.publication_intent("art", "set_weights", 2, "hk2", False)
    assert real != dry
    assert other_uid != real
    store.close()


def test_publication_intent_republishes_after_later_burn(tmp_path):
    store = Store(tmp_path / "affine.sqlite3")
    first = store.publication_intent("art", "set_weights", 1, "hk1", False)
    store.mark_publication(first, "confirmed")
    burn = store.publication_intent("art", "burn", None, None, False)
    store.mark_publication(burn, "confirmed")
    second = store.publication_intent("art", "set_weights", 1, "hk1", False)
    assert second != first
    store.close()


def test_store_keeps_same_task_id_at_different_iterations(tmp_path):
    store = Store(tmp_path / "affine.sqlite3")
    art = artifact_id("champ", "sha")
    backup = BackupRecord(art, "champ", "sha", "p", "p/manifest.json", "sha", "verified")
    champion = Champion(art, "champ", "sha", 1, "hk1", 0, backup.manifest_key, backup.prefix, True)
    store.set_champion(champion, backup)
    duel = store.create_duel(
        champion=champion,
        challenger_uid=2,
        challenger_hotkey="hk2",
        challenger_model="chal",
        challenger_revision="sha2",
        schedule_seed="seed",
        pairs_per_env=3,
        min_discordant=1,
        alpha=0.05,
        started_block=1,
    )
    store.add_samples([
        PairSample(duel.id, "E", 0, 0, 1, 0, 1, 1.0, 1.0, 1, 1, 3, 4),
        PairSample(duel.id, "E", 0, 1, 1, 0, 1, 1.0, 1.0, 1, 1, 3, 4),
        PairSample(duel.id, "E", 0, 2, 1, 0, 1, 1.0, 1.0, 1, 1, 3, 4),
    ])
    assert store.counts(duel.id) == PairCounts(challenger_only=3)
    store.close()


def test_store_marks_old_backup_retiring_on_champion_switch(tmp_path):
    store = Store(tmp_path / "affine.sqlite3")
    first_art = artifact_id("a", "1")
    second_art = artifact_id("b", "2")
    first_backup = BackupRecord(first_art, "a", "1", "p1", "p1/manifest.json", "sha1", "verified")
    second_backup = BackupRecord(second_art, "b", "2", "p2", "p2/manifest.json", "sha2", "verified")
    store.set_champion(Champion(first_art, "a", "1", None, None, 1,
                                first_backup.manifest_key, first_backup.prefix, False), first_backup)
    store.set_champion(Champion(second_art, "b", "2", 2, "hk2", 2,
                                second_backup.manifest_key, second_backup.prefix, True), second_backup)

    retiring = store.retiring_backups()
    assert [b.manifest_key for b in retiring] == [first_backup.manifest_key]
    store.close()


def test_pending_champion_does_not_retire_current_until_finalized(tmp_path):
    store = Store(tmp_path / "affine.sqlite3")
    first_art = artifact_id("a", "1")
    second_art = artifact_id("b", "2")
    first_backup = BackupRecord(first_art, "a", "1", "p1", "p1/manifest.json", "sha1", "verified")
    second_backup = BackupRecord(second_art, "b", "2", "p2", "p2/manifest.json", "sha2", "verified")
    first = Champion(first_art, "a", "1", 1, "hk1", 1, first_backup.manifest_key, first_backup.prefix, True)
    second = Champion(second_art, "b", "2", 2, "hk2", 2, second_backup.manifest_key, second_backup.prefix, True)

    store.set_champion(first, first_backup)
    store.set_pending_champion(second, second_backup)
    assert store.champion() == first
    assert store.pending_champion() == second
    assert store.retiring_backups() == []

    assert store.finalize_pending_champion() == second
    assert store.champion() == second
    assert [b.manifest_key for b in store.retiring_backups()] == [first_backup.manifest_key]
    store.close()


def test_clear_pending_champion_returns_backup_for_cleanup(tmp_path):
    store = Store(tmp_path / "affine.sqlite3")
    first_art = artifact_id("a", "1")
    second_art = artifact_id("b", "2")
    first_backup = BackupRecord(first_art, "a", "1", "p1", "p1/manifest.json", "sha1", "verified")
    second_backup = BackupRecord(second_art, "b", "2", "p2", "p2/manifest.json", "sha2", "verified")
    first = Champion(first_art, "a", "1", 1, "hk1", 1, first_backup.manifest_key, first_backup.prefix, True)
    second = Champion(second_art, "b", "2", 2, "hk2", 2, second_backup.manifest_key, second_backup.prefix, True)

    store.set_champion(first, first_backup)
    store.set_pending_champion(second, second_backup)

    cleared = store.clear_pending_champion()

    assert cleared == BackupRecord(second_art, "b", "2", "p2", "p2/manifest.json", "sha2", "pending")
    assert store.pending_champion() is None
    assert [b.manifest_key for b in store.staging_backups()] == [second_backup.manifest_key]
    assert store.retiring_backups() == []
    store.close()
