from __future__ import annotations

import pytest

from affine.evidence import EvidenceStore, Row


def _row(**kw):
    defaults = dict(m=1, r="rev", e="ded", c=0, p=1, t=100, l=0.5)
    defaults.update(kw)
    return Row(**defaults)


def test_append_then_read_roundtrip(tmp_path):
    store = EvidenceStore(tmp_path / "ev.jsonl")
    store.append(_row(c=0))
    store.append(_row(c=1, p=0))
    rows = store.read()
    assert len(rows) == 2
    assert rows[0].c == 0 and rows[0].p == 1
    assert rows[1].c == 1 and rows[1].p == 0


def test_next_counter_advances_only_on_append(tmp_path):
    store = EvidenceStore(tmp_path / "ev.jsonl")
    # Idempotent peek without a write — a sample whose evidence is never written
    # (infra failure) must not consume a counter slot, otherwise the seed gets
    # reused on a fresh attempt after restart.
    assert store.next_counter(1, "rev", "ded") == 0
    assert store.next_counter(1, "rev", "ded") == 0
    store.append(_row(c=0))
    assert store.next_counter(1, "rev", "ded") == 1
    store.append(_row(c=1))
    assert store.next_counter(1, "rev", "ded") == 2
    # Per-key isolation.
    assert store.next_counter(1, "rev", "abd") == 0
    assert store.next_counter(2, "rev", "ded") == 0


def test_counter_unaffected_by_failed_sample(tmp_path):
    """Regression: a failed sample (no append) must leave the counter at the same
    value so the next attempt reuses the c (and therefore the same seed). Without
    this, an infra-failed sample 'burns' a c which is then never represented on
    disk, and a restart rebuilds the counter from disk and rolls back."""
    store = EvidenceStore(tmp_path / "ev.jsonl")
    c1 = store.next_counter(1, "rev", "ded")
    # Simulate an infra failure: caller obtained c, ran the sample, got None,
    # never called append. Counter must still report c1 next time.
    c2 = store.next_counter(1, "rev", "ded")
    assert c1 == c2 == 0


def test_counter_warms_from_existing_file(tmp_path):
    p = tmp_path / "ev.jsonl"
    s1 = EvidenceStore(p)
    s1.append(_row(c=s1.next_counter(1, "rev", "ded")))
    s1.append(_row(c=s1.next_counter(1, "rev", "ded")))
    s1.append(_row(m=2, c=s1.next_counter(2, "rev", "ded")))

    s2 = EvidenceStore(p)
    assert s2.next_counter(1, "rev", "ded") == 2
    assert s2.next_counter(2, "rev", "ded") == 1


def test_read_skips_malformed_lines(tmp_path):
    p = tmp_path / "ev.jsonl"
    store = EvidenceStore(p)
    store.append(_row(c=0))
    with p.open("a") as f:
        f.write("not-json\n")
        f.write('{"incomplete": true}\n')
    store.append(_row(c=1))
    rows = EvidenceStore(p).read()
    assert len(rows) == 2
    assert {r.c for r in rows} == {0, 1}


def test_read_survives_torn_utf8_tail(tmp_path):
    """Regression: a SIGKILL mid-write of a multibyte UTF-8 char (e.g. 中 in
    a non-ASCII model name) used to crash startup — text-mode `for line in f`
    decodes at iteration time and raises UnicodeDecodeError outside the try.
    Loader must skip the corrupt line and continue, so the validator boots."""
    p = tmp_path / "ev.jsonl"
    store = EvidenceStore(p)
    store.append(_row(c=0))
    # First two bytes of a 3-byte UTF-8 codepoint, no terminator. Heal_torn_tail
    # writes a \n after; the result is an undecodable line.
    with p.open("ab") as f:
        f.write(b'{"r": "Qwen-\xe4\xb8')
    rows = EvidenceStore(p).read()
    assert len(rows) == 1 and rows[0].c == 0


def test_row_rejects_bad_p():
    for bad in (-1, 2, 10):
        with pytest.raises(ValueError):
            _row(p=bad)


def test_append_pair_writes_both_rows_in_one_syscall(tmp_path, monkeypatch):
    """Matched-task duels must persist both rows in one write(). A second
    syscall opens a torn-write window where the king row lands but the
    challenger row doesn't, leaving one-sided evidence. fsync must follow
    so the in-memory counters never advance ahead of what's on disk."""
    import os
    store = EvidenceStore(tmp_path / "ev.jsonl")
    target_fd = [None]
    write_calls = [0]
    fsync_calls = [0]
    real_open, real_write, real_fsync = os.open, os.write, os.fsync
    def counting_open(path, *a, **kw):
        fd = real_open(path, *a, **kw)
        if str(path) == str(store.path):
            target_fd[0] = fd
        return fd
    def counting_write(fd, data):
        if fd == target_fd[0]:
            write_calls[0] += 1
        return real_write(fd, data)
    def counting_fsync(fd):
        if fd == target_fd[0]:
            fsync_calls[0] += 1
        return real_fsync(fd)
    monkeypatch.setattr(os, "open", counting_open)
    monkeypatch.setattr(os, "write", counting_write)
    monkeypatch.setattr(os, "fsync", counting_fsync)
    store.append(_row(m=1, c=0, p=1), _row(m=2, c=0, p=0))
    assert write_calls[0] == 1
    assert fsync_calls[0] == 1
    rows = EvidenceStore(tmp_path / "ev.jsonl").read()
    assert [r.m for r in rows] == [1, 2]


def test_append_pair_truncates_on_short_write(tmp_path, monkeypatch):
    """Regression: ENOSPC mid-write can leave row A complete and row B partial.
    On restart heal_torn_tail seals it, row A loads, row B is dropped — and the
    challenger's counter regresses while the king's advances. Next dwell uses a
    stale SHA seed for the challenger but a fresh one for the king → matched-task
    pairing broken. Fix: truncate to pre-write size on any short/failed write."""
    import os
    store = EvidenceStore(tmp_path / "ev.jsonl")
    store.append(_row(m=1, c=0, p=1), _row(m=2, c=0, p=0))
    pre_size = store.path.stat().st_size

    real_write = os.write
    target_fd = [None]
    real_open = os.open
    def open_capture(path, *a, **kw):
        fd = real_open(path, *a, **kw)
        if str(path) == str(store.path):
            target_fd[0] = fd
        return fd
    def short_write(fd, data):
        if fd == target_fd[0]:
            return real_write(fd, data[: len(data) // 2])  # only half lands on disk
        return real_write(fd, data)
    monkeypatch.setattr(os, "open", open_capture)
    monkeypatch.setattr(os, "write", short_write)
    with pytest.raises(OSError, match="short write"):
        store.append(_row(m=1, c=1, p=1), _row(m=2, c=1, p=0))
    # File is rolled back: no torn tail, no orphan partial row.
    assert store.path.stat().st_size == pre_size
    rows = EvidenceStore(tmp_path / "ev.jsonl").read()
    assert len(rows) == 2  # only the original pre-failure pair


def test_append_pair_rolls_back_on_fsync_failure(tmp_path, monkeypatch):
    """fsync failure (EIO, ENOSPC late, hardware fault) leaves durability
    unconfirmed. Without rollback the unflushed pair sits half-visible to the next
    process while in-memory counters never advanced — the next append would write
    over a `c` that may or may not be on disk. Fix: ftruncate back to pre-write size
    so counters and disk agree."""
    import os
    store = EvidenceStore(tmp_path / "ev.jsonl")
    store.append(_row(m=1, c=0, p=1), _row(m=2, c=0, p=0))
    pre_size = store.path.stat().st_size

    real_open, real_fsync = os.open, os.fsync
    target_fd = [None]
    def open_capture(path, *a, **kw):
        fd = real_open(path, *a, **kw)
        if str(path) == str(store.path):
            target_fd[0] = fd
        return fd
    def fail_fsync(fd):
        if fd == target_fd[0]:
            raise OSError("fsync failed: EIO")
        return real_fsync(fd)
    monkeypatch.setattr(os, "open", open_capture)
    monkeypatch.setattr(os, "fsync", fail_fsync)
    with pytest.raises(OSError, match="fsync failed"):
        store.append(_row(m=1, c=1, p=1), _row(m=2, c=1, p=0))
    # File is rolled back to pre-failure size.
    assert store.path.stat().st_size == pre_size
    # Counters didn't advance (the failed pair shouldn't be considered written).
    assert store.next_counter(1, "rev", "ded") == 1
    assert store.next_counter(2, "rev", "ded") == 1
    rows = EvidenceStore(tmp_path / "ev.jsonl").read()
    assert len(rows) == 2


def test_append_pair_advances_both_counters(tmp_path):
    store = EvidenceStore(tmp_path / "ev.jsonl")
    store.append(_row(m=1, r="r1", c=0), _row(m=2, r="r2", c=0))
    assert store.next_counter(1, "r1", "ded") == 1
    assert store.next_counter(2, "r2", "ded") == 1


def test_read_rejects_nan_inf_and_wrong_types(tmp_path):
    p = tmp_path / "ev.jsonl"
    p.write_text(
        '{"m":1,"r":"rv","e":"e","c":0,"p":1,"t":0,"l":NaN}\n'
        '{"m":1,"r":"rv","e":"e","c":1,"p":1,"t":0,"l":Infinity}\n'
        '{"m":1,"r":[],"e":"e","c":2,"p":1,"t":0,"l":1.0}\n'
        '{"m":1,"r":"rv","e":"e","c":3,"p":1,"t":0,"l":1.0}\n'
    )
    rows = EvidenceStore(p).read()
    assert len(rows) == 1
    assert rows[0].c == 3


def test_read_skips_non_dict_rows(tmp_path):
    """A JSON line that's not an object (e.g., array, scalar) must be skipped,
    not crash the read. Otherwise one corrupted line takes down all subsequent
    rows — and the validator's startup."""
    p = tmp_path / "ev.jsonl"
    p.write_text(
        '{"m":1,"r":"rv","e":"e","c":0,"p":1,"t":0,"l":1.0}\n'
        '[1,2,3]\n'
        '"just a string"\n'
        '42\n'
        '{"m":2,"r":"rv","e":"e","c":1,"p":0,"t":0,"l":1.0}\n'
    )
    rows = EvidenceStore(p).read()
    assert [r.m for r in rows] == [1, 2]
