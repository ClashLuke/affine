import json
from types import SimpleNamespace

from affine.chain import Miner, _parse_commitments, _tiebreak


def _meta(*hotkeys: str):
    return SimpleNamespace(hotkeys=list(hotkeys))


def _good(model: str, rev: str) -> str:
    return json.dumps({"model": model, "revision": rev})


def test_skips_excluded_hotkey():
    meta = _meta("vali", "minerA")
    commits = {"vali": (10, _good("m", "r")), "minerA": (11, _good("m2", "r2"))}
    out = _parse_commitments(meta, commits, exclude_hotkey="vali")
    assert [m.hotkey for m in out] == ["minerA"]


def test_drops_invalid_payloads():
    meta = _meta("hk0", "hk1", "hk2", "hk3", "hk4", "hk5", "hk6")
    commits = {
        "hk0": (1, "not json"),
        "hk1": (1, "[]"),
        "hk2": (1, json.dumps({"model": ["m"], "revision": "r"})),
        "hk3": (1, json.dumps({"model": "m", "revision": ["r"]})),
        "hk4": (1, json.dumps({"model": "", "revision": "r"})),
        "hk5": (1, json.dumps({"model": "m" * 257, "revision": "r"})),
        "hk6": (1, _good("m", "r")),
    }
    out = _parse_commitments(meta, commits, exclude_hotkey=None)
    assert [m.hotkey for m in out] == ["hk6"]


def test_uid_alignment_and_block_passthrough():
    meta = _meta("hkA", "hkB", "hkC")
    commits = {"hkB": (42, _good("m", "r")), "hkC": (99, _good("m2", "r2"))}
    out = _parse_commitments(meta, commits, exclude_hotkey=None)
    by_uid = {m.uid: m for m in out}
    assert set(by_uid) == {1, 2}
    assert by_uid[1].hotkey == "hkB" and by_uid[1].block == 42
    assert by_uid[2].hotkey == "hkC" and by_uid[2].block == 99


def test_missing_hotkey_skipped():
    meta = _meta("hk0")
    commits = {"hk0": (1, _good("m", "r")), "ghost": (1, _good("g", "g"))}
    out = _parse_commitments(meta, commits, exclude_hotkey=None)
    assert [m.uid for m in out] == [0]


def test_tiebreak_is_deterministic_and_hotkey_keyed():
    a = Miner(uid=0, hotkey="alpha", model="m", revision="r1", block=10)
    b = Miner(uid=1, hotkey="beta", model="m", revision="r1", block=10)
    a2 = Miner(uid=99, hotkey="alpha", model="m", revision="r1", block=999)
    assert _tiebreak(a) == _tiebreak(a2)
    assert _tiebreak(a) != _tiebreak(b)
    a_diff_rev = Miner(uid=0, hotkey="alpha", model="m", revision="r2", block=10)
    assert _tiebreak(a) != _tiebreak(a_diff_rev)


def test_tiebreak_ordering_independent_of_uid():
    a = Miner(uid=0, hotkey="zzz", model="m", revision="r", block=5)
    b = Miner(uid=1, hotkey="aaa", model="m", revision="r", block=5)
    forward = sorted([a, b], key=lambda m: (m.block, _tiebreak(m)))
    swapped = sorted(
        [Miner(uid=1, hotkey="zzz", model="m", revision="r", block=5),
         Miner(uid=0, hotkey="aaa", model="m", revision="r", block=5)],
        key=lambda m: (m.block, _tiebreak(m)),
    )
    assert [m.hotkey for m in forward] == [m.hotkey for m in swapped]
