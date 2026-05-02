import json
from types import SimpleNamespace

from affine.chain import Miner, _parse_commitments, _tiebreak


def _meta(*hotkeys):
    return SimpleNamespace(hotkeys=list(hotkeys))


def _good(model, rev):
    return json.dumps({"model": model, "revision": rev})


def test_parse_commitments():
    meta = _meta("vali", "hkA", "hkB")
    out = _parse_commitments(
        meta,
        {"vali": (5, _good("v", "r")),
         "hkA": (42, _good("m1", "r1")),
         "hkB": (99, _good("m2", "r2")),
         "ghost": (1, _good("nope", "nope"))},
        exclude_hotkey="vali",
    )
    by_uid = {m.uid: m for m in out}
    assert set(by_uid) == {1, 2}
    assert (by_uid[1].hotkey, by_uid[1].block) == ("hkA", 42)
    assert (by_uid[2].hotkey, by_uid[2].block) == ("hkB", 99)


def test_drops_invalid_payloads():
    meta = _meta("hk0", "hk1", "hk2", "hk3", "hk4", "hk5", "hk6")
    out = _parse_commitments(meta, {
        "hk0": (1, "not json"),
        "hk1": (1, "[]"),
        "hk2": (1, json.dumps({"model": ["m"], "revision": "r"})),
        "hk3": (1, json.dumps({"model": "m", "revision": ["r"]})),
        "hk4": (1, json.dumps({"model": "", "revision": "r"})),
        "hk5": (1, json.dumps({"model": "m" * 257, "revision": "r"})),
        "hk6": (1, _good("m", "r")),
    }, exclude_hotkey=None)
    assert [m.hotkey for m in out] == ["hk6"]


def test_tiebreak_keys_only_on_hotkey_and_revision():
    a = Miner(uid=0, hotkey="alpha", model="m", revision="r1", block=10)
    a_moved = Miner(uid=99, hotkey="alpha", model="m", revision="r1", block=999)
    b = Miner(uid=1, hotkey="beta", model="m", revision="r1", block=10)
    a_diff_rev = Miner(uid=0, hotkey="alpha", model="m", revision="r2", block=10)
    assert _tiebreak(a) == _tiebreak(a_moved)
    assert _tiebreak(a) != _tiebreak(b)
    assert _tiebreak(a) != _tiebreak(a_diff_rev)
