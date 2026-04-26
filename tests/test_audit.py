from __future__ import annotations

import json

from affine.audit import audit


def test_audit_noop_without_env(monkeypatch):
    monkeypatch.delenv("AFFINE_SHADOW_LOG", raising=False)
    audit(type="duel", verdict="CHAMPION_HOLDS")  # must not raise


def test_audit_writes_jsonl(monkeypatch, tmp_path):
    path = tmp_path / "shadow.jsonl"
    monkeypatch.setenv("AFFINE_SHADOW_LOG", str(path))
    audit(type="weight_intent", netuid=120, winner_uid=7, dry_run=True)
    audit(type="duel", verdict="CHALLENGER_WINS",
          king={"uid": 1, "model": "k"}, challenger={"uid": 2, "model": "c"},
          delta=0.42, se=0.11, z=3.8, k=2.5,
          rows_per_env={"ded": [3, 4, 1, 4], "abd": [2, 4, 2, 4]})
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    wi, du = json.loads(lines[0]), json.loads(lines[1])
    assert wi["type"] == "weight_intent" and wi["winner_uid"] == 7
    assert du["verdict"] == "CHALLENGER_WINS" and du["z"] == 3.8
    assert du["rows_per_env"]["ded"] == [3, 4, 1, 4]
    assert "ts" in wi and "ts" in du


def test_audit_sanitizes_non_finite_floats(monkeypatch, tmp_path):
    """A degenerate Laplace cov produces se=0 → z=delta/0=inf. allow_nan=False
    used to drop the whole record; sanitization keeps it (with non-finite
    fields rewritten to None) so verdict logs survive numerical edge cases."""
    path = tmp_path / "shadow.jsonl"
    monkeypatch.setenv("AFFINE_SHADOW_LOG", str(path))
    audit(type="duel", verdict="CHAMPION_HOLDS",
          delta=float("nan"), se=0.0, z=float("inf"), k=1.0,
          rows_per_env={"ded": [1, 1, 0, 1]})
    line = path.read_text().strip().splitlines()
    assert len(line) == 1
    rec = json.loads(line[0])
    assert rec["delta"] is None and rec["z"] is None
    assert rec["se"] == 0.0 and rec["k"] == 1.0
    assert rec["rows_per_env"]["ded"] == [1, 1, 0, 1]
