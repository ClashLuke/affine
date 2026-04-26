from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "affine.analyze", *args],
        capture_output=True, text=True, check=False,
    )


def test_analyze_reports_ranking(tmp_path):
    path = tmp_path / "ev.jsonl"
    rows = (
        [{"m": 1, "r": "a", "e": "E", "c": i, "p": 1, "t": 0, "l": 100.0} for i in range(10)]
        + [{"m": 2, "r": "a", "e": "E", "c": i, "p": 0, "t": 0, "l": 100.0} for i in range(10)]
    )
    _write(path, rows)
    out = _run([str(path)])
    assert out.returncode == 0, out.stderr
    # uid 1 wins → appears first in ranking.
    lines = [l for l in out.stdout.splitlines() if l.strip().startswith(("1 ", "2 "))]
    assert lines and lines[0].lstrip().startswith("1"), out.stdout


def test_analyze_drop_fast_fails_changes_verdict(tmp_path):
    """The sampler fix: p=0 with latency<threshold is infra error, not a miner loss.
    Verify --drop-fast-fails materially changes the contrast the way we expect."""
    path = tmp_path / "ev.jsonl"
    # uid 2 "fails" only with 1s latency (pure infra); uid 1 legitimately passes slowly.
    rows = (
        [{"m": 1, "r": "a", "e": "E", "c": i, "p": 1, "t": 0, "l": 100.0} for i in range(10)]
        + [{"m": 2, "r": "a", "e": "E", "c": i, "p": 0, "t": 0, "l": 1.0} for i in range(20)]
        + [{"m": 2, "r": "a", "e": "E", "c": i, "p": 1, "t": 0, "l": 100.0} for i in range(20, 30)]
    )
    _write(path, rows)
    raw = _run([str(path), "--contrast", "2", "1"])
    fixed = _run([str(path), "--contrast", "2", "1", "--drop-fast-fails", "2.0"])
    assert raw.returncode == 0 and fixed.returncode == 0
    # Same rows, same uids, but the raw fit sees uid 2 as worse; the filtered fit
    # should give it a materially higher Δθ̂ (closer to or above 0).
    def _delta(o):
        for line in o.stdout.splitlines():
            if line.startswith("Δθ̂ ="):
                return float(line.split("=")[1].split()[0])
        raise AssertionError(f"no Δ in output: {o.stdout}")
    assert _delta(fixed) > _delta(raw)


def test_analyze_contrast_picks_most_recent_revision(tmp_path):
    """A re-committed uid has multiple (uid, rev) entries in evidence. Sorted
    iteration picks lex-first (≈ oldest), but the live loop's verdict is on the
    NEWEST revision. Without the most-recent-rev pick, a diff between the live
    verdict and analyze.py's --contrast looks like a scoring bug when it's just
    a stale rev. The fix tie-breaks by max-row-t."""
    path = tmp_path / "ev.jsonl"
    # uid 5 re-committed: old "rev_a" (time 100) lost a lot, new "rev_z" (time 200) wins.
    rows = (
        [{"m": 5, "r": "rev_a", "e": "E", "c": i, "p": 0, "t": 100, "l": 5.0} for i in range(10)]
        + [{"m": 5, "r": "rev_z", "e": "E", "c": i, "p": 1, "t": 200, "l": 5.0} for i in range(10)]
        + [{"m": 9, "r": "rev_a", "e": "E", "c": i, "p": 0, "t": 200, "l": 5.0} for i in range(10)]
    )
    _write(path, rows)
    out = _run([str(path), "--contrast", "5", "9"])
    assert out.returncode == 0, out.stderr
    # Most-recent rev for uid 5 is rev_z (winning). Δθ̂ should be positive.
    delta_line = next(l for l in out.stdout.splitlines() if l.startswith("Δθ̂ ="))
    delta = float(delta_line.split("=")[1].split()[0])
    assert delta > 0, f"expected positive Δθ̂ using rev_z (latest), got {delta}: {out.stdout}"
    assert "rev_z" in out.stdout, f"diagnostic should name the chosen revision: {out.stdout}"


def test_analyze_missing_file(tmp_path):
    out = _run([str(tmp_path / "nope.jsonl")])
    assert out.returncode == 1
    assert "no evidence" in out.stdout
