#!/usr/bin/env python3
"""Validator health/correctness check. Run periodically.

Detects: process down, evidence/reign missing or stale, verdicts fired from
slot-dead aborts (policy violation), reign mutated without a dwell-dethrone
log line, IRT fit blew up, synth-row fraction pathological, recent
unhandled tracebacks.

Exits 0 clean, 1 if any FAIL. Prints summary + details.
"""
from __future__ import annotations

import json, re, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from affine.chain import Miner
from affine.evidence import Row
from affine.irt import Priors
from affine.loop import _fit

PROBE_WINDOW_S = 1800   # how recently evidence/log must have moved
SYNTH_FAIL = 0.50
SYNTH_WARN = 0.20

problems: list[str] = []
warns: list[str] = []

def _pids() -> list[int]:
    try: return [int(p) for p in subprocess.check_output(["pgrep", "-f", ".shadow-venv/bin/affine"]).split()]
    except subprocess.CalledProcessError: return []

def _load_rows(path: Path) -> list[Row]:
    rows = []
    keep = {"m","r","e","c","p","t","l","i","k"}
    for ln, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip(): continue
        try: rows.append(Row(**{k:v for k,v in json.loads(line).items() if k in keep}))
        except Exception as e: problems.append(f"evidence line {ln}: {type(e).__name__}: {e}"); break
    return rows

def _verdict_blocks(text: str) -> list[tuple[int, int, str]]:
    """Returns (start_pos, end_pos, body) for each `duel: ... rows=N` block,
    delimited by the next `duel:` or `loop iteration` line. The block contains
    everything that happened in that duel attempt — abort warnings, dwell exit
    reason, and verdict log line if any."""
    starts = [(m.start(), m.end()) for m in re.finditer(r"affine\.loop INFO duel: king uid", text)]
    out = []
    for i, (s, _) in enumerate(starts):
        e = starts[i+1][0] if i+1 < len(starts) else len(text)
        out.append((s, e, text[s:e]))
    return out

# 1. process
pids = _pids()
if not pids: problems.append("no validator process running")
elif len(pids) > 1: warns.append(f"multiple validator processes: {pids}")

# 2. reign
reigns = [f for f in (REPO/".affine").glob("reign-*.json") if "contaminated" not in f.name]
reign = None
if not reigns: problems.append("no reign file")
else:
    try: reign = json.loads(reigns[0].read_text())
    except Exception as e: problems.append(f"reign file unparseable: {e}")

# 3. evidence
ev = REPO/".affine"/"evidence.jsonl"
rows: list[Row] = []
if not ev.exists(): problems.append("no evidence file")
else:
    rows = _load_rows(ev)
    age = time.time() - ev.stat().st_mtime
    if pids and age > PROBE_WINDOW_S:
        warns.append(f"evidence file unmodified {age:.0f}s while validator alive")

# 4. stderr log: scan recent duels for policy violations + tracebacks
logs = sorted((REPO/".shadow").glob("stderr-*.log"))
log_text = logs[-1].read_text() if logs else ""

if logs:
    age = time.time() - logs[-1].stat().st_mtime
    if pids and age > PROBE_WINDOW_S:
        warns.append(f"stderr log unmodified {age:.0f}s while validator alive")

# Slot-dead policy: no `verdict:` line may appear in a duel block that contains
# `dwell abort: ... slot dead`. Slot-dead must produce `duel aborted (..._slot_dead, ...)`.
sd_pat = re.compile(r"affine\.loop WARNING dwell abort: (king|chal) slot dead")
v_pat  = re.compile(r"affine\.loop INFO verdict: ")
ab_pat = re.compile(r"affine\.loop INFO duel aborted \((\w+)")
for s, e, body in _verdict_blocks(log_text):
    sd = sd_pat.search(body)
    if not sd: continue
    side = sd.group(1)
    v = v_pat.search(body)
    ab = ab_pat.search(body)
    if v is not None and (ab is None or v.start() < ab.start()):
        problems.append(f"slot-dead ({side}) followed by verdict — policy violated")
    elif ab is not None and ab.group(1) != f"{side}_slot_dead":
        warns.append(f"slot-dead ({side}) but abort reason = {ab.group(1)}")

# Recent traceback (last 200 lines). A traceback printed via logger.exception /
# exc_info=True is preceded by its WARNING/ERROR header on the prior line — that
# is where the module name sits, not on the `Traceback (most recent call last):`
# line itself. Treat sampler-attributed tracebacks as expected (infra failures
# already counted as synth rows); flag anything else.
tail_lines = log_text.splitlines()[-200:]
for i, line in enumerate(tail_lines):
    if "Traceback (most recent call last):" not in line: continue
    prev = tail_lines[i-1] if i else ""
    if "affine.sampler" in prev: continue
    problems.append("unhandled Traceback in recent stderr")
    break

# 5. fit on full evidence (sanity)
if rows:
    mids = {}
    for r in rows: mids[r.m] = (r.r, r.k or "?")
    miners = [Miner(uid=u, hotkey=f"hk{u}", model=k, revision=rev, block=0) for u,(rev,k) in mids.items()]
    env_names = sorted({r.e for r in rows})
    try:
        fit = _fit(rows, miners, env_names, Priors())
        if fit.degenerate: warns.append("full-evidence fit is degenerate")
    except Exception as e:
        problems.append(f"fit raised: {type(e).__name__}: {e}")

# 6. synth fraction
if rows:
    pct = sum(1 for r in rows if r.l == 0.0) / len(rows)
    if pct >= SYNTH_FAIL:   problems.append(f"synth fraction {pct:.0%} (>={SYNTH_FAIL:.0%})")
    elif pct >= SYNTH_WARN: warns.append(f"synth fraction {pct:.0%} (>={SYNTH_WARN:.0%})")

# Reign should match recent dethrone log line if uid != original cold-start.
# Scan for most recent `DETHRONE: uid X → uid Y`; reign uid should match Y.
if reign and log_text:
    last = None
    for m in re.finditer(r"DETHRONE: uid \d+ → uid (\d+)", log_text): last = m
    if last is not None and int(last.group(1)) != reign["uid"]:
        warns.append(f"reign uid={reign['uid']} but most recent DETHRONE log → uid {last.group(1)}")

# Output
ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
pid_s = pids[0] if pids else "NONE"
reign_s = f"uid{reign['uid']}@{reign['reign_start']}b" if reign else "NONE"
print(f"[{ts}] pid={pid_s} rows={len(rows)} reign={reign_s} | {'OK' if not problems and not warns else ('WARN' if not problems else 'FAIL')}")
for w in warns: print(f"  WARN: {w}")
for p in problems: print(f"  FAIL: {p}")
sys.exit(1 if problems else 0)
