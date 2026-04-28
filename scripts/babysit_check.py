#!/usr/bin/env python3
"""Validator health/correctness check + debug dashboard. Run periodically.

Detects: process down, evidence/reign missing or stale, verdicts fired from
slot-dead aborts (policy violation), reign mutated without a dwell-dethrone
log line, IRT fit blew up, synth-row fraction pathological, recent
unhandled tracebacks.

Beyond pass/fail, dumps the full leaderboard, throughput rates (sample/s,
tok/s, mean latency), turnover (dethrones, reign durations), duel outcome
breakdown, sampler/provisioning error stats, and evidence coverage — every
number that helps explain why the validator is or isn't making progress.

Exits 0 clean, 1 if any FAIL.
"""
from __future__ import annotations

import json, math, re, subprocess, sys, time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from affine.chain import Miner
from affine.config import Config
from affine.evidence import Row
from affine.irt import Priors, compute_k
from affine.loop import _fit

PROBE_WINDOW_S = 1800
# Fraction of (env, block, iter) pairs the fit drops because at least one side
# is synthetic (l=0). The fit retention is what shapes Δθ̂; raw row synth count
# is a sampler-health stat, not a fit-correctness one.
LOST_PAIR_FAIL = 0.80
LOST_PAIR_WARN = 0.50
RECENT_VERDICTS = 5

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
    """Per-duel slices delimited by `affine.loop INFO duel: king uid` lines.
    Each slice contains everything that happened in that duel attempt — abort
    warnings, dwell exit reason, verdict log line if any."""
    starts = [(m.start(), m.end()) for m in re.finditer(r"affine\.loop INFO duel: king uid", text)]
    out = []
    for i, (s, _) in enumerate(starts):
        e = starts[i+1][0] if i+1 < len(starts) else len(text)
        out.append((s, e, text[s:e]))
    return out

def _fmt_blocks(b: int) -> str:
    s = b * 12
    if s < 60: return f"{s}s"
    if s < 3600: return f"{s/60:.1f}m"
    if s < 86400: return f"{s/3600:.1f}h"
    return f"{s/86400:.1f}d"

def _pct(n: int, d: int) -> str:
    return f"{100*n/d:.0f}%" if d else "—"

# ---------------- collect ----------------

pids = _pids()
if not pids: problems.append("no validator process running")
elif len(pids) > 1: warns.append(f"multiple validator processes: {pids}")

reigns = [f for f in (REPO/".affine").glob("reign-*.json") if "contaminated" not in f.name]
reign = None
if not reigns: problems.append("no reign file")
else:
    try: reign = json.loads(reigns[0].read_text())
    except Exception as e: problems.append(f"reign file unparseable: {e}")

ev = REPO/".affine"/"evidence.jsonl"
rows: list[Row] = []
if not ev.exists(): problems.append("no evidence file")
else:
    rows = _load_rows(ev)
    age = time.time() - ev.stat().st_mtime
    if pids and age > PROBE_WINDOW_S:
        warns.append(f"evidence file unmodified {age:.0f}s while validator alive")

logs = sorted((REPO/".shadow").glob("stderr-*.log"))
log_text = logs[-1].read_text() if logs else ""
all_log_text = "".join(p.read_text() for p in logs)

if logs:
    age = time.time() - logs[-1].stat().st_mtime
    if pids and age > PROBE_WINDOW_S:
        warns.append(f"stderr log unmodified {age:.0f}s while validator alive")

# Slot-dead policy: no `verdict:` may appear in a duel slice that contains
# `dwell abort: ... slot dead`. Slot-dead must produce `duel aborted (..._slot_dead, ...)`.
sd_pat = re.compile(r"affine\.loop WARNING dwell abort: (king|chal) slot dead")
v_pat  = re.compile(r"affine\.loop INFO verdict: ")
ab_pat = re.compile(r"affine\.loop INFO duel aborted \((\w+)")
for s, e, body in _verdict_blocks(log_text):
    sd = sd_pat.search(body)
    if not sd: continue
    side = sd.group(1)
    v = v_pat.search(body); ab = ab_pat.search(body)
    if v is not None and (ab is None or v.start() < ab.start()):
        problems.append(f"slot-dead ({side}) followed by verdict — policy violated")
    elif ab is not None and ab.group(1) != f"{side}_slot_dead":
        warns.append(f"slot-dead ({side}) but abort reason = {ab.group(1)}")

# Recent traceback (last 200 lines). Sampler-attributed tracebacks are expected
# (infra failures already counted as synth rows); flag anything else.
tail_lines = log_text.splitlines()[-200:]
for i, line in enumerate(tail_lines):
    if "Traceback (most recent call last):" not in line: continue
    prev = tail_lines[i-1] if i else ""
    if "affine.sampler" in prev: continue
    problems.append("unhandled Traceback in recent stderr")
    break

# IRT fit on full evidence.
fit = None
miners: list[Miner] = []
env_names: list[str] = []
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

pair_groups: dict[tuple[str,int,int], list[float]] = {}
for r in rows:
    pair_groups.setdefault((r.e, r.t, r.i), []).append(r.l)
lost_pairs = sum(1 for ls in pair_groups.values() if any(l == 0.0 for l in ls))
total_pairs = len(pair_groups)
if total_pairs:
    pct = lost_pairs / total_pairs
    if pct >= LOST_PAIR_FAIL:   problems.append(f"lost-pair fraction {pct:.0%} (>={LOST_PAIR_FAIL:.0%})")
    elif pct >= LOST_PAIR_WARN: warns.append(f"lost-pair fraction {pct:.0%} (>={LOST_PAIR_WARN:.0%})")

# Reign should match recent dethrone log line.
if reign and log_text:
    last = None
    for m in re.finditer(r"DETHRONE: uid \d+ → uid (\d+)", log_text): last = m
    if last is not None and int(last.group(1)) != reign["uid"]:
        warns.append(f"reign uid={reign['uid']} but most recent DETHRONE log → uid {last.group(1)}")

# ---------------- aggregate stats ----------------

# Duels and verdicts.
duel_re      = re.compile(r"affine\.loop INFO duel: king uid(\d+) vs chal uid(\d+) \(rows=(\d+)\)")
verdict_re   = re.compile(r"affine\.loop INFO verdict: Δθ̂=([+-][\d.]+)±([\d.]+) z=([+-][\d.]+) k=([\d.]+) reign=(\d+)b")
abort_re     = re.compile(r"affine\.loop INFO duel aborted \((\w+), rows=(\d+)\)")
dethrone_re  = re.compile(r"affine\.loop INFO DETHRONE: uid (\d+) → uid (\d+)")

# Throughput (last log only — current session reflects current loop behavior).
dwell_re = re.compile(
    r"dwell pairs=(\d+) \(in_flight=(\d+)/(\d+)\) samples=(\d+) tokens=(\d+) "
    r"elapsed=([\d.]+)s throughput=([\d.]+) sample/s (\d+) tok/s mean_lat=([\d.]+)s")

# Sampler errors.
infra_re   = re.compile(r"affine\.sampler WARNING infra failure \(([^)]+)\) in ([\d.]+)s, error_type=(\w+):")
timeout_re = re.compile(r"affine\.sampler WARNING sample timeout \(([^)]+)\) in ([\d.]+)s")

# Provisioning.
slot_ready_re   = re.compile(r"affine\.vllm INFO slot ready after (\d+)s:")
crashloop_re    = re.compile(r"affine\.loop WARNING provision crashloop")
prov_timeout_re = re.compile(r"affine\.loop WARNING provision timeout")
prov_trans_re   = re.compile(r"affine\.loop WARNING provision transient")
rent_register_re= re.compile(r"affine\.vllm INFO targon rental register:")
warm_ok_re      = re.compile(r"affine\.vllm INFO warm hit ok in ([\d.]+)s:")
# warm-hit failures surface as crashloop log lines (SlotProvisionFailed raised
# from _warm_hit → caught by loop's provision error handler). Distinguish by
# the embedded message so we don't conflate "pod can't even serve a 1-token
# completion" with "pod entered POD_CRASHLOOP_BACKOFF before /v1/models 200".
warm_fail_re    = re.compile(r"affine\.loop WARNING provision crashloop[^\n]*warm hit (?:failed|returned)")
# vLLM v1 catch-all for any unhandled engine-core exception. Drop in this count
# is the primary success signal for the cudagraph-profile / warm-hit fix.
engine_dead_re  = re.compile(r"EngineCore encountered an issue")

def _scan(text: str):
    cl_total = len(crashloop_re.findall(text))
    wh_fail  = len(warm_fail_re.findall(text))
    return {
        "duels": duel_re.findall(text),
        "verdicts": verdict_re.findall(text),
        "aborts": abort_re.findall(text),
        "dethrones": dethrone_re.findall(text),
        "infra": infra_re.findall(text),
        "timeouts": timeout_re.findall(text),
        "slot_ready": [int(x) for x in slot_ready_re.findall(text)],
        "crashloops": cl_total - wh_fail,    # true POD_CRASHLOOP_BACKOFF only
        "warm_fails": wh_fail,
        "warm_ok": [float(x) for x in warm_ok_re.findall(text)],
        "engine_dead": len(engine_dead_re.findall(text)),
        "prov_timeouts": len(prov_timeout_re.findall(text)),
        "prov_transient": len(prov_trans_re.findall(text)),
        "rentals": len(rent_register_re.findall(text)),
    }

S = _scan(log_text)        # session = current stderr file
L = _scan(all_log_text)    # lifetime = all stderr files

# Dwell throughput (most recent + session avg).
dwell_lines = dwell_re.findall(log_text)
last_dwell = dwell_lines[-1] if dwell_lines else None
sess_sps = sess_tok = sess_lat = None
if dwell_lines:
    sess_sps = sum(float(x[6]) for x in dwell_lines) / len(dwell_lines)
    sess_tok = sum(int(x[7])   for x in dwell_lines) / len(dwell_lines)
    sess_lat = sum(float(x[8]) for x in dwell_lines) / len(dwell_lines)

# Reign durations from successive DETHRONE-block-mappings doesn't survive in the
# log (no block annotation on DETHRONE lines), so derive from verdict reigns.
verdict_reigns = [int(v[4]) for v in L["verdicts"]]   # blocks-of-reign at verdict time
verdict_z      = [(float(v[2]), float(v[3])) for v in L["verdicts"]]  # (z,k)
n_dethrone_v   = sum(1 for z, k in verdict_z if z >  k)
n_hold_v       = sum(1 for z, k in verdict_z if z < -k)
n_skip_v       = len(verdict_z) - n_dethrone_v - n_hold_v

# Lifetime dethrone rate.
def _file_age_h(p: Path) -> float:
    try: return (time.time() - p.stat().st_mtime) / 3600
    except OSError: return 0.0
lifetime_h = (time.time() - logs[0].stat().st_ctime) / 3600 if logs else 0.0

# Per-(model, revision) art coverage from evidence.
art_rows: dict[tuple[str|None,str], int] = Counter()
art_uids: dict[tuple[str|None,str], list[int]] = {}
for r in rows:
    k = (r.k, r.r)
    art_rows[k] += 1
    art_uids.setdefault(k, [])
    if r.m not in art_uids[k]: art_uids[k].append(r.m)

# Per-env counts.
env_counts = Counter(r.e for r in rows)

# Config snapshot (best-effort — uses env vars at run time, not the running validator's).
try: cfg = Config.from_env()
except Exception: cfg = None

# Current k(reign). max_evidence_block lags actual chain head; right after a
# dethrone it can be < reign_start (no rows in the new reign yet) — clamp.
max_evidence_block = max((r.t for r in rows), default=0)
current_block = max(max_evidence_block, int(reign["reign_start"])) if reign else max_evidence_block
reign_blocks_now = (current_block - int(reign["reign_start"])) if reign else 0
k_now = compute_k(reign_blocks_now, cfg.k_init, cfg.k_final, cfg.k_halflife) if cfg else None

# ---------------- output ----------------

ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
pid_s = pids[0] if pids else "NONE"
reign_s = f"uid{reign['uid']}@{reign['reign_start']}b" if reign else "NONE"
status = "OK" if not problems and not warns else ("WARN" if not problems else "FAIL")
print(f"[{ts}] pid={pid_s} rows={len(rows)} reign={reign_s} | {status}")

# Config.
if cfg:
    envs_s = " ".join(f"{e.name}({e.params.get('timeout','?')}s)" for e in cfg.environments)
    print(f"  config: dwell_batch={cfg.dwell_batch} k_init={cfg.k_init} k_final={cfg.k_final} "
          f"halflife={cfg.k_halflife}b σβ={cfg.sigma_beta} σα={cfg.sigma_alpha}")
    print(f"  envs:   {envs_s}")

# Reign.
if reign:
    span = reign_blocks_now
    rows_this_reign = sum(1 for r in rows if r.t >= int(reign["reign_start"]))
    lp = reign.get("last_published_uid")
    print(f"  reign:  uid{reign['uid']}@{reign['revision'][:7]}  start={reign['reign_start']}  "
          f"now={current_block}  span={span}b ({_fmt_blocks(span)})  k(reign)={k_now:.2f}" if k_now is not None
          else f"  reign:  uid{reign['uid']}@{reign['revision'][:7]}  span={span}b")
    print(f"  reign:  rows_this_reign={rows_this_reign}  published_to_chain={lp}")

# Turnover.
print(f"  duels:  {len(S['duels'])} session / {len(L['duels'])} lifetime "
      f"· dethrones {len(S['dethrones'])}/{len(L['dethrones'])} "
      f"· {len(L['dethrones'])/lifetime_h:.2f}/h" if lifetime_h > 0 else
      f"  duels:  {len(S['duels'])} session / {len(L['duels'])} lifetime")

# Verdicts vs aborts breakdown.
abort_reasons_s = Counter(r for r, _ in S["aborts"])
abort_reasons_l = Counter(r for r, _ in L["aborts"])
def _outcome(z: float, k: float) -> str:
    if z >  k: return "dethrone"
    if z < -k: return "hold"
    return "skip"
verdict_outcome_s = Counter(_outcome(float(v[2]), float(v[3])) for v in S["verdicts"])
print(f"  verdicts: {len(S['verdicts'])}/{len(L['verdicts'])} "
      f"(session: {dict(verdict_outcome_s) or '—'})  "
      f"lifetime: dethrone={n_dethrone_v} hold={n_hold_v} skip={n_skip_v}")
if abort_reasons_l:
    top = ", ".join(f"{r}={n}" for r, n in abort_reasons_l.most_common(6))
    print(f"  aborts: {sum(abort_reasons_s.values())}/{sum(abort_reasons_l.values())} "
          f"by reason (lifetime): {top}")
denom_l = len(L['duels'])
if denom_l: print(f"  abort rate: {_pct(sum(abort_reasons_s.values()), len(S['duels']))} session "
                  f"/ {_pct(sum(abort_reasons_l.values()), denom_l)} lifetime")

# Recent verdicts (with timestamps).
v_with_ts = list(re.finditer(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})[,\d]* affine\.loop INFO verdict: "
    r"Δθ̂=([+-][\d.]+)±([\d.]+) z=([+-][\d.]+) k=([\d.]+) reign=(\d+)b",
    log_text))
if v_with_ts:
    print(f"  recent verdicts (last {min(RECENT_VERDICTS, len(v_with_ts))}):")
    for m in v_with_ts[-RECENT_VERDICTS:]:
        d, dt, se, z, k, rb = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5), m.group(6)
        decision = "DETHRONE" if float(z) > float(k) else ("hold" if float(z) < -float(k) else "?")
        print(f"    {d[11:]}  z={z}  Δθ̂={dt}±{se}  k={k}  reign={rb}b  → {decision}")

# Throughput.
if last_dwell:
    p, inf, cap, smp, tok, el, sps, tps, lat = last_dwell
    print(f"  current: {sps} sample/s · {tps} tok/s · mean_lat={lat}s · in_flight={inf}/{cap} "
          f"(pairs={p} samples={smp} tokens={tok} elapsed={el}s)")
if sess_sps is not None:
    print(f"  session avg: {sess_sps:.2f} sample/s · {sess_tok:.0f} tok/s · mean_lat={sess_lat:.1f}s "
          f"({len(dwell_lines)} dwell-log lines)")

# Sampler health.
infra_types_l = Counter(t[2] for t in L["infra"])
infra_types_s = Counter(t[2] for t in S["infra"])
print(f"  sampler: infra_fail {len(S['infra'])}/{len(L['infra'])} · "
      f"timeouts {len(S['timeouts'])}/{len(L['timeouts'])}")
if infra_types_l:
    top = ", ".join(f"{k}={n}" for k, n in infra_types_l.most_common(6))
    print(f"  sampler: infra by error_type (lifetime): {top}")

# Synth (sampler health) and lost-pair fraction (what the fit ignores).
if rows:
    synth = sum(1 for r in rows if r.l == 0.0)
    print(f"  synth rows: {synth}/{len(rows)} ({100*synth/len(rows):.1f}%) "
          f"· lost pairs: {lost_pairs}/{total_pairs} ({100*lost_pairs/total_pairs:.1f}%)")

# Provisioning.
if L["slot_ready"]:
    sr = sorted(L["slot_ready"])
    p50 = sr[len(sr)//2]; p90 = sr[int(0.9*len(sr))]
    print(f"  provision: rentals_registered={L['rentals']} slot_ready_n={len(sr)} "
          f"avg={sum(sr)/len(sr):.0f}s p50={p50}s p90={p90}s")
if L["warm_ok"]:
    wh = sorted(L["warm_ok"])
    p50w = wh[len(wh)//2]; p90w = wh[int(0.9*len(wh))]
    print(f"  warm hit:  ok={len(wh)} fail={L['warm_fails']} "
          f"avg={sum(wh)/len(wh):.1f}s p50={p50w:.1f}s p90={p90w:.1f}s")
elif L["warm_fails"]:
    print(f"  warm hit:  ok=0 fail={L['warm_fails']}")
print(f"  provision: crashloop {S['crashloops']}/{L['crashloops']} · "
      f"timeout {S['prov_timeouts']}/{L['prov_timeouts']} · "
      f"transient {S['prov_transient']}/{L['prov_transient']}")
# EngineCore deaths: HTTP 500s with the vLLM catch-all message. Tracks whether
# the cudagraph-profile env var + warm-hit + 0.85 mem-util fix is doing its job.
print(f"  vllm: EngineCore-dead 500s {S['engine_dead']}/{L['engine_dead']}")

# Evidence coverage.
if env_counts:
    top_envs = " · ".join(f"{e}={n}" for e, n in env_counts.most_common(8))
    print(f"  envs ({len(env_counts)}): {top_envs}")
print(f"  miners: {len(miners)} unique uid · {len(art_rows)} (model,rev) artifacts")

# Full leaderboard.
if fit is not None and not fit.degenerate:
    by_uid_rev = {(m.uid, m.revision): m for m in miners}
    keys = []
    seen = set()
    for m in miners:
        k = (m.model, m.revision)
        if k not in seen: keys.append(k); seen.add(k)
    for r in rows:
        k = (r.k, r.r) if r.k is not None else (
            (by_uid_rev[(r.m, r.r)].model, r.r) if (r.m, r.r) in by_uid_rev
            else (f"?ghost:{r.m}", r.r))
        if k not in seen: keys.append(k); seen.add(k)
    se = fit.theta_se
    order = sorted(range(len(keys)), key=lambda i: -fit.theta[i])
    print(f"  leaderboard ({len(order)} artifacts):")
    for i in order:
        model, rev = keys[i]
        k_evi = (model, rev) if model is not None else (None, rev)
        uids = art_uids.get(k_evi, art_uids.get((model, rev), []))
        n = art_rows.get(k_evi, art_rows.get((model, rev), 0))
        crown = " 👑" if reign and reign["uid"] in uids else ""
        # Crop model for tidy column.
        mtxt = (model or "?")
        if len(mtxt) > 50: mtxt = mtxt[:47] + "..."
        print(f"    θ̂={fit.theta[i]:+.3f}±{se[i]:.3f}  rows={n:>4}  {mtxt}@{rev[:7]}  "
              f"uids={uids}{crown}")

for w in warns: print(f"  WARN: {w}")
for p in problems: print(f"  FAIL: {p}")
sys.exit(1 if problems else 0)
