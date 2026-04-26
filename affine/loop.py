"""Pairwise king-of-the-hill validator loop backed by a global 2PL IRT fit.

The champion reigns until a challenger statistically proves superiority.

Each outer iteration:

  1. Read miners. If the champion is unknown or no longer registered, elect
     argmax(θ̂) from the global fit and publish them.
  2. Pick the lowest-block challenger not yet attempted this reign.
  3. Provision king and challenger on two slots in parallel.
  4. Dwell: for cfg.dwell env picks, choose the env maximizing Fisher info
     for the contrast (θ_chal − θ_king), sample both models concurrently,
     append two rows, refit. The posterior sharpens as evidence accrues.
  5. Teardown. Contrast z = (θ̂_chal − θ̂_king) / √Var against a ratcheting
     threshold k(reign). If z > k, challenger dethrones; reset the reign
     block; publish. Else champion holds; try the next challenger.

Evidence rows are the durable unit of contribution. Validators contribute
independent rows (validator-private seeds); a single global IRT fit over all
rows produces θ̂ and its Laplace covariance, which drives both env selection
and the dethronement test.
"""

from __future__ import annotations
import asyncio
import hashlib
import json
import logging
import os
import secrets
import signal
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Awaitable, Callable

import httpx
import numpy as np
import affinetes as af

from .audit import audit
from .chain import Miner, Subtensor, _tiebreak, _truthy_env, get_miners, set_weights
from .config import BASELINE_MODELS, Config
from .evidence import EvidenceStore, Row
from .irt import Fit, Priors, compute_k, fisher_env, fit_2pl
from .sampler import run_one
from .vllm import Slot, SlotProvisionFailed, TargonSlots, health_ping, inference_ping

log = logging.getLogger(__name__)

ENV_FAIL_QUARANTINE = 3   # consecutive infra fails on one env → stop picking it this dwell


@dataclass
class Chain:
    """Everything the loop needs from the world besides slots, envs, and evidence."""
    hotkey: str
    list_miners: Callable[[], Awaitable[list[Miner]]]
    current_block: Callable[[], Awaitable[int]]
    publish_winner: Callable[[int, str], Awaitable[bool]]


class Skiplist:
    """Two-tier skiplist over (model, revision) pairs.

    durable: crashloops — artifact itself is broken, penalty survives restart.
    session: provision timeouts — could be miner or Targon infra; prevent retry
             this session but don't persist (a restart after a Targon incident
             should recover, not carry the false positive forward).
    """
    def __init__(self, models: set[str] = frozenset(), path: str | Path | None = None):
        self._models = set(models)
        self._session: set[tuple[str, str]] = set()
        self._durable: set[tuple[str, str]] = set()
        self.path = Path(path) if path else None
        if self.path and self.path.exists():
            for line in self.path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    self._durable.add((d["model"], d["revision"]))
                except (json.JSONDecodeError, KeyError) as e:
                    log.warning(f"skiplist: skipping malformed line: {e}")
            if self._durable:
                log.info(f"skiplist: loaded {len(self._durable)} durable entries from {self.path}")

    def add(self, model: str, revision: str, *, durable: bool = True) -> None:
        key = (model, revision)
        tier = "persist" if durable else "session"
        if durable:
            if key in self._durable:
                return
            # Write disk first, mutate memory only on success — otherwise a failed
            # write leaves in-memory marked as skipped while disk knows nothing,
            # so a restart silently re-admits the artifact. Single os.write +
            # fsync: a SIGKILL between buffered write and OS flush would lose the
            # mark and re-admit a known-crashloop model on restart.
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                payload = (json.dumps({"model": model, "revision": revision}) + "\n").encode()
                fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
                try:
                    n = os.write(fd, payload)
                    if n != len(payload):
                        raise OSError(f"short write: {n}/{len(payload)} bytes")
                    os.fsync(fd)
                finally:
                    os.close(fd)
            self._durable.add(key)
        else:
            if key in self._session or key in self._durable:
                return
            self._session.add(key)
        log.info(f"skiplist ({tier}): {model}@{revision}")

    def __contains__(self, m: Miner) -> bool:
        key = (m.model, m.revision)
        return m.model in self._models or key in self._session or key in self._durable

    def filter(self, miners: list[Miner]) -> list[Miner]:
        return [m for m in miners if m not in self]


def _respondents(miners: list[Miner], rows: list[Row]) -> list[tuple[int, str]]:
    """Registered miners first; historical ghosts at the tail so their evidence
    still informs env parameters even after a re-commit replaces them."""
    keys = [(m.uid, m.revision) for m in miners]
    seen = set(keys)
    for r in rows:
        k = (r.m, r.r)
        if k not in seen:
            keys.append(k); seen.add(k)
    return keys


def _fit(rows: list[Row], miners: list[Miner], env_names: list[str],
         priors: Priors) -> Fit:
    """Drops envs whose every observed outcome is identical (all-pass or all-fail)
    from the fit data — they tighten the contrast SE based on Hessian mass that
    has no per-miner discrimination. Filtered envs keep their slot in the returned
    Fit at the prior (β=α=0).

    Acquisition (fisher_env) is *not* told to mask these — degeneracy is the
    posterior's job to express via wide marginals. Masking them at acquisition
    time is what triggers "all envs quarantined at i=0" when an early run of
    coincidental pass-pass on every env temporarily makes every env all-same:
    we'd then refuse to ever sample those envs again, denying ourselves the one
    fail that would un-degenerate them."""
    keys = _respondents(miners, rows)
    k2i = {k: i for i, k in enumerate(keys)}
    e2i = {n: i for i, n in enumerate(env_names)}
    outcomes: dict[str, set[int]] = {}
    for r in rows:
        if r.e in e2i:
            outcomes.setdefault(r.e, set()).add(r.p)
    drop = {n for n, outs in outcomes.items() if len(outs) < 2}
    filtered = [r for r in rows if r.e in e2i and r.e not in drop]
    m_idx = np.fromiter((k2i[(r.m, r.r)] for r in filtered), dtype=np.intp, count=len(filtered))
    e_idx = np.fromiter((e2i[r.e] for r in filtered), dtype=np.intp, count=len(filtered))
    y = np.fromiter((r.p for r in filtered), dtype=np.float64, count=len(filtered))
    return fit_2pl(m_idx, e_idx, y, len(keys), len(env_names), priors)


def _index(miners: list[Miner], uid: int, revision: str) -> int:
    for i, m in enumerate(miners):
        if m.uid == uid and m.revision == revision:
            return i
    raise KeyError(f"uid={uid} rev={revision} not in miners")


def _rows_per_env(rows: list[Row], challenger_uid: int, king_uid: int) -> dict[str, tuple[int, int]]:
    """(challenger_pass_count, king_pass_count) per env over this duel's rows."""
    if challenger_uid == king_uid:
        raise ValueError(f"challenger_uid == king_uid == {king_uid}")
    out: dict[str, list[int]] = {}
    for r in rows:
        if r.m not in (challenger_uid, king_uid):
            continue
        slot = out.setdefault(r.e, [0, 0])
        slot[0 if r.m == challenger_uid else 1] += int(r.p)
    return {e: (c, k) for e, (c, k) in out.items()}


async def _load_envs(cfg: Config) -> dict[str, tuple]:
    # Dedupe by image: load each unique image once, share across envs that use it.
    # params are call-time (passed in /call body), so they may differ; env_vars and
    # mem_limit are container-init, so they MUST match — silently sharing a wrapper
    # under conflicting container config would deploy one env's settings to both.
    loaded: dict[str, tuple] = {}
    out = {}
    try:
        for spec in cfg.environments:
            if spec.image in loaded:
                wrapper, shared_spec = loaded[spec.image]
                if spec.env_vars != shared_spec.env_vars or spec.mem_limit != shared_spec.mem_limit:
                    raise ValueError(
                        f"env '{spec.name}' shares image {spec.image} with '{shared_spec.name}' "
                        f"but env_vars/mem_limit differ — split into distinct images or align config"
                    )
                merged_params = {**shared_spec.params, **spec.params}
                out[spec.name] = (wrapper, replace(spec, params=merged_params))
                log.info(f"env: {spec.name} ({spec.image}) [shared]")
                continue
            env_vars = dict(spec.env_vars)
            for k in ("CHUTES_API_KEY", "HF_TOKEN"):
                if (v := os.environ.get(k)):
                    env_vars.setdefault(k, v)
            env_vars.setdefault("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", "vllm"))
            # pull=False: rely on cached images (significant startup speedup)
            wrapper = af.load_env(image=spec.image, mode="docker", env_vars=env_vars or None,
                                  mem_limit=spec.mem_limit, pull=False, cleanup=True)
            backend = getattr(wrapper, "_backend", None)
            if backend is not None and hasattr(backend, "_check_container_restart"):
                backend._check_container_restart = lambda: False
            loaded[spec.image] = (wrapper, spec)
            out[spec.name] = (wrapper, spec)
            log.info(f"env: {spec.name} ({spec.image})")
        return out
    except BaseException:
        for wrapper, _ in loaded.values():
            try: await wrapper.cleanup()
            except Exception: pass
        raise


async def _cancellable(coro, stop: asyncio.Event, on_orphan=None):
    """Await `coro` unless `stop` fires, in which case cancel and raise CancelledError.

    Used to make long-running provisions interruptible by SIGTERM/SIGINT without
    threading the stop event through every slot backend. Worker is cancelled and
    awaited in the finally so cleanup (provision's BaseException handler) runs
    even when our outer task is cancelled — without this, an outer cancel between
    `wait` returning and the post-finally cancel would orphan the worker, leaving
    a Targon rental in flight.

    `on_orphan(result)`: optional async cleanup for the case where the worker
    produced a result that the caller never received (outer cancel raced with
    worker completion). Without this, an outer task cancellation that lands in
    the same scheduling tick as `slots.provision(...)` returning a Slot would
    leak the rental — the worker's cleanup didn't fire (it succeeded), and the
    caller's didn't either (it never saw the slot).
    """
    worker = asyncio.ensure_future(coro)
    stopper = asyncio.ensure_future(stop.wait())
    delivered = False
    try:
        done, _ = await asyncio.wait({worker, stopper}, return_when=asyncio.FIRST_COMPLETED)
        # Prefer the worker if both finished in the same batch — losing a
        # successful Slot to a same-tick stop fires `_provision`'s cleanup but
        # not the caller's, leaking the rental.
        if worker in done:
            delivered = True
            return worker.result()
        raise asyncio.CancelledError("stop event set")
    finally:
        stopper.cancel()
        if not worker.done():
            worker.cancel()
            # 60s covers Targon delete's 30s budget plus probe timeouts; under
            # this we'd drop the reference mid-cleanup and leak the rental.
            try: await asyncio.wait_for(asyncio.shield(worker), timeout=60)
            except asyncio.TimeoutError:
                # Worker exceeded its cleanup budget — it's still pending, we're
                # losing the reference. Reconcile() at next startup is the backstop.
                log.warning("_cancellable: worker cleanup exceeded 60s; rental may leak until reconcile")
            except BaseException: pass
        if not delivered and worker.done() and not worker.cancelled() and worker.exception() is None:
            orphan = worker.result()
            if on_orphan is not None and orphan is not None:
                try: await asyncio.shield(on_orphan(orphan))
                except BaseException as e:
                    log.warning(f"_cancellable: on_orphan failed: {type(e).__name__}: {e}")


async def _safe_teardown(slots, slot: Slot, ctx: str = "") -> None:
    try: await slots.teardown(slot)
    except Exception as e: log.warning(f"teardown error{f' ({ctx})' if ctx else ''}: {e}")


async def _provision(slots, miner: Miner, stop: asyncio.Event) -> tuple[Slot | None, str]:
    """Provision and probe a slot. Returns (slot|None, status). Status is one of:
    "ok" (slot returned), "crashloop" (artifact-level fault — caller should durable-skip),
    "timeout" (could be Targon infra — caller may session-skip non-king miners),
    "unhealthy" (provisioned but failed health/inference probes), "transient"
    (network/protocol blip — no skip, retry next iter), "error" (unclassified;
    treat as miner-side artifact bug). Slot-bearing exceptions during the probe
    phase tear down the slot before re-raising."""
    try:
        slot = await _cancellable(
            slots.provision(miner.model, miner.revision), stop,
            on_orphan=lambda s: _safe_teardown(slots, s, "provision-orphan"),
        )
    except SlotProvisionFailed as e:
        log.warning(f"provision crashloop {miner.model}@{miner.revision}: {e}")
        return None, "crashloop"
    except TimeoutError as e:
        log.warning(f"provision timeout uid {miner.uid} {miner.model}@{miner.revision}: {e}")
        return None, "timeout"
    except asyncio.CancelledError:
        raise
    except httpx.HTTPError as e:
        # Targon API blips (ConnectError, ReadTimeout, 5xx). Session-skipping
        # on these would empty the queue during a 30s Targon outage and stall
        # the validator until queue refresh. Retry next iter instead.
        log.warning(f"provision transient uid {miner.uid}: {type(e).__name__}: {e}")
        return None, "transient"
    except Exception as e:
        log.error(f"provision error uid {miner.uid}: {e}")
        return None, "error"
    try:
        if not await _cancellable(health_ping(slot.base_url), stop):
            log.warning(f"slot unhealthy post-provision uid {miner.uid}: {slot.base_url}")
            await _safe_teardown(slots, slot, "unhealthy")
            return None, "unhealthy"
        if not await _cancellable(inference_ping(slot.base_url, miner.model), stop):
            log.warning(f"slot inference probe failed uid {miner.uid}: {slot.base_url}")
            await _safe_teardown(slots, slot, "unhealthy")
            return None, "unhealthy"
    except BaseException:
        await _safe_teardown(slots, slot, "probe-aborted")
        raise
    return slot, "ok"


def _apply_skip(skip: Skiplist, miner: Miner, status: str, *,
                is_king: bool, king_artifact: tuple[str, str] | None = None) -> None:
    """Translate a provision status into skiplist policy.

    The current king is never session-skipped on a transient timeout — that would
    force a re-election on Targon infra hiccups and burn weight quota.

    Challenger artifacts that match a *proven-healthy cached* king's are exempt:
    a popular open model deployed by multiple miners means a session/durable skip
    on chal would filter king out on the next iter, costing us the cached slot.
    `king_artifact` should be passed ONLY when the king slot is currently cached
    and healthy. During concurrent first-time provisioning (no cached king), this
    is None — otherwise a crashloop on a shared artifact gets suppressed on both
    sides (king's task cancelled by fail-fast, challenger exempted) and the loop
    retries the same pair indefinitely."""
    if not is_king and king_artifact and (miner.model, miner.revision) == king_artifact:
        return
    if status == "crashloop":
        skip.add(miner.model, miner.revision, durable=True)
    elif status in ("timeout", "unhealthy", "error") and not is_king:
        # "unhealthy" = post-provision probe failed after retries (3x w/ backoff
        # already inside inference_ping). Strong signal the artifact is broken
        # in a way that's not crashloop-detectable. Session-skip avoids tight
        # retry loops; durable-skip is too aggressive since probes can transient.
        # "error" covers unclassified provision exceptions (e.g. rental register
        # returned no uid). Without session-skipping, a chal that consistently
        # raises a non-classified exception would be retried forever — the king
        # task gets cancelled by fail-fast, so chal's skip is the only escape.
        skip.add(miner.model, miner.revision, durable=False)


async def _provision_pair(slots, king: Miner, chal: Miner, skip: Skiplist,
                          stop: asyncio.Event) -> tuple[Slot | None, Slot | None, bool]:
    """Provision king and challenger concurrently with fail-fast cancellation.

    Returns `(king_slot, chal_slot, king_attempt_failed)`. `king_attempt_failed`
    is True iff king's provision ran to completion with a non-ok status; False
    if king was cancelled because chal failed first. The caller uses this to
    distinguish king-fail backoff from "advance to next challenger": without
    the bool, a chal with a persistent provision error trips the king-fail
    backoff path which `attempted.discard`s the chal — and the next iteration
    re-picks the same broken chal, looping forever.

    If the first finisher fails to produce a slot, the duel is dead — cancel the
    sibling immediately rather than wait out a 15-min Targon provision we'll
    discard. On outer cancellation (SIGTERM, parent task) tear down any slot
    that did complete; gather() otherwise drops survivors and leaks rentals.
    """
    t_k = asyncio.create_task(_provision(slots, king, stop))
    t_c = asyncio.create_task(_provision(slots, chal, stop))
    pending = {t_k, t_c}

    async def _drain_and_teardown(reason: str) -> None:
        for t in (t_k, t_c):
            if not t.done(): t.cancel()
        for t in (t_k, t_c):
            try: await asyncio.shield(t)
            except BaseException: pass
            if t.done() and not t.cancelled() and t.exception() is None:
                slot, _ = t.result()
                if slot is not None:
                    await asyncio.shield(_safe_teardown(slots, slot, reason))

    fail_fast = False
    try:
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                if t.cancelled() or t.exception() is not None: continue
                slot, _ = t.result()
                if slot is None and pending:
                    fail_fast = True
                    for p in pending: p.cancel()
    except asyncio.CancelledError:
        await asyncio.shield(_drain_and_teardown("outer-cancelled"))
        raise

    pairs = [(t_k, king, True), (t_c, chal, False)]
    slots_got: list[Slot | None] = []
    outer_cancelled = False
    # Collect slots first; only then run skiplist writes that can raise. Any
    # exception (e.g. disk I/O while persisting a durable skip) must still tear
    # down whatever provisioned, otherwise the rental leaks.
    try:
        for t, m, is_king in pairs:
            if t.cancelled() or isinstance(t.exception(), asyncio.CancelledError):
                if not fail_fast: outer_cancelled = True
                slots_got.append(None); continue
            exc = t.exception()
            if exc is not None:
                log.error(f"provision-pair unexpected exception uid {m.uid}: {exc}")
                slots_got.append(None); continue
            slot, _ = t.result()
            slots_got.append(slot)
        # King's status is known once both tasks have completed; if king is "ok"
        # by then it IS "proven healthy" for this iteration, so the chal exemption
        # applies. The earlier policy of unconditionally passing king_artifact=None
        # session-skipped (model, rev) on every chal failure, which on the next
        # filter pass also removed the (now-cached) king and forced a re-elect.
        king_status = None
        if not (t_k.cancelled() or t_k.exception() is not None):
            _, king_status = t_k.result()
        king_artifact = (king.model, king.revision) if king_status == "ok" else None
        for (t, m, is_king), slot in zip(pairs, slots_got):
            if t.cancelled() or t.exception() is not None:
                continue
            _, status = t.result()
            _apply_skip(skip, m, status, is_king=is_king, king_artifact=king_artifact)
    except BaseException:
        for s in slots_got:
            if s is not None: await asyncio.shield(_safe_teardown(slots, s, "pair-error"))
        raise

    if outer_cancelled:
        for s in slots_got:
            if s is not None: await _safe_teardown(slots, s, "pair-cancelled")
        raise asyncio.CancelledError()
    # king_attempt_failed: king's task ran to completion AND failed to produce a
    # slot. False when king was cancelled (fail_fast by chal, or outer cancel).
    # `Task.exception()` raises on cancelled tasks — guard with cancelled() first.
    if t_k.cancelled():
        king_attempt_failed = False
    elif t_k.exception() is not None:
        king_attempt_failed = True   # unexpected exception leaking out of _provision
    else:
        king_attempt_failed = slots_got[0] is None
    if fail_fast:
        for s in slots_got:
            if s is not None: await _safe_teardown(slots, s, "pair-fail-fast")
        return None, None, king_attempt_failed
    king_slot, chal_slot = slots_got
    # If only the chal failed, retain the king slot — it's the long-lived 600GB
    # download we cache across duels. Tearing it down on every chal-only failure
    # would force a fresh download per challenger when a churn of bad chals arrives.
    # Caller's `if chal_slot is None: continue` handles the chal absence.
    if king_slot is None and chal_slot is not None:
        await _safe_teardown(slots, chal_slot, "king-failed-chal-orphan")
        return None, None, king_attempt_failed
    return king_slot, chal_slot, king_attempt_failed


def _seed(uid: int, rev: str, env: str, c: int, salt: str = "") -> int:
    """31 bits (signed int32 max). vLLM accepts ≤ 2^63-1, but game's openspiel env
    feeds the seed into np.random.RandomState which only accepts [0, 2^32-1] and
    internally computes seed+1..seed+100, so we mask to 2^31-1 for portability.
    `salt` is mixed in so the seed depends on the validator's hotkey rather than
    on public protocol state alone — hotkey is on-chain so this isn't a fully
    private salt, but it forces an adversary to enumerate per-validator instead
    of computing one universal precomputed answer set."""
    h = hashlib.sha256(f"{salt}\0{uid}\0{rev}\0{env}\0{c}".encode()).digest()
    return int.from_bytes(h[:4], "big") & ((1 << 31) - 1)


def _task_id(king_uid: int, chal_uid: int, env: str, iter_idx: int,
             lo: int, hi: int, salt: str = "") -> int:
    """Per-iteration task_id, shared by king and challenger so the verdict is
    matched-task. Order-invariant in (king, chal) so reversing the duel picks the
    same sequence. Uniform over [lo, hi]. `salt` mixes in the validator hotkey for
    the same reason as _seed — see that docstring."""
    a, b = (king_uid, chal_uid) if king_uid <= chal_uid else (chal_uid, king_uid)
    h = hashlib.sha256(f"{salt}\0{a}\0{b}\0{env}\0{iter_idx}".encode()).digest()
    n = hi - lo + 1
    return lo + (int.from_bytes(h[:8], "big") % n)


async def _sample(chain: Chain, wrapper, params: dict, timeout: float, slot: Slot,
                  miner: Miner, env_name: str, store: EvidenceStore,
                  task_id: int, block: int) -> Row | None:
    """One evaluation of `miner` in `env_name` on `slot`. Returns a row on success,
    None on infra failure. Counter/seed state lives in `store`. `block` is supplied
    by the caller — reading current_block per sample puts the chain RPC in the
    sample's exception path, where any blip would be misattributed as an env
    failure (env_fails counter triggers spurious quarantine)."""
    c = store.next_counter(miner.uid, miner.revision, env_name)
    outcome, latency = await run_one(wrapper, params, timeout, slot,
                                     seed=_seed(miner.uid, miner.revision, env_name, c, salt=chain.hotkey),
                                     task_id=task_id)
    if outcome is None:
        return None
    return Row(m=miner.uid, r=miner.revision, e=env_name, c=c,
               p=int(outcome), t=int(block), l=float(latency), i=int(task_id))


SIDE_FAIL_THRESHOLD = 3   # consecutive single-side Nones across diverse envs → side broken


async def _dwell(chain: Chain, king: Miner, king_slot: Slot, challenger: Miner,
                 chal_slot: Slot, miners: list[Miner], rows: list[Row],
                 envs, env_names, store: EvidenceStore, cfg: Config, priors: Priors,
                 rng: np.random.Generator, stop: asyncio.Event
                 ) -> tuple[list[Row], Fit, str | None]:
    """Fisher-best env → sample king & challenger concurrently → append rows → refit.

    Returns (rows, fit, abort_reason). `abort_reason` ∈ {None, "chal_broken",
    "king_broken", "envs_quarantined"} signals to the caller why we stopped early.

    Three independent streaks track failure attribution:
      env_fails[e]: BOTH sides None on env e → env-side issue; quarantine env after
        ENV_FAIL_QUARANTINE consecutive same-env fails. If all envs quarantine, abort.
      chal_streak: chal None / king ok across envs → chal-side artifact issue (broken
        endpoint, model OOM). After SIDE_FAIL_THRESHOLD aborts with "chal_broken" so
        the caller can session-skip the chal artifact and advance.
      king_streak: king None / chal ok across envs → king-side issue (memory leak,
        host-level fault that 1-token health probes don't detect). Aborts with
        "king_broken"; caller drops the king slot. Re-provision next iter — Targon
        may give a different physical host that doesn't have the leak.

    Mixing all three into env_fails (the prior bug) made a chal-only-broken miner
    eventually quarantine every env and abort with no rows, while a king-only-broken
    king reigned forever because no signal differentiated it from env failures."""
    fit = _fit(rows, miners, env_names, priors)
    k_idx = _index(miners, king.uid, king.revision)
    c_idx = _index(miners, challenger.uid, challenger.revision)
    env_fails = [0] * len(env_names)
    chal_streak = 0
    king_streak = 0
    for i in range(cfg.dwell):
        if stop.is_set():
            return rows, fit, None
        excluded = frozenset(j for j, n in enumerate(env_fails) if n >= ENV_FAIL_QUARANTINE)
        if len(excluded) >= len(env_names):
            log.warning(f"all envs quarantined at i={i}; aborting dwell (king={king.uid} chal={challenger.uid})")
            return rows, fit, "envs_quarantined"
        e = fisher_env(fit, c_idx, k_idx, rng, excluded=excluded)
        env_name = env_names[e]
        wrapper, spec = envs[env_name]
        params = {k: v for k, v in spec.params.items() if k != "timeout"}
        timeout = float(spec.params.get("timeout", 600))
        lo, hi = spec.task_range
        task_id = _task_id(king.uid, challenger.uid, env_name, i, lo, hi, salt=chain.hotkey)
        # Read block once per dwell iter, share across king+chal samples. Per-sample
        # reads put the chain RPC inside the sample's exception path: a transient
        # subtensor blip would surface as a sample exception, which env_fails counts
        # toward quarantine — wrong attribution. RPC failures here fall back to 0
        # so the duel proceeds on best-effort metadata; t is record-keeping only.
        try: block = await chain.current_block()
        except Exception as ex:
            log.warning(f"current_block failed at dwell i={i}: {ex}; using 0")
            block = 0

        # Explicit tasks + shielded drain: if one sample raises, the sibling is
        # cancelled AND awaited before we return. asyncio.gather lets the loser
        # keep running, which leaks inference onto a slot we may be tearing down.
        t_k = asyncio.create_task(_sample(chain, wrapper, params, timeout, king_slot, king, env_name, store, task_id, block))
        t_c = asyncio.create_task(_sample(chain, wrapper, params, timeout, chal_slot, challenger, env_name, store, task_id, block))
        try:
            results = await _cancellable(asyncio.gather(t_k, t_c, return_exceptions=True), stop)
        except asyncio.CancelledError:
            for t in (t_k, t_c):
                if not t.done(): t.cancel()
            for t in (t_k, t_c):
                try: await asyncio.shield(t)
                except BaseException: pass
            return rows, fit, None
        if any(isinstance(r, BaseException) for r in results):
            for r in results:
                if isinstance(r, BaseException) and not isinstance(r, asyncio.CancelledError):
                    log.warning(f"_dwell sample raised on env={env_name}: {type(r).__name__}: {r}")
            # Sample exception: ambiguous evidence (env-side plumbing failure can't
            # attribute to a miner). Track as env-side; don't feed side streaks.
            env_fails[e] += 1
            chal_streak = king_streak = 0
            continue
        k_row, c_row = results
        if k_row is None and c_row is None:
            # Ambiguous: env failed; don't accumulate side-specific streak signal.
            env_fails[e] += 1
            chal_streak = king_streak = 0
            if env_fails[e] == ENV_FAIL_QUARANTINE:
                log.warning(f"quarantine env '{env_name}' after {env_fails[e]} consecutive fails (king={king.uid} chal={challenger.uid})")
            continue
        # Single-side None below: env produced a valid result for at least one side,
        # so env_fails resets — env is fine, the issue is side-attributable.
        if k_row is None:
            king_streak += 1; chal_streak = 0; env_fails[e] = 0
            if king_streak >= SIDE_FAIL_THRESHOLD:
                log.warning(f"king uid{king.uid} appears broken: {king_streak} consecutive king-only Nones across envs; aborting dwell")
                return rows, fit, "king_broken"
            continue
        if c_row is None:
            chal_streak += 1; king_streak = 0; env_fails[e] = 0
            if chal_streak >= SIDE_FAIL_THRESHOLD:
                log.warning(f"chal uid{challenger.uid} appears broken: {chal_streak} consecutive chal-only Nones across envs; aborting dwell")
                return rows, fit, "chal_broken"
            continue
        chal_streak = king_streak = 0
        env_fails[e] = 0

        store.append_pair(k_row, c_row)
        rows.extend((k_row, c_row))
        fit = _fit(rows, miners, env_names, priors)
    return rows, fit, None


async def _elect(rows: list[Row], miners: list[Miner], env_names: list[str],
                 priors: Priors) -> tuple[int, str]:
    """Pick champion. Cold start (no evidence): seat a hardcoded baseline model if
    registered; otherwise pick the lowest-block miner (fairest tiebreaker — first
    to commit holds the throne until evidence dethrones them). With evidence:
    argmax(θ̂)."""
    if not rows:
        for target in BASELINE_MODELS:
            for m in miners:
                if m.model == target:
                    log.info(f"cold start: seating baseline {m.model} as uid {m.uid}")
                    return m.uid, m.revision
        # No baseline registered and no rows: argmax(θ̂) on a prior-only fit
        # returns uid 0 by tie-break, which is misleading. Pick lowest-block
        # explicitly so the choice is interpretable (and the loop's normal
        # queue-by-block logic dominates from there).
        m = min(miners, key=lambda x: (x.block, _tiebreak(x)))
        log.info(f"cold start: no baseline registered; seating lowest-block uid {m.uid} model {m.model}")
        return m.uid, m.revision
    fit = _fit(rows, miners, env_names, priors)
    n = len(miners)
    i = int(np.argmax(fit.theta[:n]))
    log.info(f"elect: uid {miners[i].uid} (θ̂={fit.theta[i]:+.3f})")
    return miners[i].uid, miners[i].revision


def _load_reign_state(path: Path) -> tuple[tuple[int, str] | None, int, int | None]:
    """Read persisted (champion, reign_start, last_published_uid). Without this, a
    restart with the same champion still on chain re-elects them and resets
    reign_start to the current block — undoing the k(reign) decay and forcing the
    threshold back to k_init. last_published_uid is restored separately so a
    restart mid-reign doesn't re-submit set_weights for the same uid (rate-limited
    on chain, burns weight quota)."""
    if not path.exists():
        return None, 0, None
    try:
        d = json.loads(path.read_text())
        uid = int(d["uid"]); rev = str(d["revision"]); rs = int(d["reign_start"])
        lp = d.get("last_published_uid")
        return (uid, rev), rs, (int(lp) if lp is not None else None)
    except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        log.warning(f"reign state at {path} unreadable, ignoring: {e}")
        return None, 0, None


def _save_reign_state(path: Path, champion: tuple[int, str], reign_start: int,
                       last_published_uid: int | None = None) -> None:
    # fsync the tmp before rename: without it, power loss between rename commit
    # and the data flush can leave an empty/partial file. _load_reign_state would
    # then return None and the loop re-elects, resetting reign_start and undoing
    # all k(reign) decay that had accumulated.
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps({
        "uid": int(champion[0]), "revision": str(champion[1]),
        "reign_start": int(reign_start),
        "last_published_uid": (int(last_published_uid) if last_published_uid is not None else None),
    }).encode()
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        n = os.write(fd, payload)
        if n != len(payload):
            raise OSError(f"short write: {n}/{len(payload)} bytes")
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)


async def run(cfg: Config, chain: Chain, slots=None):
    slots = slots or TargonSlots(cfg, hotkey=chain.hotkey)
    if hasattr(slots, "reconcile"):
        await slots.reconcile()
    envs = await _load_envs(cfg)
    env_names = [spec.name for spec in cfg.environments]
    store = EvidenceStore(cfg.evidence_path)
    reign_state_path = Path(cfg.evidence_path).parent / f"reign-{chain.hotkey[:8]}.json"
    skiplist_path = Path(cfg.evidence_path).parent / f"skiplist-{chain.hotkey[:8]}.jsonl"
    skip = Skiplist(
        {s.strip() for s in os.getenv("AFFINE_MODEL_SKIPLIST", "").split(",") if s.strip()},
        path=skiplist_path,
    )
    priors = Priors(cfg.sigma_beta, cfg.sigma_alpha)
    # Mix hotkey with OS entropy so fisher_env's Thompson stream isn't replayed
    # across restarts. Hotkey-only seeding makes a tight crashloop deterministically
    # repick the same env that triggered the crash; entropy breaks the cycle and
    # gives fresh exploration on each restart. Determinism we DO need (per-sample
    # task seeds, counter monotonicity) is anchored elsewhere — in evidence rows
    # and SHA(uid||rev||env||counter) — and is unaffected by rng-state choice.
    hotkey_int = int.from_bytes(chain.hotkey.encode()[:8].ljust(8, b"\0"), "big")
    rng = np.random.default_rng(np.random.SeedSequence([hotkey_int, secrets.randbits(64)]))

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for s in (signal.SIGTERM, signal.SIGINT):
        try: loop.add_signal_handler(s, stop.set)
        except (NotImplementedError, ValueError): signal.signal(s, lambda *_: stop.set())

    rows = store.read()
    champion, reign_start, last_published_uid = _load_reign_state(reign_state_path)
    if champion is not None:
        log.info(f"resuming reign: uid {champion[0]}@{champion[1]} from block {reign_start} "
                 f"(last_published_uid={last_published_uid})")
    attempted: set[tuple[int, str]] = set()   # (uid, revision) tried this reign
    king_fail_backoff: int = 60               # 60→120→240→480→600 on repeated failures
    # King slot persists across duels within a reign. Re-provisioning a 600GB
    # model download for every challenger was the dominant cost: this single
    # cache saves ~10min per duel on average.
    king_slot: Slot | None = None

    async def _drop_king():
        nonlocal king_slot
        if king_slot is None:
            return
        slot, king_slot = king_slot, None
        # Shield: a parent cancel mid-teardown (SIGTERM during shutdown) would
        # otherwise abort the Targon DELETE call and orphan the rental on the
        # account. Shield ensures the delete completes; reconcile() at next
        # startup is the backstop if even that fails.
        try:
            await asyncio.shield(_safe_teardown(slots, slot, "king-drop"))
        except asyncio.CancelledError:
            log.warning("king teardown cancelled mid-flight; reconcile will clean up")
            raise

    async def _maybe_publish(uid: int, expected_hotkey: str) -> None:
        nonlocal last_published_uid
        if uid == last_published_uid:
            return
        audit(type="weight_intent", netuid=cfg.netuid, winner_uid=uid,
              dry_run=_truthy_env("AFFINE_DRY_RUN"))
        # Don't wrap in _cancellable: cancelling between broadcast and inclusion-wait
        # leaves the extrinsic on-chain but `last_published_uid` unset, so a restart
        # double-submits and burns weight quota. set_weights owns its own retry budget.
        if not await chain.publish_winner(uid, expected_hotkey):
            log.warning(f"publish_winner uid={uid} failed; will retry next iteration")
            return
        # Persist BEFORE mutating in-memory: if _save raises (disk full, etc.) and we'd
        # already advanced last_published_uid in memory, we'd skip the next idempotent
        # retry but the disk would still say "not published" — restart would re-submit
        # and burn weight quota. Save-first means a save failure leaves both memory and
        # disk un-advanced; the next iteration retries publish (idempotent on chain).
        if champion is not None:
            _save_reign_state(reign_state_path, champion, reign_start, uid)
        last_published_uid = uid

    try:
        while not stop.is_set():
            try:
                miners = skip.filter(await chain.list_miners())
                if not miners:
                    log.info("no miners; sleeping 120s")
                    await _cancellable(asyncio.sleep(120), stop); continue

                by_key = {(m.uid, m.revision): m for m in miners}
                if champion is None or champion not in by_key:
                    await _drop_king()
                    # Stage the new reign in locals; only commit (champion/reign_start
                    # mutation + persist) once every step succeeds. A failure mid-way
                    # used to assign champion before reign_start, leaving the next
                    # iteration with the new champion paired against the prior reign's
                    # reign_start and an artificially low k(reign).
                    new_champion = await _elect(rows, miners, env_names, priors)
                    new_reign_start = await chain.current_block()
                    last_published_uid = None
                    _save_reign_state(reign_state_path, new_champion, new_reign_start, None)
                    champion, reign_start, attempted = new_champion, new_reign_start, set()
                    if stop.is_set(): break
                    log.info(f"champion: uid {champion[0]}@{champion[1]} (reign start block {reign_start})")
                if stop.is_set(): break
                king = by_key[champion]
                # Idempotent: no-op when uid == last_published_uid. Retries a publish
                # that failed mid-RPC on a prior iteration without re-electing.
                await _maybe_publish(king.uid, king.hotkey)
                queue = sorted(
                    (m for m in miners if (m.uid, m.revision) != champion
                                       and (m.uid, m.revision) not in attempted),
                    key=lambda m: (m.block, _tiebreak(m)),
                )
                if not queue:
                    attempted.clear()
                    log.info("queue exhausted this reign; sleeping 120s before re-probe")
                    await _cancellable(asyncio.sleep(120), stop); continue

                challenger = queue[0]
                attempted.add((challenger.uid, challenger.revision))

                if king_slot is not None and not (
                    await _cancellable(health_ping(king_slot.base_url), stop)
                    and await _cancellable(inference_ping(king_slot.base_url, king.model), stop)
                ):
                    log.warning(f"cached king slot unhealthy; re-provisioning")
                    await _drop_king()

                chal_slot: Slot | None = None
                king_attempt_failed = False
                try:
                    if king_slot is None:
                        king_slot, chal_slot, king_attempt_failed = await _provision_pair(slots, king, challenger, skip, stop)
                    else:
                        chal_slot, chal_status = await _provision(slots, challenger, stop)
                        _apply_skip(skip, challenger, chal_status, is_king=False,
                                    king_artifact=(king.model, king.revision))
                except asyncio.CancelledError:
                    if stop.is_set(): break
                    raise

                if king_slot is None:
                    if chal_slot is not None: await _safe_teardown(slots, chal_slot, "king-failed")
                    if king_attempt_failed:
                        log.warning(f"king uid {king.uid} provision failed; sleeping {king_fail_backoff}s before retry")
                        # Keep champion identity — only reset if chain says they're gone. Don't
                        # re-elect just because Targon hiccuped, or we burn `set_weights` quota.
                        attempted.discard((challenger.uid, challenger.revision))
                        await _cancellable(asyncio.sleep(king_fail_backoff), stop)
                        king_fail_backoff = min(king_fail_backoff * 2, 600)
                    else:
                        # King was cancelled by chal-fail-fast. Chal already got
                        # session-skipped via _apply_skip; advance to next challenger
                        # without backing off the king.
                        log.info(f"chal uid {challenger.uid} provision failed; advancing without king backoff")
                    continue
                king_fail_backoff = 60
                if chal_slot is None:
                    continue

                # chal_slot lifetime is owned by this block: it's only retained on a
                # successful dethrone (where it's promoted to king_slot). Every other
                # exit path — verdict aborts, exceptions, shutdown — must tear it down.
                promoted = False
                try:
                    fit = _fit(rows, miners, env_names, priors)
                    c0, _ = fit.contrast(_index(miners, challenger.uid, challenger.revision),
                                         _index(miners, king.uid, king.revision))
                    log.info(
                        f"duel: king uid{king.uid} vs chal uid{challenger.uid} "
                        f"(pre-dwell Δθ̂={c0:+.3f}, rows={len(rows)})"
                    )

                    rows_before = len(rows)
                    rows, fit, abort_reason = await _dwell(chain, king, king_slot, challenger, chal_slot,
                                                            miners, rows, envs, env_names, store, cfg,
                                                            priors, rng, stop)

                    if abort_reason == "chal_broken":
                        # Session-skip the chal artifact UNLESS it's the same as king's:
                        # chal_broken implies king was sampling successfully, so the king
                        # is healthy on this artifact. Skipping it would filter king on
                        # the next iteration. (Cached-king + chal sharing king's
                        # artifact CAN reach _dwell — two miners committing the same
                        # popular base model, second uid as challenger.)
                        if (challenger.model, challenger.revision) != (king.model, king.revision):
                            skip.add(challenger.model, challenger.revision, durable=False)
                        else:
                            log.info(f"chal_broken on king-shared artifact {king.model}@{king.revision}; "
                                     f"not session-skipping to preserve king cache")
                    elif abort_reason == "king_broken":
                        # Drop the slot. Re-provision next iter — Targon may give a different
                        # host. Don't re-elect: a transient leak shouldn't flip the throne, and
                        # re-elect would just pick the same uid back if it still has max θ̂.
                        log.warning(f"king uid{king.uid} mid-dwell broken; dropping slot for re-provision")
                        await _drop_king()

                    if len(rows) == rows_before:
                        log.info(f"duel aborted with no evidence (king uid{king.uid} vs chal uid{challenger.uid}); skipping verdict")
                        audit(type="duel_aborted", reason=abort_reason or "no_rows",
                              king={"uid": king.uid, "model": king.model, "revision": king.revision},
                              challenger={"uid": challenger.uid, "model": challenger.model, "revision": challenger.revision})
                        continue

                    delta, se = fit.contrast(_index(miners, challenger.uid, challenger.revision),
                                             _index(miners, king.uid, king.revision))
                    z = delta / se if se > 0 else 0.0
                    reign = await chain.current_block() - reign_start
                    k = compute_k(reign, cfg.k_init, cfg.k_final, cfg.k_halflife)
                    log.info(f"verdict: Δθ̂={delta:+.3f}±{se:.3f} z={z:+.2f} k={k:.2f} reign={reign}b")

                    verdict = "CHALLENGER_WINS" if z > k else "CHAMPION_HOLDS"
                    audit(type="duel", verdict=verdict, abort_reason=abort_reason,
                          king={"uid": king.uid, "model": king.model, "revision": king.revision},
                          challenger={"uid": challenger.uid, "model": challenger.model, "revision": challenger.revision},
                          delta=float(delta), se=float(se), z=float(z), k=float(k),
                          reign_blocks=int(reign),
                          rows_per_env={e: [p, n] for e, (p, n) in _rows_per_env(
                              rows[rows_before:], challenger.uid, king.uid).items()})

                    if z > k and not stop.is_set():
                        # Re-list before promoting: a re-commit during the duel would
                        # have changed the artifact identity, and publishing the uid
                        # would weight a revision that no longer exists on chain.
                        fresh = {(m.uid, m.revision) for m in await chain.list_miners()}
                        if (king.uid, king.revision) not in fresh:
                            log.warning(f"verdict skipped: king uid{king.uid}@{king.revision} no longer registered")
                        elif (challenger.uid, challenger.revision) not in fresh:
                            log.warning(f"verdict skipped: challenger uid{challenger.uid}@{challenger.revision} no longer registered")
                        else:
                            new_reign_start = await chain.current_block()
                            await _drop_king()
                            king_slot = chal_slot
                            promoted = True
                            champion = (challenger.uid, challenger.revision)
                            reign_start = new_reign_start
                            attempted = set()
                            last_published_uid = None
                            _save_reign_state(reign_state_path, champion, reign_start, None)
                            await _maybe_publish(challenger.uid, challenger.hotkey)
                            log.info(f"DETHRONE: uid {king.uid} → uid {challenger.uid}")
                finally:
                    if not promoted:
                        await _safe_teardown(slots, chal_slot, "chal-end-of-duel")

            except Exception as e:
                log.error(f"loop iteration: {e}", exc_info=True)
                await _cancellable(asyncio.sleep(60), stop)
    finally:
        log.info("shutdown")
        await _drop_king()
        for wrapper, _ in envs.values():
            try: await asyncio.wait_for(wrapper.cleanup(), timeout=30.0)
            except (Exception, asyncio.TimeoutError) as e:
                log.warning(f"env cleanup: {e}")


def bittensor_chain(cfg: Config) -> tuple[Chain, Subtensor]:
    """Wire a Chain from Bittensor subtensor + wallet. Caller owns sub.close()."""
    import bittensor as bt
    sub = Subtensor(cfg.subtensor_endpoint, cfg.subtensor_fallback)
    wallet = bt.Wallet(name=cfg.wallet_name, hotkey=cfg.hotkey_name)
    hotkey = wallet.hotkey.ss58_address

    async def publish(uid: int, expected_hotkey: str) -> bool:
        return await set_weights(sub, wallet, cfg.netuid, uid, expected_hotkey)

    return Chain(
        hotkey=hotkey,
        list_miners=lambda: get_miners(sub, cfg.netuid, hotkey),
        current_block=sub.get_current_block,
        publish_winner=publish,
    ), sub


def static_chain(miners: list[Miner], hotkey: str = "local-validator") -> Chain:
    """A Chain wired to a fixed miner list. For local dev / tests."""
    async def _miners(): return miners
    async def _block(): return 0
    async def _publish(uid: int, expected_hotkey: str) -> bool:
        log.info(f"local winner: uid {uid} (hk {expected_hotkey[:8]})"); return True
    return Chain(hotkey=hotkey, list_miners=_miners, current_block=_block, publish_winner=_publish)
