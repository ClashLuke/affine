"""Pairwise king-of-the-hill validator loop backed by a global 2PL IRT fit.

The champion reigns until a challenger statistically proves superiority.

Each outer iteration:

  1. Read miners. If the champion is unknown or no longer registered, elect
     argmax(θ̂) from the global fit and publish them.
  2. Pick the lowest-block challenger not yet attempted this reign.
  3. Provision king and challenger on two slots in parallel.
  4. Dwell: keep `cfg.dwell_batch` matched-task pairs in flight; for each
     pick, choose the env maximizing Fisher info for the contrast
     (θ_chal − θ_king), sample both models concurrently, append two rows,
     refit. The posterior sharpens as evidence accrues. Exits only on
     principled stops: z > k (dethrone), z < −k (chal can't recover under
     unbounded future info), shutdown, or one side's endpoint is
     persistently dead (≥ SLOT_DEAD consec same-side delivery failures).
     No budget cap.
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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Awaitable, Callable

import httpx
import numpy as np
import affinetes as af

from .audit import audit
from .chain import Miner, Subtensor, _tiebreak, _truthy_env, get_miners, set_weights
from .config import BASELINE_MODELS, Config
from .evidence import EvidenceStore, Row, atomic_append
from .irt import Fit, Priors, compute_k, fisher_env, fit_2pl
from .sampler import run_one
from .verdict import ArtKey, Dethrone, DuelStatus, Hold, Skip, VerdictEvidence, decide
from .vllm import Slot, SlotProvisionFailed, TargonSlots

log = logging.getLogger(__name__)

SLOT_DEAD = 30   # consecutive same-slot delivery failures → abort dwell, reprovision


@dataclass
class Chain:
    """Everything the loop needs from the world besides slots, envs, and evidence."""
    hotkey: str
    list_miners: Callable[[], Awaitable[list[Miner]]]
    current_block: Callable[[], Awaitable[int]]
    publish_winner: Callable[[int, str], Awaitable[bool]]


def art_key(m: Miner) -> ArtKey:
    return (m.model, m.revision)


class MinerStates:
    """`durable` (disk-backed JSONL) is per-(uid, art_key): vLLM crashed loading
    this artifact. Per-uid so a fresh uid that re-commits a known-broken artifact
    gets one shot before being marked itself. `attempted` (in-memory) is
    per-art_key: deduplicating multi-uid-per-artifact wastes — IRT pools evidence
    by art_key, so one duel's worth is sufficient regardless of how many uids
    share the artifact. Cleared on reign change or queue exhaustion. UNTRIED is
    the implicit default (no record either side)."""
    def __init__(self, excluded_models: set[str] = frozenset(),
                 path: str | Path | None = None):
        self._durable: set[tuple[int, ArtKey]] = set()
        self._attempted: set[ArtKey] = set()
        self._excluded_models = set(excluded_models)
        self.path = Path(path) if path else None
        if self.path and self.path.exists():
            skipped = 0
            for line in self.path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                    self._durable.add((int(d["uid"]), (d["model"], d["revision"])))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    skipped += 1
            if self._durable or skipped:
                log.info(f"states: loaded {len(self._durable)} durable entries"
                         f"{f' (skipped {skipped} malformed lines)' if skipped else ''} from {self.path}")

    def mark_durable(self, uid: int, k: ArtKey, reason: str) -> None:
        if (uid, k) in self._durable:
            return
        # Disk first, memory second — a failed write must not leave us thinking
        # we persisted a skip we'll lose on restart.
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps({"uid": uid, "model": k[0],
                                   "revision": k[1], "reason": reason}) + "\n"
            atomic_append(self.path, payload.encode())
        self._durable.add((uid, k))
        log.info(f"durable: uid{uid} {k[0]}@{k[1]} ({reason})")

    def mark_attempted(self, k: ArtKey) -> None:
        self._attempted.add(k)

    def is_attempted(self, k: ArtKey) -> bool:
        return k in self._attempted

    def is_durable(self, uid: int, k: ArtKey) -> bool:
        return (uid, k) in self._durable

    def filter(self, miners: list[Miner]) -> list[Miner]:
        return [m for m in miners
                if m.model not in self._excluded_models
                and (m.uid, art_key(m)) not in self._durable]

    def clear_attempted(self) -> None:
        self._attempted.clear()


def _row_art(r: Row, by_uid_rev: dict[tuple[int, str], Miner]) -> ArtKey:
    """Resolve a row to its artifact key. New rows carry `k` (model) directly;
    legacy rows fall back to (1) the current miner with (uid, revision), (2) a
    per-uid ghost so two retired uids that happened to share a `revision` string
    never accidentally pool. Ghost evidence still informs env parameters."""
    if r.k is not None:
        return (r.k, r.r)
    m = by_uid_rev.get((r.m, r.r))
    if m is not None:
        return (m.model, r.r)
    return (f"?ghost:{r.m}:{r.r}", r.r)


def _respondents(miners: list[Miner], rows: list[Row]) -> list[ArtKey]:
    """Registered artifacts first; historical ghosts at the tail so their evidence
    still informs env parameters even after a re-commit replaces them. Two miners
    sharing the same (model, revision) collapse to a single respondent."""
    by_uid_rev = {(m.uid, m.revision): m for m in miners}
    keys: list[ArtKey] = []
    seen: set[ArtKey] = set()
    for m in miners:
        k = art_key(m)
        if k not in seen:
            keys.append(k); seen.add(k)
    for r in rows:
        k = _row_art(r, by_uid_rev)
        if k not in seen:
            keys.append(k); seen.add(k)
    return keys


def _fit(rows: list[Row], miners: list[Miner], env_names: list[str],
         priors: Priors, init_x: np.ndarray | None = None) -> Fit:
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
    by_uid_rev = {(m.uid, m.revision): m for m in miners}
    outcomes: dict[str, set[int]] = {}
    pairs: dict[tuple[str, int, int], list[float]] = {}
    for r in rows:
        if r.e in e2i:
            outcomes.setdefault(r.e, set()).add(r.p)
        pairs.setdefault((r.e, r.t, r.i), []).append(r.l)
    drop = {n for n, outs in outcomes.items() if len(outs) < 2}
    # Synth-on-both pairs (both sides l=0) carry no information but in 2PL add
    # Hessian mass α²σ(η)(1-σ(η)) per row, sharply tightening any wide posterior
    # (e.g. fresh chal at prior). One such pair shifted z from -0.35 → -3.56 in
    # the 2026-04-28 incident — fit-time drop is the principled fix; gaming
    # defense via single-side synth-loss rows (l=0 with l>0 partner) is preserved.
    synth = {g for g, ls in pairs.items() if all(l == 0.0 for l in ls)}
    filtered = [r for r in rows if r.e in e2i and r.e not in drop
                and (r.e, r.t, r.i) not in synth]
    m_idx = np.fromiter((k2i[_row_art(r, by_uid_rev)] for r in filtered),
                        dtype=np.intp, count=len(filtered))
    e_idx = np.fromiter((e2i[r.e] for r in filtered), dtype=np.intp, count=len(filtered))
    y = np.fromiter((r.p for r in filtered), dtype=np.float64, count=len(filtered))
    return fit_2pl(m_idx, e_idx, y, len(keys), len(env_names), priors, init_x=init_x)


async def _load_envs(cfg: Config) -> dict[str, tuple]:
    # Dedupe by image: load each unique image once, share across envs that use it.
    # params are call-time (passed in /call body) so each env carries its own;
    # env_vars and mem_limit are container-init so they MUST match across shared envs.
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
                out[spec.name] = (wrapper, spec)
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
            try: await asyncio.wait_for(wrapper.cleanup(), timeout=30.0)
            except Exception as e:
                log.warning(f"env cleanup-on-fail: {e}")
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
            except BaseException as e:
                # Recursive cancellation during shielded await: worker still running,
                # not awaited. Same leak class as TimeoutError above.
                log.warning(f"_cancellable: shielded wait interrupted ({type(e).__name__}); rental may leak until reconcile")
        if not delivered and worker.done() and not worker.cancelled() and worker.exception() is None:
            orphan = worker.result()
            if on_orphan is not None and orphan is not None:
                try: await asyncio.shield(on_orphan(orphan))
                except BaseException as e:
                    log.warning(f"_cancellable: on_orphan failed: {type(e).__name__}: {e}")


async def _safe_teardown(slots, slot: Slot, ctx: str) -> None:
    """Shielded so an outer cancel during teardown doesn't interrupt the Targon
    DELETE — the request keeps flying as a fire-and-forget task; reconcile() at
    next startup is the backstop. CancelledError still propagates to the caller."""
    try: await asyncio.shield(slots.teardown(slot))
    except Exception as e: log.warning(f"teardown error ({ctx}): {e}")


def _slot_from_task(t: asyncio.Task) -> Slot | None:
    """Extract the slot from a task wrapping `_provision`. Returns None for any
    not-clean state (still running, cancelled, raised). The not-done guard is
    load-bearing: t.exception() raises InvalidStateError on a running task,
    which can happen if `_cleanup`'s drain await is interrupted."""
    if not t.done() or t.cancelled() or t.exception() is not None: return None
    return t.result()[0]


async def _provision(slots, miner: Miner, stop: asyncio.Event) -> tuple[Slot | None, str]:
    """Returns (slot|None, status). `crashloop` is the only miner-fault signal —
    vLLM crashed loading the artifact. `timeout`/`transient`/`error` are our
    infrastructure (we picked the resource, image, timeout). The slot is *not*
    further probed here: a `/chat/completions` health check is our test of our
    slot, and a failure would conflate orchestration with miner fault. A slot
    that provisions but can't actually serve produces no rows for that side;
    SLOT_DEAD aborts the dwell and the outer loop reprovisions or marks
    durable as appropriate."""
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
        log.warning(f"provision transient uid {miner.uid}: {type(e).__name__}: {e}")
        return None, "transient"
    except Exception as e:
        log.error(f"provision error uid {miner.uid}: {e}")
        return None, "error"
    return slot, "ok"


def _apply_skip(states: MinerStates, miner: Miner, status: str, *, is_king: bool) -> None:
    """crashloop → durable (miner fault). Other non-king → attempted at
    provision time, so the queue advances regardless of duel outcome. King
    isn't queued; non-crashloop is a no-op."""
    if status == "crashloop":
        states.mark_durable(miner.uid, art_key(miner), reason=status)
    elif not is_king:
        states.mark_attempted(art_key(miner))


async def _provision_pair(slots, king: Miner, chal: Miner, states: MinerStates,
                          stop: asyncio.Event,
                          ) -> tuple[Slot | None, Slot | None, bool]:
    """Provision king and challenger concurrently with fail-fast cancellation.

    Returns `(king_slot, chal_slot, king_attempt_failed)`. `king_attempt_failed`
    is True iff king's provision ran to completion with a non-ok status (False
    if king was cancelled because chal failed first). `_apply_skip` runs for both
    miners; chal status doesn't propagate.

    If the first finisher fails to produce a slot, the duel is dead — cancel the
    sibling rather than wait out a 15-min Targon provision we'll discard. On
    outer cancellation, drain pending tasks and teardown any completed slot;
    gather() otherwise drops survivors and leaks rentals.
    """
    t_k = asyncio.create_task(_provision(slots, king, stop))
    t_c = asyncio.create_task(_provision(slots, chal, stop))
    tasks = (t_k, t_c)

    async def _cleanup(reason: str, drain: bool = False) -> None:
        if drain:
            for t in tasks:
                if not t.done(): t.cancel()
            for t in tasks:
                try: await asyncio.shield(t)
                except BaseException as e:
                    log.warning(f"_provision_pair drain interrupted ({type(e).__name__}); rental may leak")
        for t in tasks:
            if (slot := _slot_from_task(t)) is not None:
                await _safe_teardown(slots, slot, reason)

    pending = set(tasks)
    fail_fast = False
    try:
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                if t.cancelled() or t.exception() is not None: continue
                if t.result()[0] is None and pending:
                    fail_fast = True
                    for p in pending: p.cancel()
    except asyncio.CancelledError:
        await asyncio.shield(_cleanup("outer-cancelled", drain=True))
        raise

    # Outer-cancel during inner provision (stop fired): tasks ended cancelled
    # but we didn't trigger fail_fast.
    outer_cancelled = not fail_fast and any(t.cancelled() for t in tasks)

    try:
        for t, m, is_king in ((t_k, king, True), (t_c, chal, False)):
            if t.cancelled(): continue
            if (exc := t.exception()) is not None:
                log.error(f"provision-pair unexpected exception uid {m.uid}: {exc}")
                continue
            _apply_skip(states, m, t.result()[1], is_king=is_king)
    except BaseException:
        await _cleanup("pair-error")
        raise

    if outer_cancelled:
        await _cleanup("pair-cancelled")
        raise asyncio.CancelledError()

    king_slot, chal_slot = _slot_from_task(t_k), _slot_from_task(t_c)
    king_attempt_failed = not t_k.cancelled() and king_slot is None
    if fail_fast:
        await _cleanup("pair-fail-fast")
        return None, None, king_attempt_failed
    # King-only failure: tear down the orphan chal slot. Chal-only failure
    # (king ok) keeps king cached — the 600GB model download reuses across chals.
    if king_slot is None and chal_slot is not None:
        await _safe_teardown(slots, chal_slot, "king-failed-chal-orphan")
        return None, None, king_attempt_failed
    return king_slot, chal_slot, king_attempt_failed


def _seed(uid: int, rev: str, env: str, c: int, salt: str = "") -> int:
    """31 bits (signed int32 max). vLLM accepts ≤ 2^63-1, but game's openspiel env
    feeds the seed into np.random.RandomState which only accepts [0, 2^32-1] and
    internally computes seed+1..seed+100, so we mask to 2^31-1 for portability.
    `salt` is mixed in to give per-validator task sequences; it's the validator's
    public hotkey today (placeholder for the commit-reveal seed described in
    notes/design.md). Replay verification requires the salt be recoverable by
    peers, so it must NOT be made permanently private."""
    h = hashlib.sha256(f"{salt}\0{uid}\0{rev}\0{env}\0{c}".encode()).digest()
    return int.from_bytes(h[:4], "big") & ((1 << 31) - 1)


def _task_id(king_uid: int, chal_uid: int, env: str, iter_idx: int,
             lo: int, hi: int, salt: str = "") -> int:
    """Per-iter task id, shared by king and challenger. Order-invariant in the
    pair so swapping who-is-king doesn't reshuffle the schedule. design.md
    §"Challenge-centric evidence": both miners face the same instance,
    eliminating between-task variance — the duel measures relative capability
    on the same work, not relative luck on adjacent work. Uniform over [lo, hi].
    See `_seed` for the salt/commit-reveal contract."""
    a, b = sorted((king_uid, chal_uid))
    h = hashlib.sha256(f"{salt}\0{a}\0{b}\0{env}\0{iter_idx}".encode()).digest()
    n = hi - lo + 1
    return lo + (int.from_bytes(h[:8], "big") % n)


async def _sample(chain: Chain, wrapper, params: dict, timeout: float, slot: Slot,
                  miner: Miner, env_name: str, c: int,
                  task_id: int, block: int) -> tuple[Row, bool, int]:
    """One evaluation of `miner` in `env_name` on `slot`. Returns
    (row, delivered, tokens). `delivered=False` iff `run_one` returned None —
    the miner's endpoint failed to produce a parseable response (vLLM 5xx,
    conn refused, env-side error, wrapper exception). The synthetic row
    (p=0, l=0) is constructed regardless; the caller decides whether to keep
    it based on the *paired* sample's outcome — see dwell's row-append block.

    Real per-task miner timeouts (the outer asyncio.wait deadline in run_one)
    return outcome=False, not None: those are decisive misses per plan.md
    "Challenger times out on a task → that task is a loss" and yield a
    real-loss row (p=0, l=actual_latency, delivered=True).

    `c` is pre-allocated by the caller: the dwell may dispatch many samples for
    the same (uid, rev, env) in parallel before any append, and reading
    `store.next_counter` inside each call would return the same value for all of
    them — colliding on Row identity and seed. Hoisting allocation lets the
    caller assign sequential counters across the in-flight batch."""
    outcome, latency, tokens = await run_one(
        wrapper, params, timeout, slot,
        seed=_seed(miner.uid, miner.revision, env_name, c, salt=chain.hotkey),
        task_id=task_id)
    if outcome is None:
        return Row(m=miner.uid, r=miner.revision, e=env_name, c=c,
                   p=0, t=int(block), l=0.0, i=int(task_id), k=miner.model), False, 0
    return Row(m=miner.uid, r=miner.revision, e=env_name, c=c,
               p=int(bool(outcome)), t=int(block),
               l=float(latency), i=int(task_id), k=miner.model), True, int(tokens)


async def _pair_sample(chain: Chain, wrapper, params: dict, timeout: float,
                       king_slot: Slot, king: Miner, ck: int,
                       chal_slot: Slot, challenger: Miner, cc: int,
                       env_name: str, task_id: int, block: int, e_idx: int):
    """One matched-task king/chal pair on env e. Returns
    (e_idx, env_name, king_result, chal_result) where each result is (Row, bool)
    or BaseException — exception per-side is tolerated so the dwell can
    attribute env-side plumbing failures separately from miner outcomes."""
    rk, rc = await asyncio.gather(
        _sample(chain, wrapper, params, timeout, king_slot, king, env_name, ck, task_id, block),
        _sample(chain, wrapper, params, timeout, chal_slot, challenger, env_name, cc, task_id, block),
        return_exceptions=True,
    )
    return e_idx, env_name, rk, rc


async def dwell(chain: Chain, king: Miner, king_slot: Slot,
                challenger: Miner, chal_slot: Slot, miners: list[Miner],
                rows: list[Row], envs, env_names, store: EvidenceStore,
                cfg: Config, priors: Priors,
                rng: np.random.Generator, stop: asyncio.Event,
                reign_start_block: int,
                ) -> DuelOutcome:
    """Queue-driven dwell: keep `cfg.dwell_batch` matched-task pairs in flight at
    all times, harvest via FIRST_COMPLETED, refill on completion. Refit on every
    wakeup that yielded new rows. Exits only when the evidence answers the
    question — no budget cap.

    Termination:
      - z > k(reign)  → dethrone (DuelStatus.COMPLETED)
      - z < −k(reign) → asymptotic mirror; under unbounded future info the chal
        cannot reach k, so HOLDS is decided (DuelStatus.COMPLETED)
      - stop event   → shutdown (DuelStatus.CANCELLED)
      - king-side delivery dead (>= SLOT_DEAD consec) → DuelStatus.KING_SLOT_DEAD
      - chal-side delivery dead (>= SLOT_DEAD consec) → DuelStatus.CHAL_SLOT_DEAD

    Slot health is tracked per-side as `king_consec_fails` / `chal_consec_fails`:
    consecutive pair-completions where that side's endpoint failed to deliver,
    reset on any delivery. Crossing SLOT_DEAD aborts with KING_SLOT_DEAD or
    CHAL_SLOT_DEAD so the outer loop reprovisions the dead side.

    Row-append rule: always append both rows; failing sides get synthetic
    p=0,l=0 (constructed in `_sample`). Matched-pair contrast contribution
    is zero either way on both-fail (Δp=0), but appending preserves row
    count and prevents an attacker from suppressing their loss rate via
    selective crashes that coincide with king failures (drop would inflate
    θ̂_chal by P(king_fail) — non-trivial in hard envs at reign-end k≈1).
    Asymmetric pairs (one delivered, one not) implement plan.md L47 "first
    functioning challenger wins by default": broken-king pairs contribute
    chal-pass + king-synthetic-loss rows that drive z>k → Dethrone.

    Real per-task miner timeouts come back as outcome=False (not None) from
    run_one and produce real-loss rows (delivered=True, p=0, l=actual).
    Synthetic vs real is distinguishable downstream by l==0.0.

    Counter allocation is local to the dwell: store.next_counter is read once
    per (uid, rev, env) at first dispatch, then incremented in-process.

    The z<−k stop is the iters_remaining→∞ limit of the Bayesian projection
    `Δθ̂_n < k·(SE_∞ − √(SE_n²−SE_∞²))`: with no cap, SE_∞ → 0, drift_var → SE_n²,
    criterion collapses to z < −k. Symmetric mirror of dethrone, no
    hyperparameters beyond k(reign)."""
    import time as _time
    rows0 = len(rows)
    def _out(status: DuelStatus, fit: Fit) -> DuelOutcome:
        return DuelOutcome(rows=rows, fit=fit, status=status, rows_added=len(rows) - rows0)
    fit = _fit(rows, miners, env_names, priors)
    init_x = (np.concatenate([fit.theta, fit.beta, fit.alpha])
              if not fit.degenerate else None)
    art_keys = _respondents(miners, rows)
    k_idx = art_keys.index(art_key(king))
    c_idx = art_keys.index(art_key(challenger))
    king_consec_fails = 0
    chal_consec_fails = 0
    next_c: dict[tuple[int, str, str], int] = {}
    inflight: set[asyncio.Task] = set()
    iters_started = 0
    iters_done = 0
    samples_done = 0; tokens_done = 0; latency_sum = 0.0
    t_start = _time.monotonic()

    def alloc_c(uid: int, rev: str, env: str) -> int:
        key = (uid, rev, env)
        c = next_c.setdefault(key, store.next_counter(uid, rev, env))
        next_c[key] = c + 1
        return c

    def pick_env() -> int:
        if fit.degenerate:
            return int(rng.choice(len(env_names)))
        return fisher_env(fit, c_idx, k_idx, rng, excluded=frozenset())

    def dispatch_one(e_idx: int, iter_idx: int, block: int) -> asyncio.Task:
        env_name = env_names[e_idx]
        wrapper, spec = envs[env_name]
        params = {k: v for k, v in spec.params.items() if k != "timeout"}
        timeout = float(spec.params.get("timeout", 600))
        lo, hi = spec.task_range
        task_id = _task_id(king.uid, challenger.uid, env_name, iter_idx, lo, hi, salt=chain.hotkey)
        ck = alloc_c(king.uid, king.revision, env_name)
        cc = alloc_c(challenger.uid, challenger.revision, env_name)
        return asyncio.create_task(_pair_sample(
            chain, wrapper, params, timeout,
            king_slot, king, ck, chal_slot, challenger, cc,
            env_name, task_id, block, e_idx))

    async def drain(reason: str):
        for t in inflight:
            if not t.done(): t.cancel()
        for t in list(inflight):
            try: await asyncio.shield(t)
            except BaseException as ex:
                log.warning(f"dwell drain ({reason}) interrupted: {type(ex).__name__}")
        inflight.clear()

    try: block = await chain.current_block()
    except Exception as ex:
        log.warning(f"current_block failed at dwell start: {ex}; using 0")
        block = 0

    while True:
        if stop.is_set():
            await drain("stop"); return _out(DuelStatus.CANCELLED, fit)
        while len(inflight) < cfg.dwell_batch:
            inflight.add(dispatch_one(pick_env(), iters_started, block))
            iters_started += 1
        try:
            done, _pending = await _cancellable(
                asyncio.wait(inflight, return_when=asyncio.FIRST_COMPLETED), stop)
        except asyncio.CancelledError:
            await drain("cancel"); return _out(DuelStatus.CANCELLED, fit)

        new_rows: list[Row] = []
        for t in done:
            inflight.discard(t)
            _e_idx, env_name, rk, rc = t.result()
            iters_done += 1
            if isinstance(rk, BaseException):
                log.warning(f"dwell king-side raised on env={env_name}: {type(rk).__name__}: {rk}")
                k_row, k_ok, k_tok = None, False, 0
            else:
                k_row, k_ok, k_tok = rk
            if isinstance(rc, BaseException):
                log.warning(f"dwell chal-side raised on env={env_name}: {type(rc).__name__}: {rc}")
                c_row, c_ok, c_tok = None, False, 0
            else:
                c_row, c_ok, c_tok = rc
            king_consec_fails = 0 if k_ok else king_consec_fails + 1
            chal_consec_fails = 0 if c_ok else chal_consec_fails + 1
            # Always append both rows. Failing sides get synthetic p=0,l=0
            # (constructed in _sample). The matched-pair contrast contribution
            # is identical whether we drop or append-zero on both-fail (Δp=0
            # either way), but appending preserves the row count: an attacker
            # who crashes selectively on tasks they'd lose cannot suppress
            # their loss rate by hiding behind a coinciding king-fail. Drop
            # would let θ̂_chal float above true ability proportional to
            # P(king_fail) — non-trivial in hard envs at reign-end k≈1.
            # Validator-side env outages still inflate row count but not the
            # contrast (both p=0 → zero matched-pair contribution).
            if k_ok:
                samples_done += 1; tokens_done += k_tok; latency_sum += k_row.l
            if c_ok:
                samples_done += 1; tokens_done += c_tok; latency_sum += c_row.l
            if k_row is not None: new_rows.append(k_row)
            if c_row is not None: new_rows.append(c_row)
        if new_rows:
            store.append(*new_rows); rows.extend(new_rows)
        if king_consec_fails >= SLOT_DEAD and chal_consec_fails >= SLOT_DEAD:
            log.warning(f"dwell abort: BOTH slots dead (king={king.uid} consec={king_consec_fails}, "
                        f"chal={challenger.uid} consec={chal_consec_fails})")
            await drain("both_dead"); return _out(DuelStatus.KING_SLOT_DEAD, fit)
        if king_consec_fails >= SLOT_DEAD:
            log.warning(f"dwell abort: king slot dead after {king_consec_fails} consec fails (uid {king.uid})")
            await drain("king_dead"); return _out(DuelStatus.KING_SLOT_DEAD, fit)
        if chal_consec_fails >= SLOT_DEAD:
            log.warning(f"dwell abort: chal slot dead after {chal_consec_fails} consec fails (uid {challenger.uid})")
            await drain("chal_dead"); return _out(DuelStatus.CHAL_SLOT_DEAD, fit)

        if new_rows:
            try: block = await chain.current_block()
            except Exception as ex:
                log.warning(f"current_block refresh failed at i={iters_done}: {ex}; reusing previous")
            elapsed = max(1e-3, _time.monotonic() - t_start)
            sps = samples_done / elapsed
            mean_lat = latency_sum / samples_done if samples_done else 0.0
            tok_s = tokens_done / elapsed  # aggregate over both slots
            log.info(f"dwell pairs={iters_done} (in_flight={len(inflight)}/{cfg.dwell_batch}) "
                     f"samples={samples_done} tokens={tokens_done} elapsed={elapsed:.1f}s "
                     f"throughput={sps:.2f} sample/s {tok_s:.0f} tok/s mean_lat={mean_lat:.1f}s")
            fit = _fit(rows, miners, env_names, priors, init_x=init_x)
            if not fit.degenerate:
                init_x = np.concatenate([fit.theta, fit.beta, fit.alpha])
                delta, se = fit.contrast(c_idx, k_idx)
                z = delta / se if se > 0 else 0.0
                k = compute_k(block - reign_start_block,
                              cfg.k_init, cfg.k_final, cfg.k_halflife)
                if z > k:
                    log.info(f"dwell dethrone after {iters_done} iters: z={z:+.2f} > k={k:.2f}")
                    await drain("z>k"); return _out(DuelStatus.COMPLETED, fit)
                if z < -k:
                    log.info(f"dwell hold after {iters_done} iters: z={z:+.2f} < -k={-k:.2f}")
                    await drain("z<-k"); return _out(DuelStatus.COMPLETED, fit)


async def _elect(rows: list[Row], miners: list[Miner], env_names: list[str],
                 priors: Priors) -> tuple[int, str]:
    """Pick champion. Cold start (no evidence): seat a hardcoded baseline model if
    registered; otherwise pick the lowest-block miner (fairest tiebreaker — first
    to commit holds the throne until evidence dethrones them). With evidence:
    argmax(θ̂)."""
    def _seat_baseline_or_lowest(reason: str) -> tuple[int, str]:
        for target in BASELINE_MODELS:
            for m in miners:
                if m.model == target:
                    log.info(f"{reason}: seating baseline {m.model} as uid {m.uid}")
                    return m.uid, m.revision
        m = min(miners, key=lambda x: (x.block, _tiebreak(x)))
        log.info(f"{reason}: no baseline registered; seating lowest-block uid {m.uid} model {m.model}")
        return m.uid, m.revision
    if not rows:
        return _seat_baseline_or_lowest("cold start")
    # Uninformative rows (every env all-pass or all-fail) get fully dropped by
    # _fit's filter — the fit then runs on zero observations, theta stays at
    # the prior mean (zero), argmax breaks ties to index 0. Functionally
    # identical to cold start, so route there. Must mirror _fit's `r.e in env_names`
    # filter: a retired-env row with both outcomes would otherwise pass this check
    # while _fit drops it, leaving the fit with zero active observations.
    active = set(env_names)
    outcomes: dict[str, set[int]] = {}
    for r in rows:
        if r.e in active:
            outcomes.setdefault(r.e, set()).add(r.p)
    if not any(len(s) >= 2 for s in outcomes.values()):
        return _seat_baseline_or_lowest("elect: no informative env (all-pass or all-fail)")
    fit = _fit(rows, miners, env_names, priors)
    if fit.degenerate:
        # argmax(θ̂) on a non-MAP fit can publish a winner from a fabricated
        # posterior — same risk the verdict path refuses for. Fall back to the
        # cold-start choice; once new evidence accumulates the next election
        # gets a real fit.
        return _seat_baseline_or_lowest("elect: degenerate fit")
    # θ̂ is per-artifact (model, revision). With multiple uids on the winning
    # artifact, pick the lowest-block one — same tiebreak as cold start, so the
    # named seat-holder is deterministic even when uids share an artifact.
    art_keys = _respondents(miners, rows)
    registered = {art_key(m) for m in miners}
    candidates = [(i, k) for i, k in enumerate(art_keys) if k in registered]
    i_best, key_best = max(candidates, key=lambda ik: fit.theta[ik[0]])
    holders = [m for m in miners if art_key(m) == key_best]
    seat = min(holders, key=lambda m: (m.block, _tiebreak(m)))
    log.info(f"elect: uid {seat.uid} model {seat.model} (θ̂={fit.theta[i_best]:+.3f})")
    return seat.uid, seat.revision


@dataclass
class Reign:
    """Persisted champion + reign_start + last_published_uid. Without persistence,
    a restart with the same champion still on chain would re-elect them and reset
    reign_start to the current block, undoing all k(reign) decay. Save format and
    atomic-rename semantics are load-bearing for that durability."""
    champion: tuple[int, str]
    start_block: int
    last_published_uid: int | None = None

    @classmethod
    def load(cls, path: Path) -> "Reign | None":
        if not path.exists():
            return None
        try:
            d = json.loads(path.read_text())
            if not isinstance(d, dict):
                raise TypeError(f"reign state must be a JSON object, got {type(d).__name__}")
            lp = d.get("last_published_uid")
            return cls(
                champion=(int(d["uid"]), str(d["revision"])),
                start_block=int(d["reign_start"]),
                last_published_uid=(int(lp) if lp is not None else None),
            )
        except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            log.warning(f"reign state at {path} unreadable, ignoring: {e}")
            return None

    def save(self, path: Path) -> None:
        # fsync the tmp before rename: without it, power loss between rename commit
        # and the data flush can leave an empty/partial file. load() would then
        # return None and the loop re-elects, resetting start_block and undoing
        # all k(reign) decay that had accumulated.
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        payload = json.dumps({
            "uid": self.champion[0], "revision": self.champion[1],
            "reign_start": self.start_block,
            "last_published_uid": self.last_published_uid,
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


@dataclass
class DuelOutcome:
    rows: list[Row]
    fit: Fit
    status: DuelStatus
    rows_added: int


@dataclass
class LoopState:
    reign: Reign | None = None
    king_slot: Slot | None = None
    # Prefetched next-challenger provision. Started while the current duel dwells
    # so a fast z<-k or z>k early-exit doesn't stall on a 5-15min Targon spin.
    # Tuple of (task, miner). The task awaits to (slot|None, status) like _provision.
    next_chal: tuple[asyncio.Task, Miner] | None = None
    provision_backoff: int = 60   # exponential Targon-API-wide backoff; 60→600s


async def _drop_king(state: LoopState, slots) -> None:
    if state.king_slot is None:
        return
    slot, state.king_slot = state.king_slot, None
    await _safe_teardown(slots, slot, "king-drop")


async def _drop_next_chal(state: LoopState, slots) -> None:
    """Cancel any in-flight prefetch and tear down its slot if it managed to land.
    Idempotent. Called on shutdown and on cold-start (where the king isn't even
    alive — prefetch assumptions don't hold)."""
    if state.next_chal is None:
        return
    task, _miner = state.next_chal
    state.next_chal = None
    if not task.done():
        task.cancel()
    try:
        slot, _status = await asyncio.shield(task)
    except (asyncio.CancelledError, Exception):
        return
    if slot is not None:
        await _safe_teardown(slots, slot, "prefetch-drop")


def _start_prefetch(state: LoopState, slots, queue: list[Miner],
                    current_chal: Miner, states: MinerStates,
                    stop: asyncio.Event) -> None:
    """Kick off provisioning for the next-in-queue candidate. No-op if a
    prefetch is already pending or no candidate exists. We do NOT mark
    `attempted` at provision time — that defers to `_take_prefetched` so a
    Dethrone (which clears `attempted`) doesn't strand a rental we want to
    re-evaluate later."""
    if state.next_chal is not None:
        return
    cur_art = art_key(current_chal)
    nxt = next((m for m in queue if art_key(m) != cur_art), None)
    if nxt is None:
        return
    task = asyncio.create_task(_provision(slots, nxt, stop))
    state.next_chal = (task, nxt)
    log.info(f"prefetch: provisioning next chal uid{nxt.uid} {nxt.model}@{nxt.revision}")


async def _take_prefetched(state: LoopState, slots, states: MinerStates,
                           want_art: ArtKey) -> Slot | None:
    """If the prefetched miner matches `want_art`, await its provision and
    return the slot (mark_attempted on success). Otherwise teardown and return
    None — the queue moved beneath us (typically a Dethrone). `crashloop` is
    always recorded as `durable` even on mismatch: an artifact that crashed
    loading is a miner fault regardless of which iter saw it."""
    if state.next_chal is None:
        return None
    task, miner = state.next_chal
    state.next_chal = None
    try:
        slot, status = await task
    except asyncio.CancelledError:
        raise
    except Exception as e:
        log.warning(f"prefetch task raised: {e}")
        return None
    if status == "crashloop":
        states.mark_durable(miner.uid, art_key(miner), reason=status)
    if art_key(miner) != want_art:
        if slot is not None:
            log.info(f"prefetch mismatch: had {art_key(miner)}, want {want_art}; tearing down")
            await _safe_teardown(slots, slot, "prefetch-mismatch")
        return None
    if slot is None:
        return None
    states.mark_attempted(art_key(miner))
    log.info(f"prefetch hit: reusing slot for {want_art}")
    return slot


async def _maybe_publish(state: LoopState, chain: Chain, cfg: Config,
                         reign_state_path: Path, uid: int, expected_hotkey: str) -> None:
    """Idempotent: no-op when uid == reign.last_published_uid. Persists before
    advancing memory so a save failure leaves the next iter free to retry —
    the reverse order would lose a publish on save error and burn weight quota.
    Dry-run skips persistence (would mislead a later real run into skipping the
    actual on-chain write); memory advance still suppresses re-logging in-session."""
    if state.reign is None or uid == state.reign.last_published_uid:
        return
    dry = _truthy_env("AFFINE_DRY_RUN")
    audit(type="weight_intent", netuid=cfg.netuid, winner_uid=uid, dry_run=dry)
    # Not wrapped in _cancellable: cancelling between broadcast and inclusion-wait
    # leaves the extrinsic on-chain but last_published_uid unset; restart double-submits.
    if not await chain.publish_winner(uid, expected_hotkey):
        log.warning(f"publish_winner uid={uid} failed; will retry next iteration")
        return
    if not dry:
        Reign(state.reign.champion, state.reign.start_block, uid).save(reign_state_path)
    state.reign.last_published_uid = uid


def _miner_pl(m: Miner) -> dict:
    return {"uid": m.uid, "model": m.model, "revision": m.revision}


def _audit_verdict(verdict_str: str, king: Miner, chal: Miner, ev: VerdictEvidence) -> None:
    log.info(f"verdict: Δθ̂={ev.delta:+.3f}±{ev.se:.3f} z={ev.z:+.2f} k={ev.k:.2f} reign={ev.reign_blocks}b")
    audit(type="duel", verdict=verdict_str,
          king=_miner_pl(king), challenger=_miner_pl(chal), **asdict(ev))


async def run(cfg: Config, chain: Chain, slots=None):
    slots = slots or TargonSlots(cfg, hotkey=chain.hotkey)
    if hasattr(slots, "reconcile"):
        await slots.reconcile()
    envs: dict[str, tuple] = await _load_envs(cfg)
    # state.king_slot persists across duels within a reign — re-provisioning a
    # 600GB model download per challenger is the dominant cost. Hoisted out of
    # the inner try so the outer finally can shield-tear it down even on init failure.
    state = LoopState()
    try:
        env_names = [spec.name for spec in cfg.environments]
        store = EvidenceStore(cfg.evidence_path)
        reign_state_path = Path(cfg.evidence_path).parent / f"reign-{chain.hotkey[:8]}.json"
        states_path = Path(cfg.evidence_path).parent / f"skiplist-{chain.hotkey[:8]}.jsonl"
        states = MinerStates(
            {s.strip() for s in os.getenv("AFFINE_MODEL_SKIPLIST", "").split(",") if s.strip()},
            path=states_path,
        )
        priors = Priors(cfg.sigma_beta, cfg.sigma_alpha)
        # OS entropy so a crashloop doesn't deterministically replay the env that
        # triggered the crash. Per-sample task determinism is anchored on the
        # SHA(uid||rev||env||counter) seed elsewhere, not on this rng.
        rng = np.random.default_rng(secrets.randbits(64))

        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for s in (signal.SIGTERM, signal.SIGINT):
            try: loop.add_signal_handler(s, stop.set)
            except (NotImplementedError, ValueError): signal.signal(s, lambda *_: stop.set())

        rows = store.read()
        state.reign = Reign.load(reign_state_path)
        if state.reign is not None:
            log.info(f"resuming reign: uid {state.reign.champion[0]}@{state.reign.champion[1]} "
                     f"from block {state.reign.start_block} "
                     f"(last_published_uid={state.reign.last_published_uid})")

        while not stop.is_set():
            try:
                all_miners = await chain.list_miners()
                miners = states.filter(all_miners)
                if not miners:
                    log.info("no miners; sleeping 120s")
                    await _cancellable(asyncio.sleep(120), stop); continue

                by_key = {(m.uid, m.revision): m for m in miners}
                if state.reign is None or state.reign.champion not in by_key:
                    await _drop_king(state, slots)
                    # Build the new Reign value first; assign to state.reign only after
                    # save() succeeds. A failure mid-way used to leave the loop with the
                    # new champion paired against the prior reign's start_block and an
                    # artificially low k(reign).
                    new_champion = await _elect(rows, miners, env_names, priors)
                    new_reign = Reign(new_champion, await chain.current_block())
                    new_reign.save(reign_state_path)
                    state.reign = new_reign
                    states.clear_attempted()
                    if stop.is_set(): break
                    log.info(f"champion: uid {new_champion[0]}@{new_champion[1]} "
                             f"(reign start block {new_reign.start_block})")
                if stop.is_set(): break
                king = by_key[state.reign.champion]
                await _maybe_publish(state, chain, cfg, reign_state_path, king.uid, king.hotkey)
                champion_art = art_key(king)
                queue = sorted(
                    (m for m in miners if art_key(m) != champion_art
                                       and not states.is_attempted(art_key(m))),
                    key=lambda m: (m.block, _tiebreak(m)),
                )
                if not queue:
                    states.clear_attempted()
                    log.info("queue exhausted this reign; sleeping 120s before re-probe")
                    await _cancellable(asyncio.sleep(120), stop); continue

                challenger = queue[0]

                chal_slot: Slot | None = None
                king_attempt_failed = False
                try:
                    if state.king_slot is None:
                        # Cold-start: any pending prefetch was based on a stale
                        # king assumption — drop it before _provision_pair.
                        await _drop_next_chal(state, slots)
                        state.king_slot, chal_slot, king_attempt_failed = await _provision_pair(
                            slots, king, challenger, states, stop)
                    else:
                        chal_slot = await _take_prefetched(state, slots, states, art_key(challenger))
                        if chal_slot is None:
                            chal_slot, status = await _provision(slots, challenger, stop)
                            _apply_skip(states, challenger, status, is_king=False)
                except asyncio.CancelledError:
                    if stop.is_set(): break
                    raise

                # King-failed → backoff to avoid hammering Targon. Chal failures are
                # already ATTEMPTED via _apply_skip; the queue advances next iter,
                # and queue exhaustion provides the natural multi-chal backoff (120s).
                if king_attempt_failed:
                    log.warning(f"king provision failed; sleeping {state.provision_backoff}s")
                    await _cancellable(asyncio.sleep(state.provision_backoff), stop)
                    state.provision_backoff = min(state.provision_backoff * 2, 600)
                if chal_slot is None:
                    continue
                state.provision_backoff = 60

                # Prefetch the next candidate now so dwell overlaps a 5-15min
                # Targon spin. Hold/Skip on next iter reuses; Dethrone tears down.
                _start_prefetch(state, slots, queue, challenger, states, stop)

                # chal_slot lifetime is owned by this block: only retained on a
                # successful dethrone (promoted to state.king_slot). Every other
                # exit path — Skip, Hold, exception, shutdown — must tear it down.
                promoted = False
                try:
                    log.info(f"duel: king uid{king.uid} vs chal uid{challenger.uid} (rows={len(rows)})")
                    out = await dwell(chain, king, state.king_slot, challenger, chal_slot,
                                      miners, rows, envs, env_names, store, cfg,
                                      priors, rng, stop,
                                      reign_start_block=state.reign.start_block)
                    duel_rows = rows[-out.rows_added:] if out.rows_added > 0 else []
                    needs_reign = (out.status is DuelStatus.COMPLETED
                                   and out.rows_added > 0
                                   and not out.fit.degenerate)
                    # Falling back to 0 (i.e. k_init) on RPC failure is conservative —
                    # k_init is the strictest threshold, so a chain blip can never
                    # cause a spurious dethrone. Mirrors dwell's same fallback.
                    if needs_reign:
                        try: cur_block = await chain.current_block()
                        except Exception as ex:
                            log.warning(f"current_block failed at verdict: {ex}; using 0")
                            cur_block = state.reign.start_block
                        reign_blocks = max(cur_block - state.reign.start_block, 0)
                    else:
                        reign_blocks = 0

                    king_id = (king.uid, king.revision)
                    chal_id = (challenger.uid, challenger.revision)
                    duel_art_keys = _respondents(miners, rows)
                    verdict = decide(duel_rows, out.fit, out.status,
                                     king=king, chal=challenger, art_keys=duel_art_keys,
                                     reign_blocks=reign_blocks, cfg=cfg)

                    if not isinstance(verdict, Dethrone):
                        # Slot-dead ⇒ tear down the dead slot. Durable-marking
                        # is now driven by the verdict path (Hold over synthetic
                        # rows): if chal really lost, decide() returns Hold and
                        # the standard mark_attempted/durable flow applies. The
                        # chal side does not need an extra mark_durable on
                        # CHAL_SLOT_DEAD — that was asymmetric with king-side
                        # handling and was attributing slot health (validator
                        # signal) as a model-quality decision (mark_durable).
                        if out.status is DuelStatus.KING_SLOT_DEAD:
                            await _drop_king(state, slots)

                    match verdict:
                        case Skip(reason=reason):
                            log.info(f"duel aborted ({reason}, rows={out.rows_added}); skipping verdict")
                            audit(type="duel_aborted", reason=reason,
                                  king=_miner_pl(king), challenger=_miner_pl(challenger))
                            continue
                        case Hold(evidence=ev):
                            _audit_verdict("CHAMPION_HOLDS", king, challenger, ev)
                            continue
                        case Dethrone(new_champion=new_champion, evidence=ev):
                            _audit_verdict("CHALLENGER_WINS", king, challenger, ev)
                            if stop.is_set(): break
                            fresh = {(m.uid, m.revision) for m in await chain.list_miners()}
                            if king_id not in fresh:
                                log.warning(f"verdict skipped: king uid{king.uid}@{king.revision} no longer registered")
                            elif chal_id not in fresh:
                                log.warning(f"verdict skipped: challenger uid{challenger.uid}@{challenger.revision} no longer registered")
                            else:
                                new_reign = Reign(new_champion, await chain.current_block())
                                new_reign.save(reign_state_path)
                                await _drop_king(state, slots)
                                state.king_slot = chal_slot
                                promoted = True
                                state.reign = new_reign
                                states.clear_attempted()
                                await _maybe_publish(state, chain, cfg, reign_state_path,
                                                     challenger.uid, challenger.hotkey)
                                log.info(f"DETHRONE: uid {king.uid} → uid {challenger.uid}")
                finally:
                    if not promoted:
                        await _safe_teardown(slots, chal_slot, "chal-end-of-duel")

            except Exception as e:
                log.error(f"loop iteration: {e}", exc_info=True)
                await _cancellable(asyncio.sleep(60), stop)
    finally:
        log.info("shutdown")
        await _drop_next_chal(state, slots)
        await _drop_king(state, slots)
        seen: set[int] = set()
        for wrapper, _ in envs.values():
            if id(wrapper) in seen: continue
            seen.add(id(wrapper))
            try: await asyncio.wait_for(wrapper.cleanup(), timeout=30.0)
            except Exception as e:
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
