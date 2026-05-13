import asyncio
import contextlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import affine.loop as loop_mod
from affine.chain import Miner
from affine.config import Config, EnvSpec
from affine.envs._base import ExactAnswerEnv, Spec
from affine.loop import run, static_chain
from affine.store import Store
from affine.vllm import LocalSlots


class StubEnv(ExactAnswerEnv):
    env_id = "stub"
    option_keys = frozenset()
    spec = Spec(title="Stub", rules=("Reply <ANSWER>PASS</ANSWER>",),
                example_challenge="x", example_answer="PASS")

    def _generate(self, params, rng):
        self._target = "PASS"
        return "trivial", {}

    def parse_answer(self, body):
        return body.strip() if body else None

    @classmethod
    def _validate(cls, options):
        return {}


def _stub_server(pass_rate: float):
    """Stub HTTP server that answers PASS at rate `pass_rate`, WRONG otherwise.

    Mid-range pass rates avoid the σ-boundary explosion in the IRT rating
    contrast: at p ≈ 0 or p ≈ 1, dR/dS = 1/(S(1-S)) is huge, and the
    nuisance covariance times that gradient inflates SE_R. Real validators
    don't see 0%-or-100% miners, so the stub matches reality."""
    counter = {"n": 0}
    lock = threading.Lock()

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            ok = self.path.rstrip("/") in ("/v1/models", "/models")
            self._send(200 if ok else 404, {"object": "list", "data": [
                {"id": "stub", "object": "model", "owned_by": "stub", "created": 0}]} if ok else {})

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            with lock:
                counter["n"] += 1
                idx = counter["n"]
            # Hash idx → uniform draw in [0,1); below pass_rate → PASS.
            import hashlib
            h = int.from_bytes(hashlib.sha256(str(idx).encode()).digest()[:4], "big")
            u = h / 0xFFFFFFFF
            reply = "<ANSWER>PASS</ANSWER>" if u < pass_rate else "<ANSWER>WRONG</ANSWER>"
            self._send(200, {
                "id": f"stub-{idx}", "object": "chat.completion", "model": "stub",
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": reply}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            })

        def _send(self, code, obj):
            payload = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                self.wfile.write(payload)

        def log_message(self, *_a, **_kw): pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _seed_cells(db_path, env_meta, cohort, *, n_per_env=200, context="test-calib"):
    """Inject historical cells for a list of (miner_artifact_id, per-env pass-rates).

    Real validators see backfill-populated cells for current and prior champions.
    The test DB starts empty, so we fixture in cells for both calibration miners
    AND the bootstrap champion (so θ_i is data-fit, not prior-only)."""
    import time
    import uuid
    from affine.store import CellObservation, Store
    store = Store(db_path)
    try:
        for env_name, ev, tsh, gh, sh in env_meta:
            store.upsert_env_state(env_name, "score_active", ev, tsh, gh, sh)
        task_id = 10000
        for mid, rates in cohort:
            for (env_name, ev, tsh, gh, sh), rate in zip(env_meta, rates):
                for i in range(n_per_env):
                    outcome = 1 if i < int(n_per_env * rate) else 0
                    obs = CellObservation(
                        observation_id=uuid.uuid4().hex,
                        miner_artifact_id=mid,
                        env_id=env_name, env_version=ev, task_id=task_id,
                        task_spec_hash=tsh, grader_hash=gh, serving_hash=sh,
                        raw_outcome=outcome, outcome=outcome, gated=0,
                        latency_s=0.01, tokens=1, observed_at=int(time.time()),
                        collection_context=context, sampler_policy_hash="testfixture",
                    )
                    store.add_observation(obs)
                    task_id += 1
    finally:
        store.close()


def _seed_calibration(db_path, env_meta):
    """5 calibration miners spanning 0.10-0.90, slightly different pass rates
    per env (so β_e is identified). Wider spreads (e.g. 0.05-0.95) make
    σ_log_a fit higher and inflate the (μ, a) common-mode covariance."""
    cohort = [
        ("calib0-aaaaaaaaaaaaaaaaaaaaaaaaaaaa", [0.10, 0.20, 0.30]),
        ("calib1-bbbbbbbbbbbbbbbbbbbbbbbbbbbb", [0.25, 0.35, 0.45]),
        ("calib2-cccccccccccccccccccccccccccc", [0.50, 0.55, 0.60]),
        ("calib3-dddddddddddddddddddddddddddd", [0.65, 0.70, 0.75]),
        ("calib4-eeeeeeeeeeeeeeeeeeeeeeeeeeee", [0.80, 0.85, 0.90]),
    ]
    _seed_cells(db_path, env_meta, cohort)


async def test_challenger_dethrones(tmp_path, monkeypatch):
    holder: dict = {}
    monkeypatch.setattr(loop_mod, "_install_signal_handlers",
                        lambda stop: holder.setdefault("stop", stop))
    monkeypatch.setenv("AFFINE_BASELINE_MODEL", "stub-champ")
    monkeypatch.setenv("AFFINE_BASELINE_REVISION", "rev-c")

    # Mid-range pass rates: champion 30%, challenger 80%. Clear ΔR signal,
    # no σ-boundary saturation (which would inflate nuisance SE).
    s_champ = _stub_server(pass_rate=0.30)
    s_chal = _stub_server(pass_rate=0.80)
    try:
        chain = static_chain([
            Miner(uid=1, hotkey="hk-champ", model="stub-champ", revision="rev-c", block=0),
            Miner(uid=2, hotkey="hk-chal", model="stub-chal", revision="rev-x", block=10),
        ])
        published: list[int] = []
        original = chain.publish_winner
        async def wrapped(uid, hk):
            published.append(uid)
            return await original(uid, hk)
        chain.publish_winner = wrapped

        cfg = Config(
            db_path=str(tmp_path / "affine.sqlite3"),
            environments=(
                EnvSpec(name="stub_a", entrypoint="test_loop:StubEnv",
                        params={"timeout": 30}),
                EnvSpec(name="stub_b", entrypoint="test_loop:StubEnv",
                        params={"timeout": 30}),
                EnvSpec(name="stub_c", entrypoint="test_loop:StubEnv",
                        params={"timeout": 30}),
            ),
            irt_look_interval_cells=6, irt_look_max=20,
            irt_n_min_per_env=1, champion_se_theta_max=2.0,
            irt_alpha_dethrone=0.025, irt_beta_hold=0.10,
            delta_theta=0.3, provision_timeout=15,
            calibration_min_miners=3, calibration_min_cells_per_env=30,
        )
        # Pre-populate calibration cells across the 3 envs so (μ, β, a) are
        # jointly identified. With 1 env, μ and θ are confounded and
        # nuisance SE doesn't shrink with data; multi-env breaks that.
        from affine.score import env_versioning
        from affine.store import artifact_id
        env_meta = []
        for spec in cfg.environments:
            ev, tsh, gh = env_versioning(spec)
            env_meta.append((spec.name, ev, tsh, gh, "default"))
        _seed_calibration(cfg.db_path, env_meta)
        # Seed historical cells for the champion at the stub's actual pass rate
        # (~30%), so θ_i is data-fit at duel start. Real validators see champion
        # cells from backfill; the test fixture replicates that. Note: stub-chal
        # has no historical cells — those accumulate from the duel itself.
        champ_art = artifact_id("stub-champ", "rev-c")
        _seed_cells(cfg.db_path, env_meta, [(champ_art, [0.30, 0.30, 0.30])],
                    n_per_env=120, context="test-king-backfill")
        slots = [LocalSlots(f"http://127.0.0.1:{s_champ.server_port}/v1",
                            f"http://127.0.0.1:{s_chal.server_port}/v1")]
        task = asyncio.create_task(run(cfg, chain, slots=slots))

        async def wait():
            while not task.done():
                with contextlib.closing(Store(cfg.db_path)) as s:
                    c = s.champion()
                    if c is not None and c.uid == 2:
                        return
                await asyncio.sleep(0.1)
            if exc := task.exception():
                raise exc

        try:
            await asyncio.wait_for(wait(), timeout=120)
        finally:
            if stop := holder.get("stop"):
                stop.set()
            with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError):
                await asyncio.wait_for(task, timeout=15)
            if not task.done():
                task.cancel()
                with contextlib.suppress(BaseException):
                    await task

        with contextlib.closing(Store(cfg.db_path)) as s:
            c = s.champion()
            assert c.uid == 2 and c.model == "stub-chal" and c.revision == "rev-x"
            statuses = [r["status"] for r in s.db.execute(
                "SELECT status FROM duels").fetchall()]
            assert "dethrone" in statuses, f"expected dethrone; got {statuses}"
            assert published == [1, 2], f"expected publish [1, 2]; got {published}"
    finally:
        s_champ.shutdown()
        s_chal.shutdown()


async def test_calibration_needed_does_not_provision_or_attempt(tmp_path, monkeypatch):
    """Livelock guard: with insufficient calibration, the loop must NOT
    re-provision the same challenger every cycle, and must NOT mark them
    attempted. The pre-provision sufficiency gate handles this; this test
    verifies it.

    Setup: empty calibration cohort. The challenger should never get a slot
    provisioned, no duel rows created, and no attempted_artifacts entry.
    """
    holder: dict = {}
    sleeps: list[float] = []
    real_sleep = asyncio.sleep

    monkeypatch.setattr(loop_mod, "_install_signal_handlers",
                        lambda stop: holder.setdefault("stop", stop))
    monkeypatch.setenv("AFFINE_BASELINE_MODEL", "stub-champ")
    monkeypatch.setenv("AFFINE_BASELINE_REVISION", "rev-c")

    # Patch asyncio.sleep so we can detect the long backoff and short-circuit.
    async def fake_sleep(t):
        sleeps.append(float(t))
        if t >= 200:
            stop = holder.get("stop")
            if stop is not None:
                stop.set()
            return
        await real_sleep(min(t, 0.05))
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    s_champ = _stub_server(pass_rate=0.30)
    s_chal = _stub_server(pass_rate=0.80)
    try:
        chain = static_chain([
            Miner(uid=1, hotkey="hk-champ", model="stub-champ", revision="rev-c", block=0),
            Miner(uid=2, hotkey="hk-chal", model="stub-chal", revision="rev-x", block=10),
        ])
        cfg = Config(
            db_path=str(tmp_path / "affine.sqlite3"),
            environments=(
                EnvSpec(name="stub_a", entrypoint="test_loop:StubEnv",
                        params={"timeout": 30}),
            ),
            irt_look_interval_cells=4, irt_look_max=10,
            irt_n_min_per_env=1, champion_se_theta_max=2.0,
            irt_alpha_dethrone=0.025, irt_beta_hold=0.10,
            delta_theta=0.3, provision_timeout=15,
            # Demand more cohort than the test setup has → trigger the gate.
            calibration_min_miners=10, calibration_min_cells_per_env=1000,
        )
        # No calibration cells at all.
        from affine.score import env_versioning
        env_meta = []
        for spec in cfg.environments:
            ev, tsh, gh = env_versioning(spec)
            env_meta.append((spec.name, ev, tsh, gh, "default"))

        slots = [LocalSlots(f"http://127.0.0.1:{s_champ.server_port}/v1",
                            f"http://127.0.0.1:{s_chal.server_port}/v1")]
        # Track provision calls — if the gate works, only the king gets one.
        original_provision = LocalSlots.provision
        provision_calls: list[str] = []

        async def tracked_provision(self, model, revision, **kwargs):
            provision_calls.append(model)
            return await original_provision(self, model, revision, **kwargs)
        monkeypatch.setattr(LocalSlots, "provision", tracked_provision)

        task = asyncio.create_task(run(cfg, chain, slots=slots))
        try:
            await asyncio.wait_for(task, timeout=30)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass

        with contextlib.closing(Store(cfg.db_path)) as s:
            duels = s.db.execute("SELECT * FROM duels").fetchall()
            attempted = s.db.execute("SELECT * FROM attempted_artifacts").fetchall()

        # No duel for the challenger ever started.
        chal_duels = [d for d in duels if d["challenger_model"] == "stub-chal"]
        assert chal_duels == [], f"challenger should not have started a duel; got {chal_duels}"
        # Challenger NOT marked attempted.
        chal_attempted = [a for a in attempted if a["model"] == "stub-chal"]
        assert chal_attempted == [], f"challenger should not be marked attempted; got {chal_attempted}"
        # Challenger slot was NEVER provisioned.
        assert "stub-chal" not in provision_calls, (
            f"challenger slot must not be provisioned under calibration_needed; "
            f"got provision_calls={provision_calls}"
        )
        # The long backoff should have been hit at least once.
        assert any(t >= 200 for t in sleeps), \
            f"expected long backoff sleep (≥200s); got {sleeps}"
    finally:
        s_champ.shutdown()
        s_chal.shutdown()
