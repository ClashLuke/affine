import asyncio
import contextlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import affine.loop as loop_mod
from affine.chain import Miner
from affine.config import Config, EnvSpec
from affine.envs._base import ExactAnswerEnv, Spec
from affine.loop import run, static_chain
from affine.store import Store
from affine.vllm import LocalSlots


_SPEC = Spec(title="Stub", rules=("Reply <ANSWER>PASS</ANSWER>",),
             example_challenge="x", example_answer="PASS")


class StubEnv(ExactAnswerEnv):
    env_id = "stub"
    option_keys = frozenset()
    spec = _SPEC

    def _generate(self, params, rng):
        self._target = "PASS"
        return "trivial", {}

    def parse_answer(self, body):
        return body.strip() if body else None

    @classmethod
    def _validate(cls, options):
        return {}


class _StubHandler(BaseHTTPRequestHandler):
    response_text = ""

    def do_GET(self):
        if self.path.rstrip("/") in ("/v1/models", "/models"):
            self._json(200, {"object": "list", "data": [
                {"id": "stub", "object": "model", "owned_by": "stub", "created": 0}]})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if "/chat/completions" in self.path:
            self._json(200, {
                "id": "stub-0", "object": "chat.completion", "model": body.get("model", "stub"),
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": self.response_text}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            })
        else:
            self._json(404, {"error": "not found"})

    def _json(self, code, obj):
        payload = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *_args, **_kwargs):
        pass


def _stub_server(response_text):
    handler = type("H", (_StubHandler,), {"response_text": response_text})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _config(db_path):
    return Config(
        db_path=str(db_path),
        environments=(EnvSpec(name="stub", entrypoint="test_loop:StubEnv",
                              params={"timeout": 30}),),
        duel_pairs_per_env=16,
        duel_min_discordant=1,
        dwell_batch=4,
        provision_timeout=15,
    )


_MINERS = (
    Miner(uid=1, hotkey="hk-champ", model="stub-champ", revision="rev-c", block=0),
    Miner(uid=2, hotkey="hk-chal", model="stub-chal", revision="rev-x", block=10),
)


@pytest.fixture
def captured_stop(monkeypatch):
    captured = {}
    monkeypatch.setattr(loop_mod, "_install_signal_handlers",
                        lambda stop: captured.setdefault("stop", stop))
    yield captured


@contextlib.contextmanager
def _harness(tmp_path, monkeypatch, champ_resp, chal_resp):
    monkeypatch.setenv("AFFINE_BASELINE_MODEL", "stub-champ")
    monkeypatch.setenv("AFFINE_BASELINE_REVISION", "rev-c")
    s_champ = _stub_server(champ_resp)
    s_chal = _stub_server(chal_resp)
    try:
        chain = static_chain(list(_MINERS))
        published: list[int] = []
        original = chain.publish_winner

        async def wrapped(uid, hk):
            published.append(uid)
            return await original(uid, hk)

        chain.publish_winner = wrapped
        slots = LocalSlots(
            f"http://127.0.0.1:{s_champ.server_port}/v1",
            f"http://127.0.0.1:{s_chal.server_port}/v1",
        )
        yield _config(tmp_path / "affine.sqlite3"), chain, slots, published
    finally:
        s_champ.shutdown()
        s_chal.shutdown()


async def _await_state(task, cfg, predicate, timeout=60):
    async def watcher():
        while True:
            if task.done():
                exc = task.exception()
                if exc is not None:
                    raise exc
                return
            store = Store(cfg.db_path)
            try:
                if predicate(store):
                    return
            finally:
                store.close()
            await asyncio.sleep(0.1)
    await asyncio.wait_for(watcher(), timeout=timeout)


async def _shutdown(task, captured_stop):
    stop = captured_stop.get("stop")
    if stop is not None:
        stop.set()
    try:
        await asyncio.wait_for(task, timeout=15)
    except asyncio.CancelledError:
        pass  # _cancellable(asyncio.sleep, stop) raises this on shutdown by design
    except asyncio.TimeoutError:
        task.cancel()
        with contextlib.suppress(BaseException):
            await task


async def test_champion_holds(tmp_path, monkeypatch, captured_stop):
    with _harness(tmp_path, monkeypatch,
                  "<ANSWER>PASS</ANSWER>", "<ANSWER>PASS</ANSWER>") as (cfg, chain, slots, published):
        task = asyncio.create_task(run(cfg, chain, slots=slots))
        try:
            await _await_state(task, cfg, lambda s: any(
                r["status"] in ("hold", "dethrone")
                for r in s.db.execute("SELECT status FROM duels").fetchall()
            ))
        finally:
            await _shutdown(task, captured_stop)
        store = Store(cfg.db_path)
        try:
            champ = store.champion()
            assert champ is not None and champ.uid == 1
            statuses = [r["status"] for r in store.db.execute(
                "SELECT status FROM duels ORDER BY id").fetchall()]
            assert statuses == ["hold"]
            assert published == [1]
        finally:
            store.close()


async def test_challenger_dethrones(tmp_path, monkeypatch, captured_stop):
    with _harness(tmp_path, monkeypatch,
                  "<ANSWER>WRONG</ANSWER>", "<ANSWER>PASS</ANSWER>") as (cfg, chain, slots, published):
        task = asyncio.create_task(run(cfg, chain, slots=slots))
        try:
            await _await_state(task, cfg, lambda s: (
                (c := s.champion()) is not None and c.uid == 2))
        finally:
            await _shutdown(task, captured_stop)
        store = Store(cfg.db_path)
        try:
            champ = store.champion()
            assert champ is not None and champ.uid == 2
            assert champ.model == "stub-chal" and champ.revision == "rev-x"
            statuses = [r["status"] for r in store.db.execute(
                "SELECT status FROM duels ORDER BY id").fetchall()]
            assert "dethrone" in statuses
            assert 1 in published and 2 in published
        finally:
            store.close()
