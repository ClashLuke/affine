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


def _stub_server(reply: str):
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            ok = self.path.rstrip("/") in ("/v1/models", "/models")
            self._send(200 if ok else 404, {"object": "list", "data": [
                {"id": "stub", "object": "model", "owned_by": "stub", "created": 0}]} if ok else {})

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self._send(200, {
                "id": "stub-0", "object": "chat.completion", "model": "stub",
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


async def test_challenger_dethrones(tmp_path, monkeypatch):
    holder: dict = {}
    monkeypatch.setattr(loop_mod, "_install_signal_handlers",
                        lambda stop: holder.setdefault("stop", stop))
    monkeypatch.setenv("AFFINE_BASELINE_MODEL", "stub-champ")
    monkeypatch.setenv("AFFINE_BASELINE_REVISION", "rev-c")

    s_champ = _stub_server("<ANSWER>WRONG</ANSWER>")
    s_chal = _stub_server("<ANSWER>PASS</ANSWER>")
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
            environments=(EnvSpec(name="stub", entrypoint="test_loop:StubEnv",
                                  params={"timeout": 30}),),
            duel_pairs_per_env=16, duel_min_discordant=1,
            dwell_batch=4, provision_timeout=15,
        )
        slots = LocalSlots(f"http://127.0.0.1:{s_champ.server_port}/v1",
                           f"http://127.0.0.1:{s_chal.server_port}/v1")
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
            await asyncio.wait_for(wait(), timeout=60)
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
            assert "dethrone" in [r["status"] for r in s.db.execute(
                "SELECT status FROM duels").fetchall()]
            assert published == [1, 2]
    finally:
        s_champ.shutdown()
        s_chal.shutdown()
