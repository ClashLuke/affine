#!/usr/bin/env python3
"""Minimal OpenAI-compatible stub that passes health checks but returns garbage completions."""
from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread


class StubHandler(BaseHTTPRequestHandler):
    model: str = "stub-model"

    def do_GET(self):
        if self.path.rstrip("/") in ("/v1/models", "/models"):
            self._json(200, {
                "object": "list",
                "data": [{"id": self.model, "object": "model", "created": 0, "owned_by": "stub"}],
            })
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        model = body.get("model", self.model)

        if self.path.rstrip("/") in ("/v1/chat/completions", "/chat/completions"):
            self._json(200, {
                "id": "stub-0",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "I don't know"},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 3, "total_tokens": 3},
            })
        elif self.path.rstrip("/") in ("/v1/completions", "/completions"):
            self._json(200, {
                "id": "stub-0",
                "object": "text_completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{"text": "I don't know", "index": 0, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 3, "total_tokens": 3},
            })
        else:
            self._json(404, {"error": "not found"})

    def _json(self, code: int, obj: dict):
        payload = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):
        pass  # silence per-request logging


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--model", default="stub-model")
    args = p.parse_args()

    StubHandler.model = args.model
    server = HTTPServer(("0.0.0.0", args.port), StubHandler)

    def _stop(sig, _):
        Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    print(f"stub vllm ready on port {args.port} (model={args.model})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
