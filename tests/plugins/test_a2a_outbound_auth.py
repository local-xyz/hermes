"""Configured credentials reach a peer through a proxy, never a different origin."""

from contextlib import contextmanager
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading

import pytest

from hermes_constants import get_hermes_home
from plugins.platforms.a2a import protocol, tools


@contextmanager
def server(respond):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            respond(self)

        do_POST = do_GET

        def log_message(self, *_):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def reply(handler, payload):
    body = json.dumps(payload).encode()
    handler.send_response(200)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def configure(url):
    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(json.dumps({"a2a_agents": {"bob": {
        "url": url, "auth": {"type": "bearer", "token": "peer-token"},
        "headers": {"X-Sam-Authentication": "Bearer ${SAM_TEST_TOKEN}",
                    "authorization": "Bearer must-not-override-peer"},
        "capabilities": ["review"],
    }}}))


def test_call_and_fanout_send_both_credentials_from_config(monkeypatch):
    monkeypatch.setenv("SAM_TEST_TOKEN", "mesh-token")
    received = []

    def respond(handler):
        received.append((handler.command, handler.headers.get("Authorization"),
                         handler.headers.get("X-Sam-Authentication")))
        if received[-1][1:] != ("Bearer peer-token", "Bearer mesh-token"):
            handler.send_error(401)
        elif handler.command == "GET":
            reply(handler, {"supportedInterfaces": [{"protocolBinding": "JSONRPC", "url": base + "/rpc"}]})
        else:
            body = json.loads(handler.rfile.read(int(handler.headers["Content-Length"])))
            task = protocol.build_task("task", "context", protocol.STATE_COMPLETED, "authenticated reply")
            reply(handler, protocol.jsonrpc_result(body["id"], protocol.send_message_response(task)))

    with server(respond) as base:
        configure(base)
        assert "authenticated reply" in tools.a2a_call({"agent": "bob", "message": "hello"})
        assert "authenticated reply" in tools.a2a_orchestrate({"capability": "review", "message": "review"})
    assert [r[0] for r in received] == ["GET", "POST", "GET", "POST"]


@pytest.mark.parametrize("escape", ["redirect", "card"])
def test_credentials_never_follow_cross_origin_routes(monkeypatch, escape):
    monkeypatch.setenv("SAM_TEST_TOKEN", "mesh-token")
    leaked = []

    def other(handler):
        leaked.append(dict(handler.headers))
        reply(handler, {})

    with server(other) as destination:
        def respond(handler):
            if escape == "redirect":
                handler.send_response(302)
                handler.send_header("Location", destination)
                handler.end_headers()
            else:
                reply(handler, {"supportedInterfaces": [{"protocolBinding": "JSONRPC", "url": destination}]})

        with server(respond) as base:
            configure(base)
            result = tools.a2a_call({"agent": "bob", "message": "hello"})
    assert not leaked
    assert result.startswith("Error:")
    assert "mesh-token" not in result and "peer-token" not in result


@pytest.mark.parametrize("credential", ["header", "bearer"])
def test_invalid_credentials_are_rejected_without_echoing_secrets(monkeypatch, credential):
    monkeypatch.setenv("SAM_TEST_TOKEN", "mesh-token")
    secret = "private-value\r\ninjected: value"
    contacted = []
    with server(lambda handler: (contacted.append(True), reply(handler, {}))) as base:
        configure(base)
        path = get_hermes_home() / "config.yaml"
        config = json.loads(path.read_text())
        peer = config["a2a_agents"]["bob"]
        if credential == "header":
            peer["headers"]["X-Sam-Authentication"] = secret
        else:
            peer["auth"]["token"] = secret
        path.write_text(json.dumps(config))
        result = tools.a2a_call({"agent": "bob", "message": "hello"})
    assert result.startswith("Error:")
    assert "private-value" not in result
    assert not contacted
