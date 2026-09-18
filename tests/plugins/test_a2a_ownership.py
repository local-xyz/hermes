"""Authenticated task access and conversation continuation through the real HTTP adapter."""

import asyncio
import json
import urllib.request

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.a2a import protocol
from plugins.platforms.a2a.adapter import A2AAdapter


def request(url, method, params, token="alice-token"):
    body = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={
        "Content-Type": "application/json", "Authorization": "Bearer " + token,
    })
    with urllib.request.urlopen(req, timeout=10) as response:
        raw = response.read().decode()
        if response.headers.get_content_type() == "text/event-stream":
            return [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: ")]
        return json.loads(raw)


def adapter(monkeypatch, calls, task_access="peer"):
    monkeypatch.setenv("A2A_PEER_TOKENS", "alice:alice-token,alice:rotated-token,carol:carol-token")
    monkeypatch.setenv("A2A_RATE_LIMIT", "1000")
    monkeypatch.delenv("A2A_PORT", raising=False)
    instance = A2AAdapter(PlatformConfig(enabled=True, extra={"port": 0, **({"task_access": task_access} if task_access else {})}))

    async def reply(event):
        calls.append(event)
        await instance.send(event.source.chat_id, "owner reply", metadata={"notify": True})

    instance.handle_message = reply
    instance._message_handler = reply
    return instance


def test_every_task_surface_checks_authenticated_owner(monkeypatch):
    async def run():
        instance = adapter(monkeypatch, [])
        assert await instance.connect()
        url = f"http://127.0.0.1:{instance._httpd.server_port}"
        try:
            instance.tasks.create("alice-task", "alice-context", "alice")
            instance.tasks.create("carol-task", "carol-context", "carol")
            instance.tasks.create("other-agent-task", "other-context", "alice", "other", "other")
            instance.tasks.set_push_config("alice-task", "https://example.com/original")
            operations = {
                "GetTask": {"id": "alice-task"},
                "CancelTask": {"id": "alice-task"},
                "SubscribeToTask": {"id": "alice-task"},
                "CreateTaskPushNotificationConfig": {"taskId": "alice-task", "config": {"url": "https://example.com/stolen"}},
                "GetTaskPushNotificationConfig": {"taskId": "alice-task"},
                "ListTaskPushNotificationConfigs": {"taskId": "alice-task"},
                "DeleteTaskPushNotificationConfig": {"taskId": "alice-task"},
                "tasks/get": {"id": "alice-task"},
                "tasks/cancel": {"id": "alice-task"},
                "tasks/subscribe": {"id": "alice-task"},
            }
            for method, params in operations.items():
                # Claimed ownership in the body must never override the credential.
                denied = await asyncio.to_thread(request, url, method, params | {"peer": "alice"}, "carol-token")
                assert denied["error"]["code"] == protocol.ERR_TASK_NOT_FOUND, method
            listing = await asyncio.to_thread(request, url, "ListTasks", {"pageSize": 1})
            assert [t["id"] for t in listing["result"]["tasks"]] == ["alice-task"]
            assert listing["result"]["totalSize"] == 1
            assert listing["result"]["nextPageToken"] == ""
            foreign = await asyncio.to_thread(request, url, "GetTask", {"id": "other-agent-task"})
            assert foreign["error"]["code"] == protocol.ERR_TASK_NOT_FOUND
            own = await asyncio.to_thread(request, url, "GetTask", {"id": "alice-task"}, "rotated-token")
            assert own["result"]["id"] == "alice-task"
            cfg = await asyncio.to_thread(request, url, "GetTaskPushNotificationConfig", {"taskId": "alice-task"})
            assert cfg["result"]["pushNotificationConfig"]["url"] == "https://example.com/original"
            canceled = await asyncio.to_thread(request, url, "CancelTask", {"id": "alice-task"})
            assert canceled["result"]["status"]["state"] == protocol.STATE_CANCELED
            subscribed = await asyncio.to_thread(request, url, "SubscribeToTask", {"id": "alice-task"})
            assert subscribed[-1]["result"]["statusUpdate"]["status"]["state"] == protocol.STATE_CANCELED
        finally:
            await instance.disconnect()

    asyncio.run(run())


def test_context_owner_survives_restart_and_credential_rotation(monkeypatch):
    async def run():
        calls = []
        for restart in (False, True):
            instance = adapter(monkeypatch, calls)
            assert await instance.connect()
            url = f"http://127.0.0.1:{instance._httpd.server_port}"
            params = {"message": protocol.text_message(protocol.ROLE_USER, "hello", context_id="private-context")}
            try:
                owner = await asyncio.to_thread(request, url, "SendMessage", params,
                                                "rotated-token" if restart else "alice-token")
                assert owner["result"]["task"]["status"]["state"] == protocol.STATE_COMPLETED
                before = len(calls)
                for method in ("SendMessage", "SendStreamingMessage"):
                    denied = await asyncio.to_thread(request, url, method, params, "carol-token")
                    assert denied["error"]["code"] == protocol.ERR_TASK_NOT_FOUND
                assert len(calls) == before
                # These IDs collide with the existing transcript filename sanitizer.
                collision = {"message": protocol.text_message(protocol.ROLE_USER, "hello", context_id="private/context")}
                await asyncio.to_thread(request, url, "SendMessage", collision)
                alias = {"message": protocol.text_message(protocol.ROLE_USER, "hello", context_id="private.context")}
                denied = await asyncio.to_thread(request, url, "SendMessage", alias, "carol-token")
                assert denied["error"]["code"] == protocol.ERR_TASK_NOT_FOUND
            finally:
                await instance.disconnect()

    asyncio.run(run())


@pytest.mark.parametrize("task_access", [None, "shared"])
def test_shared_policy_preserves_cross_peer_collaboration(monkeypatch, task_access):
    async def run():
        instance = adapter(monkeypatch, [], task_access)
        assert await instance.connect()
        url = f"http://127.0.0.1:{instance._httpd.server_port}"
        try:
            params = {"message": protocol.text_message(protocol.ROLE_USER, "hello", context_id="shared-context")}
            created = await asyncio.to_thread(request, url, "SendMessage", params)
            task = created["result"]["task"]
            read = await asyncio.to_thread(request, url, "GetTask", {"id": task["id"]}, "carol-token")
            assert read["result"]["id"] == task["id"]
            continued = await asyncio.to_thread(request, url, "SendMessage", params, "carol-token")
            assert continued["result"]["task"]["status"]["state"] == protocol.STATE_COMPLETED
            instance.tasks.create("foreign", "foreign-context", "carol", "other", "other")
            denied = await asyncio.to_thread(request, url, "GetTask", {"id": "foreign"}, "carol-token")
            assert denied["error"]["code"] == protocol.ERR_TASK_NOT_FOUND
        finally:
            await instance.disconnect()
    asyncio.run(run())


@pytest.mark.parametrize("mode, tokens, shared", [
    ("typo", "alice:token", ""), ("peer", "", ""),
    ("peer", "", "shared-token"), ("peer", "alice:token", "shared-token"),
])
def test_invalid_or_ambiguous_peer_policy_fails_at_startup(monkeypatch, mode, tokens, shared):
    monkeypatch.setenv("A2A_PEER_TOKENS", tokens)
    monkeypatch.setenv("A2A_BEARER_TOKEN", shared)
    with pytest.raises(ValueError):
        A2AAdapter(PlatformConfig(enabled=True, extra={"task_access": mode}))


def test_context_bindings_are_atomic_and_cannot_alias_transcript_paths(monkeypatch, tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    import threading

    home = tmp_path / "owner-home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    owners = protocol.ContextOwners()
    barrier = threading.Barrier(2)

    def claim(peer):
        barrier.wait(timeout=5)
        return owners.claim("contested", peer, "", "")

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(claim, ("alice", "carol")))
    assert sorted(results) == [False, True]
    winner = "alice" if results[0] else "carol"
    # Reopening the store must preserve the winning identity.
    assert protocol.ContextOwners().claim("contested", winner, "", "")
    assert owners.claim("CaseContext", "alice", "", "")
    assert not owners.claim("casecontext", "carol", "", "")
    protocol.persist_message("legacy", "user", "existing unowned conversation")
    assert not owners.claim("legacy", "alice", "", "")
    # A second profile may reuse the ID; existing instances retain their own home.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "other-home"))
    assert protocol.ContextOwners().claim("contested", "different-peer", "", "")
    assert owners.claim("contested", winner, "", "")
