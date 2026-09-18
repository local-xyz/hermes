"""Durable A2A outcomes are recoverable, never replayed implicitly."""

import asyncio
import threading

from agent.interrupt_scope import track_in_interrupt_scope
from gateway.config import PlatformConfig
from plugins.platforms.a2a import protocol
from plugins.platforms.a2a.adapter import A2AAdapter


def test_restart_preserves_outcomes_and_ownership_and_fails_unfinished(tmp_path):
    path = tmp_path / "tasks.db"
    before = protocol.TaskStore(path)
    before.create("done", "conversation", "alice", "reviewer", "team")
    before.complete("done", protocol.STATE_COMPLETED, "the actual result")
    before.create("running", "conversation", "alice", "reviewer", "team")
    before.set_state("running", protocol.STATE_WORKING)
    before.create("queued", "another", "bob")
    before.create("canceled", "third", "alice")
    before.complete("canceled", protocol.STATE_CANCELED)
    before.set_push_config("done", "https://example.com/callback")

    after = protocol.TaskStore(path)
    assert after.get("done") == before.get("done")
    assert after.to_task(after.get("done")) == before.to_task(before.get("done"))
    assert after.get("done", peer="bob") is None
    assert after.get("done", agent_slug="") is None
    assert after.get("canceled")["state"] == protocol.STATE_CANCELED
    for task_id in ("running", "queued"):
        record = after.get(task_id)
        assert record["state"] == protocol.STATE_FAILED
        assert "restart" in record["reply"]
        assert "retry" in record["reply"]
        assert after.watch(task_id).result(timeout=1) == (record["state"], record["reply"])
        assert after.complete(task_id, protocol.STATE_COMPLETED, "late reply") is None
    again = protocol.TaskStore(path)
    assert again.list(with_total=True) == after.list(with_total=True)


def test_retention_and_push_mutations_survive_restart(tmp_path):
    path = tmp_path / "tasks.db"
    store = protocol.TaskStore(path)
    store._MAX_TERMINAL = 2
    for number in range(3):
        task_id = str(number)
        store.create(task_id, "context", "alice")
        store.complete(task_id, protocol.STATE_COMPLETED, task_id)
        store.set_push_config(task_id, "https://example.com/callback")
    store.delete_push_config("1")
    assert store.pop_push_url("2") == "https://example.com/callback"
    after = protocol.TaskStore(path)
    assert after.get("0") is None
    assert after.list_push_configs("1") == []
    assert after.list_push_configs("2") == []
    assert after.get("2")["reply"] == "2"


def test_cancellation_targets_execution_and_leaves_queued_neighbor_alone(monkeypatch):
    monkeypatch.delenv("A2A_PORT", raising=False)

    async def run():
        instance = A2AAdapter(PlatformConfig(enabled=True, extra={"port": 0}))
        instance._loop = asyncio.get_running_loop()
        started, interrupted, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        calls = []

        async def handler(event):
            calls.append(event.message_id)
            if len(calls) == 1:
                started.set()
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    interrupted.set()
                    raise
            await instance.send(event.source.chat_id, "result " + event.message_id, metadata={"notify": True})

        instance.handle_message = instance._message_handler = handler
        params = {"message": protocol.text_message(protocol.ROLE_USER, "hello", context_id="shared")}
        _, first = await asyncio.to_thread(instance._prepare_task, params, "alice")
        await asyncio.wait_for(started.wait(), 2)
        _, queued = await asyncio.to_thread(instance._prepare_task, params, "alice")
        canceled = await asyncio.to_thread(instance._rpc_tasks_cancel, 1, {"id": queued["task_id"]}, peer="alice")
        assert canceled["result"]["status"]["state"] == protocol.STATE_CANCELED
        assert calls == [first["task_id"]]
        assert not interrupted.is_set()
        _, neighbor = await asyncio.to_thread(instance._prepare_task, params, "alice")
        await asyncio.to_thread(instance._rpc_tasks_cancel, 1, {"id": first["task_id"]}, peer="alice")
        assert interrupted.is_set()
        state, reply = await asyncio.wait_for(asyncio.wrap_future(neighbor["future"]), 2)
        assert (state, reply) == (protocol.STATE_COMPLETED, "result " + neighbor["task_id"])
        assert calls == [first["task_id"], neighbor["task_id"]]
        assert instance.tasks.get(first["task_id"])["state"] == protocol.STATE_CANCELED

    asyncio.run(run())


def test_cancel_before_agent_creation_is_latched_into_worker(monkeypatch):
    monkeypatch.delenv("A2A_PORT", raising=False)

    async def run():
        instance = A2AAdapter(PlatformConfig(enabled=True, extra={"port": 0}))
        instance._loop = asyncio.get_running_loop()
        started, create_agent, stopped, finished = (threading.Event() for _ in range(4))
        effects = []

        class Agent:
            def hard_interrupt(self, message=None, *, tool_reason=None):
                stopped.set()

        def worker():
            started.set()
            try:
                assert create_agent.wait(5)
                with track_in_interrupt_scope(Agent()):
                    if not stopped.is_set():
                        effects.append("model or tool executed")
            finally:
                finished.set()

        async def handler(event):
            await asyncio.to_thread(worker)
            return "done"

        instance.set_message_handler(handler)
        params = {"message": protocol.text_message(protocol.ROLE_USER, "hello", context_id="early")}
        _, pending = await asyncio.to_thread(instance._prepare_task, params, "alice")
        try:
            assert await asyncio.to_thread(started.wait, 3)
            canceled = await asyncio.to_thread(instance._rpc_tasks_cancel, 1, {"id": pending["task_id"]}, peer="alice")
            assert canceled["result"]["status"]["state"] == protocol.STATE_CANCELED
        finally:
            create_agent.set()
            assert await asyncio.to_thread(finished.wait, 3)
        assert stopped.is_set()
        assert effects == []

    asyncio.run(run())


def test_disconnected_stream_does_not_fail_execution(monkeypatch):
    monkeypatch.delenv("A2A_PORT", raising=False)

    class Gone:
        def send_response(self, status):
            raise BrokenPipeError()

    async def run():
        instance = A2AAdapter(PlatformConfig(enabled=True, extra={"port": 0}))
        instance._loop = asyncio.get_running_loop()
        release = asyncio.Event()

        async def handler(event):
            await release.wait()
            await instance.send(event.source.chat_id, "survived disconnect", metadata={"notify": True})

        instance.handle_message = instance._message_handler = handler
        params = {"message": protocol.text_message(protocol.ROLE_USER, "hello", context_id="detached")}
        await asyncio.to_thread(instance._rpc_message_stream, Gone(), 1, params, "alice")
        rec = instance.tasks.list(context_id="detached")[0][0]
        assert rec["state"] not in protocol.TERMINAL_STATES
        watcher = instance.tasks.watch(rec["task_id"])
        release.set()
        assert await asyncio.wait_for(asyncio.wrap_future(watcher), 2) == (protocol.STATE_COMPLETED, "survived disconnect")

    asyncio.run(run())
