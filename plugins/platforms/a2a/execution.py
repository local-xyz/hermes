"""Gateway-owned A2A execution, independent of HTTP request lifetimes."""

import asyncio
import contextlib
import logging

from agent.interrupt_scope import InterruptScope, bind_interrupt_scope

from . import protocol

logger = logging.getLogger(__name__)


class A2AExecution:
    async def _execute_task(self, event, pending):
        task_id = pending["task_id"]
        if self.tasks.get(task_id)["state"] in protocol.TERMINAL_STATES:
            self._pop_pending(task_id)
            return
        if self._watchdog_stop.is_set():
            await asyncio.to_thread(self._finalize_task, pending, protocol.STATE_FAILED, "[agent shutting down]")
            return
        session_key = self._event_session_key(event)
        self._executions[task_id] = (session_key, asyncio.current_task())
        lock = self._execution_locks.setdefault(session_key, asyncio.Lock())
        state, reply = protocol.STATE_FAILED, "[agent processing failed]"
        try:
            # A2A tasks are separate invocations, not chat barge-ins. Keep a queued
            # task out of the gateway until its predecessor has fully unwound.
            async with lock:
                processing = None
                interrupt_scope = InterruptScope()
                try:
                    with bind_interrupt_scope(interrupt_scope):
                        await self.handle_message(event)
                        processing = self._session_tasks.get(session_key)
                        if processing is not None:
                            await asyncio.shield(processing)
                    with self._pending_lock:
                        entry = self._pending.get(task_id)
                    if entry and entry[1].done():
                        state, reply = entry[1].result()
                    else:
                        state, reply = protocol.STATE_FAILED, "[gateway produced no task reply]"
                except asyncio.CancelledError:
                    interrupt_scope.cancel("A2A task cancellation requested")
                    if processing is not None:
                        await self.cancel_session_processing(session_key)
                    await asyncio.to_thread(self._finalize_task, pending, protocol.STATE_CANCELED, "")
                    raise
                finally:
                    # Keep the conversation lock until this result has been committed;
                    # otherwise a successor's send could satisfy the previous waiter.
                    if not asyncio.current_task().cancelling():
                        await asyncio.to_thread(self._finalize_task, pending, state, reply)
        except asyncio.CancelledError:
            await asyncio.to_thread(self._finalize_task, pending, protocol.STATE_CANCELED, "")
        except Exception:
            logger.exception("A2A: task execution failed for %s", task_id)
            await asyncio.to_thread(self._finalize_task, pending, protocol.STATE_FAILED, "[agent processing failed]")
        finally:
            self._executions.pop(task_id, None)
            if not any(key == session_key for key, _ in self._executions.values()):
                self._execution_locks.pop(session_key, None)

    async def _cancel_execution(self, task_id):
        rec = self.tasks.get(task_id)
        if rec["state"] in protocol.TERMINAL_STATES:
            return
        execution = self._executions.get(task_id)
        if execution is not None:
            task = execution[1]
            if not task.cancelling():
                task.cancel()
            await asyncio.shield(task)
        else:
            # Cancellation won admission. _execute_task checks this before dispatch.
            self.tasks.complete(task_id, protocol.STATE_CANCELED)
            self._resolve_task(task_id, protocol.STATE_CANCELED, "")

    async def _stop_executions(self):
        tasks = [task for _, task in self._executions.values()]
        for task in tasks:
            if not task.cancelling():
                task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
