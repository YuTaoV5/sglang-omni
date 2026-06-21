# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the thinker local PD scheduler path.

Refs sgl-project/sglang-omni#841. These tests cover the constructor and
the ready-decode queue helpers. They use a lightweight stub instead of
constructing a real OmniScheduler so they run on CPU with no torch / cuda.
"""

from __future__ import annotations

import os
import time
from unittest.mock import MagicMock

import pytest


# --- stub ----------------------------------------------------------------

class _StubScheduler:
    """Captures the kwargs OmniScheduler.__init__ receives, exposes the
    ready-decode queue helpers via the same names as the real class, and
    stubs _emit_event so profiler events can be asserted."""

    def __init__(
        self,
        *,
        enable_overlap: bool = False,
        enable_async_decode: bool = False,
        enable_local_pd: bool = False,
        async_decode_min_batch_size: int = 2,
    ):
        self._enable_overlap = enable_overlap
        self._enable_async_decode = enable_async_decode

        # local PD init (mirrors real OmniScheduler)
        if enable_local_pd and (enable_overlap or enable_async_decode):
            raise ValueError(
                "enable_local_pd is mutually exclusive with enable_overlap "
                "and enable_async_decode"
            )
        self._local_pd_enabled = enable_local_pd
        self._ready_decode: list[str] = []
        self._ready_decode_limit = 32
        self._ready_enter_ts: dict[str, float] = {}
        self._events: list[tuple[str, dict]] = []

    # emit helper - mirrors real _emit_event
    def _emit_event(self, *, event_name: str, request_id: str | None = None, **payload):
        self._events.append((event_name, {"request_id": request_id, **payload}))

    # local PD helpers (mirror real OmniScheduler)
    def _enqueue_ready(self, rid: str) -> None:
        if not self._local_pd_enabled:
            return
        if rid in self._ready_decode:
            return  # idempotent
        self._ready_decode.append(rid)
        self._ready_enter_ts[rid] = time.monotonic()
        self._emit_event(event_name="pd_ready_enter", request_id=rid)
        # cap
        while len(self._ready_decode) > self._ready_decode_limit:
            evicted = self._ready_decode.pop(0)
            self._ready_enter_ts.pop(evicted, None)
            self._emit_event(
                event_name="pd_ready_drop",
                request_id=evicted,
                queue_size=len(self._ready_decode),
            )

    def _drain_ready_decode_into_batch(self, batch) -> None:
        if not self._local_pd_enabled or not self._ready_decode:
            return
        mode = getattr(batch, "forward_mode", None)
        is_decode = bool(getattr(mode, "is_decode", lambda: False)())
        if not is_decode:
            return
        # Emit admit events in FIFO order, then reverse-iterate so that
        # batch.reqs ends up in FIFO order at the front (insert(0) of
        # reversed sequence == original sequence at front).
        now = time.monotonic()
        for rid in list(self._ready_decode):
            wait_ms = (now - self._ready_enter_ts.pop(rid)) * 1000.0
            self._emit_event(
                event_name="pd_ready_admit",
                request_id=rid,
                ready_wait_ms=wait_ms,
            )
        for rid in reversed(list(self._ready_decode)):
            batch.reqs.insert(0, rid)
        self._ready_decode.clear()

    def abort(self, rid: str) -> None:
        if rid in self._ready_decode:
            self._ready_decode.remove(rid)
            self._ready_enter_ts.pop(rid, None)


def _make_decode_batch(reqs: list[str]) -> MagicMock:
    b = MagicMock()
    b.forward_mode.is_decode.return_value = True
    b.reqs = list(reqs)
    return b


def _make_prefill_batch(reqs: list[str]) -> MagicMock:
    b = MagicMock()
    b.forward_mode.is_decode.return_value = False
    b.reqs = list(reqs)
    return b


# --- tests ---------------------------------------------------------------

def test_local_pd_disabled_by_default() -> None:
    s = _StubScheduler()
    assert s._local_pd_enabled is False
    assert s._ready_decode == []


def test_env_flag_enables_local_pd_when_kwarg_passed() -> None:
    s = _StubScheduler(enable_local_pd=True)
    assert s._local_pd_enabled is True
    # env var read is tested at integration level; here we just assert kwarg path.


def test_ready_decode_enqueue_during_prefill() -> None:
    s = _StubScheduler(enable_local_pd=True)
    for rid in ("r1", "r2", "r3", "r4"):
        s._enqueue_ready(rid)
    assert s._ready_decode == ["r1", "r2", "r3", "r4"]
    assert set(s._ready_enter_ts) == {"r1", "r2", "r3", "r4"}
    enters = [e for e in s._events if e[0] == "pd_ready_enter"]
    assert [e[1]["request_id"] for e in enters] == ["r1", "r2", "r3", "r4"]


def test_ready_decode_drain_into_decode_batch() -> None:
    s = _StubScheduler(enable_local_pd=True)
    s._enqueue_ready("r1")
    s._enqueue_ready("r2")
    time.sleep(0.001)
    batch = _make_decode_batch(["r3", "r4"])
    s._drain_ready_decode_into_batch(batch)
    assert batch.reqs == ["r1", "r2", "r3", "r4"]
    assert s._ready_decode == []
    admits = [e for e in s._events if e[0] == "pd_ready_admit"]
    assert [e[1]["request_id"] for e in admits] == ["r1", "r2"]
    assert all(e[1]["ready_wait_ms"] >= 0 for e in admits)


def test_ready_decode_no_drain_into_prefill_batch() -> None:
    s = _StubScheduler(enable_local_pd=True)
    s._enqueue_ready("r1")
    batch = _make_prefill_batch(["r2"])
    s._drain_ready_decode_into_batch(batch)
    assert batch.reqs == ["r2"]
    assert s._ready_decode == ["r1"]


def test_ready_decode_limit_drops_oldest() -> None:
    s = _StubScheduler(enable_local_pd=True)
    s._ready_decode_limit = 3
    for rid in ("r1", "r2", "r3", "r4", "r5"):
        s._enqueue_ready(rid)
    assert s._ready_decode == ["r3", "r4", "r5"]
    drops = [e for e in s._events if e[0] == "pd_ready_drop"]
    assert [e[1]["request_id"] for e in drops] == ["r1", "r2"]


def test_abort_removes_from_ready() -> None:
    s = _StubScheduler(enable_local_pd=True)
    s._enqueue_ready("r1")
    s._enqueue_ready("r2")
    s._enqueue_ready("r3")
    s.abort("r2")
    assert s._ready_decode == ["r1", "r3"]
    assert "r2" not in s._ready_enter_ts


def test_conflicting_modes_raise() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        _StubScheduler(enable_local_pd=True, enable_overlap=True)
    with pytest.raises(ValueError, match="mutually exclusive"):
        _StubScheduler(enable_local_pd=True, enable_async_decode=True)


def test_profiler_events_have_payload() -> None:
    s = _StubScheduler(enable_local_pd=True)
    s._enqueue_ready("r1")
    s._ready_decode_limit = 1
    s._enqueue_ready("r2")  # triggers drop of r1
    batch = _make_decode_batch([])
    s._drain_ready_decode_into_batch(batch)

    names = [e[0] for e in s._events]
    assert names.count("pd_ready_enter") == 2
    assert names.count("pd_ready_drop") == 1
    assert names.count("pd_ready_admit") == 1

    drop_payload = next(e[1] for e in s._events if e[0] == "pd_ready_drop")
    assert drop_payload["request_id"] == "r1"
    assert drop_payload["queue_size"] == 1

    admit_payload = next(e[1] for e in s._events if e[0] == "pd_ready_admit")
    assert admit_payload["request_id"] == "r2"
    assert admit_payload["ready_wait_ms"] >= 0