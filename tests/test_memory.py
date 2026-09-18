from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest
from memcode_sdk import HybridSearchResult, SourceRecord
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    LLMContextAssistantTurnFrame,
    LLMContextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.tests.utils import SleepFrame, run_test

import pipecat_memcode.memory as memory_module
from pipecat_memcode import MemcodeMemoryConfig, MemcodeMemoryService


class _TaskManager:
    def create_task(self, coroutine, name, context=None):
        return asyncio.create_task(coroutine, name=name, context=context)

    async def cancel_task(self, task, cancel_timeout=1.0):
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    def get_event_loop(self):
        return asyncio.get_running_loop()


class _ToolRoundTripLLM(FrameProcessor):
    """Test double that consumes recalled context and emits two finalized responses."""

    def __init__(self):
        super().__init__()
        self.observed_messages: list[dict[str, Any]] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, LLMContextFrame):
            self.observed_messages = list(frame.context.get_messages())
            await self.push_frame(
                LLMContextAssistantTurnFrame(text="Let me check Memcode.", timestamp="t1"),
                direction,
            )
            await self.push_frame(
                LLMContextAssistantTurnFrame(text="Your launch is Friday.", timestamp="t2"),
                direction,
            )
            return
        await self.push_frame(frame, direction)


class _RecordingSink(FrameProcessor):
    def __init__(self):
        super().__init__(enable_direct_mode=True)
        self.frames: list[Frame] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        self.frames.append(frame)


@dataclass
class _FakeClient:
    search_result: HybridSearchResult = field(default_factory=HybridSearchResult)
    search_delay: float = 0.0
    ingest_delay: float = 0.0
    close_delay: float = 0.0

    def __post_init__(self):
        self.search_calls: list[dict[str, Any]] = []
        self.ingest_calls: list[dict[str, Any]] = []
        self.close_calls = 0
        self.ingest_cancelled = False
        self.ingest_started = asyncio.Event()
        self.search_in_flight = False
        self.close_while_searching = False

    async def search_v2(self, **kwargs):
        self.search_calls.append(kwargs)
        self.search_in_flight = True
        try:
            if self.search_delay:
                await asyncio.sleep(self.search_delay)
            return self.search_result
        finally:
            self.search_in_flight = False

    async def ingest_v2(self, **kwargs):
        self.ingest_calls.append(kwargs)
        self.ingest_started.set()
        try:
            if self.ingest_delay:
                await asyncio.sleep(self.ingest_delay)
        except asyncio.CancelledError:
            self.ingest_cancelled = True
            raise
        return {"job_id": "job-1", "status": "pending"}

    async def close(self):
        self.close_calls += 1
        self.close_while_searching = self.close_while_searching or self.search_in_flight
        if self.close_delay:
            await asyncio.sleep(self.close_delay)


def _service(client: _FakeClient, **config_overrides) -> MemcodeMemoryService:
    return MemcodeMemoryService(
        client=client,  # type: ignore[arg-type]
        session_id="call-123",
        config=MemcodeMemoryConfig(**config_overrides),
    )


def _wire(processor):
    processor._task_manager = _TaskManager()
    return processor


async def _claim_shared_state(*processors) -> None:
    """Mirror successful Pipecat setup for direct processor lifecycle tests."""

    for processor in processors:
        await processor._state.acquire_processor()
        processor._state_acquired = True


@pytest.mark.asyncio
async def test_recall_uses_search_v2_once_and_replaces_its_injected_block():
    result = HybridSearchResult(
        results=[
            SourceRecord(domain="profile", content="User likes tea", score=0.9),
            SourceRecord(domain="summary", content="User likes tea", score=0.8),
            SourceRecord(
                domain="temporal",
                content="A" * 1000,
                score=0.7,
            ),
        ]
    )
    client = _FakeClient(search_result=result)
    service = _service(client, max_context_characters=300)
    recall = _wire(service.recall_processor())
    context = LLMContext(
        [
            {"role": "developer", "content": "Be helpful."},
            {"role": "user", "content": "What do I drink?"},
        ]
    )
    frame = LLMContextFrame(context=context)

    await recall.process_frame(frame, FrameDirection.DOWNSTREAM)
    await recall.process_frame(frame, FrameDirection.DOWNSTREAM)

    assert len(client.search_calls) == 1
    assert client.search_calls[0] == {
        "query": "What do I drink?",
        "top_k": 5,
        "minimum_score": 0.0,
        "search_mode": "default",
        "mode": "memories",
        "include_original_chunks": False,
    }
    messages = context.get_messages()
    injected = [
        message
        for message in messages
        if isinstance(message, dict)
        and isinstance(message.get("content"), str)
        and message["content"].startswith("[Memcode memory context")
    ]
    assert len(injected) == 1
    assert len(injected[0]["content"]) <= 300
    assert injected[0]["role"] == "developer"
    assert messages.index(injected[0]) < next(
        index for index, message in enumerate(messages) if message.get("role") == "user"
    )


@pytest.mark.asyncio
async def test_user_text_matching_internal_marker_is_never_removed():
    client = _FakeClient()
    service = _service(client)
    recall = _wire(service.recall_processor())
    user_text = "[Memcode memory context - automatically injected] is user text"
    context = LLMContext([{"role": "user", "content": user_text}])

    await recall.process_frame(LLMContextFrame(context=context), FrameDirection.DOWNSTREAM)

    assert context.get_messages() == [{"role": "user", "content": user_text}]
    assert client.search_calls[0]["query"] == user_text


@pytest.mark.asyncio
async def test_context_ending_in_assistant_does_not_requeue_prior_user_turn():
    client = _FakeClient()
    service = _service(client)
    recall = _wire(service.recall_processor())
    context = LLMContext(
        [
            {
                "role": "developer",
                "content": (
                    "[Memcode memory context - automatically injected]\nstale integration data"
                ),
            },
            {"role": "user", "content": "Already answered"},
            {"role": "assistant", "content": "The existing answer"},
        ]
    )

    await recall.process_frame(LLMContextFrame(context=context), FrameDirection.DOWNSTREAM)

    assert client.search_calls == []
    assert not service._state.pending_turns
    assert all(
        "stale integration data" not in str(message.get("content"))
        for message in context.get_messages()
    )


@pytest.mark.asyncio
async def test_tool_call_assistant_message_keeps_user_turn_active():
    client = _FakeClient()
    service = _service(client)
    recall = _wire(service.recall_processor())
    context = LLMContext(
        [
            {"role": "user", "content": "Check my saved preference"},
            {
                "role": "assistant",
                "content": "I will check.",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "done"},
        ]
    )

    await recall.process_frame(LLMContextFrame(context=context), FrameDirection.DOWNSTREAM)

    assert client.search_calls[0]["query"] == "Check my saved preference"
    assert service._state.pending_turns


@pytest.mark.asyncio
async def test_capture_only_persists_finalized_delta_and_never_injected_memory():
    client = _FakeClient(
        search_result=HybridSearchResult(
            results=[SourceRecord(domain="profile", content="SECRET MEMORY", score=1.0)]
        )
    )
    service = _service(client)
    recall = _wire(service.recall_processor())
    capture = _wire(service.capture_processor())
    context = LLMContext([{"role": "user", "content": "My clean user turn"}])

    await recall.process_frame(LLMContextFrame(context=context), FrameDirection.DOWNSTREAM)
    await capture.process_frame(
        InterimTranscriptionFrame(
            text="partial words",
            user_id="transport-user",
            timestamp="2026-09-17T10:00:00Z",
        ),
        FrameDirection.DOWNSTREAM,
    )
    assert client.ingest_calls == []

    assistant = LLMContextAssistantTurnFrame(
        text="My clean assistant turn",
        timestamp=service._state.pending_turns[0].started_at.isoformat(),
    )
    await capture.process_frame(assistant, FrameDirection.DOWNSTREAM)
    await capture.process_frame(assistant, FrameDirection.DOWNSTREAM)
    assert client.ingest_calls == []
    await capture.process_frame(EndFrame(), FrameDirection.DOWNSTREAM)

    assert len(client.ingest_calls) == 1
    assert not service._state.pending_turns
    call = client.ingest_calls[0]
    assert call["user_query"] == "My clean user turn"
    assert call["agent_response"] == "My clean assistant turn"
    assert "SECRET MEMORY" not in call["user_query"]
    assert "user_id" not in call
    assert call["idempotency_key"].startswith("pcmem_")
    assert len(call["idempotency_key"]) == len("pcmem_") + 64


@pytest.mark.asyncio
async def test_tool_preamble_and_final_answer_are_one_terminal_ingest():
    client = _FakeClient()
    service = _service(client)
    recall = _wire(service.recall_processor())
    capture = _wire(service.capture_processor())
    context = LLMContext([{"role": "user", "content": "What is my launch date?"}])

    await recall.process_frame(LLMContextFrame(context=context), FrameDirection.DOWNSTREAM)
    await capture.process_frame(
        LLMContextAssistantTurnFrame(text="Let me check that.", timestamp="t1"),
        FrameDirection.DOWNSTREAM,
    )
    await capture.process_frame(
        LLMContextAssistantTurnFrame(text="Your launch is Friday.", timestamp="t2"),
        FrameDirection.DOWNSTREAM,
    )

    assert client.ingest_calls == []
    await capture.process_frame(EndFrame(), FrameDirection.DOWNSTREAM)

    assert len(client.ingest_calls) == 1
    assert client.ingest_calls[0]["agent_response"] == (
        "Let me check that.\n\nYour launch is Friday."
    )


@pytest.mark.asyncio
async def test_real_pipeline_orders_recall_before_one_tool_round_trip_capture():
    client = _FakeClient(
        search_result=HybridSearchResult(
            results=[SourceRecord(domain="profile", content="Launch date is Friday", score=0.9)]
        )
    )
    service = _service(client)
    llm = _ToolRoundTripLLM()
    pipeline = Pipeline(
        [
            service.recall_processor(),
            llm,
            service.capture_processor(),
        ]
    )
    context = LLMContext([{"role": "user", "content": "When is my launch?"}])

    down_frames, _ = await run_test(
        pipeline,
        frames_to_send=[LLMContextFrame(context=context)],
        expected_down_frames=[
            LLMContextAssistantTurnFrame,
            LLMContextAssistantTurnFrame,
        ],
    )

    assert [frame.text for frame in down_frames] == [
        "Let me check Memcode.",
        "Your launch is Friday.",
    ]
    assert any(
        message.get("role") == "developer"
        and "Launch date is Friday" in str(message.get("content"))
        for message in llm.observed_messages
    )
    assert len(client.ingest_calls) == 1
    assert client.ingest_calls[0]["user_query"] == "When is my launch?"
    assert client.ingest_calls[0]["agent_response"] == (
        "Let me check Memcode.\n\nYour launch is Friday."
    )


@pytest.mark.asyncio
async def test_concurrent_pipeline_cleanup_quiesces_inflight_recall_before_client_close():
    client = _FakeClient(search_delay=1.0)
    service = MemcodeMemoryService(
        client=client,  # type: ignore[arg-type]
        close_client=True,
        session_id="call-123",
        config=MemcodeMemoryConfig(shutdown_timeout_seconds=0.05),
    )
    pipeline = Pipeline(
        [
            service.recall_processor(),
            service.capture_processor(),
        ]
    )
    context = LLMContext([{"role": "user", "content": "cancel during recall"}])

    await asyncio.wait_for(
        run_test(
            pipeline,
            frames_to_send=[
                LLMContextFrame(context=context),
                SleepFrame(sleep=0.02),
                CancelFrame(reason="test concurrent cleanup"),
            ],
            send_end_frame=False,
        ),
        timeout=0.5,
    )

    assert client.search_calls
    assert client.close_calls == 1
    assert not client.close_while_searching


@pytest.mark.asyncio
async def test_single_processor_pipeline_cleanup_releases_owned_client():
    client = _FakeClient()
    service = MemcodeMemoryService(
        client=client,  # type: ignore[arg-type]
        close_client=True,
        session_id="call-123",
    )

    await asyncio.wait_for(
        run_test(service.recall_processor(), frames_to_send=[]),
        timeout=0.2,
    )

    assert client.close_calls == 1


@pytest.mark.asyncio
async def test_next_user_turn_finalizes_prior_accumulated_assistant_segments():
    client = _FakeClient()
    service = _service(client)
    recall = _wire(service.recall_processor())
    capture = _wire(service.capture_processor())

    first = LLMContext([{"role": "user", "content": "Question one"}])
    await recall.process_frame(LLMContextFrame(context=first), FrameDirection.DOWNSTREAM)
    await capture.process_frame(
        LLMContextAssistantTurnFrame(text="Tool preamble.", timestamp="t1"),
        FrameDirection.DOWNSTREAM,
    )
    await capture.process_frame(
        LLMContextAssistantTurnFrame(text="Final answer one.", timestamp="t2"),
        FrameDirection.DOWNSTREAM,
    )

    second = LLMContext(
        [
            {"role": "user", "content": "Question one"},
            {"role": "assistant", "content": "Final answer one."},
            {"role": "user", "content": "Question two"},
        ]
    )
    await recall.process_frame(LLMContextFrame(context=second), FrameDirection.DOWNSTREAM)
    assert service._state.pending_turns[-1].user_text == "Question two"

    await capture.process_frame(EndFrame(), FrameDirection.DOWNSTREAM)

    assert len(client.ingest_calls) == 1
    assert client.ingest_calls[0]["user_query"] == "Question one"
    assert client.ingest_calls[0]["agent_response"] == ("Tool preamble.\n\nFinal answer one.")


@pytest.mark.asyncio
async def test_idempotency_key_is_stable_for_same_session_and_turn():
    keys = []
    for _ in range(2):
        client = _FakeClient()
        service = _service(client)
        recall = _wire(service.recall_processor())
        capture = _wire(service.capture_processor())
        context = LLMContext([{"role": "user", "content": "Remember blue"}])
        await recall.process_frame(LLMContextFrame(context=context), FrameDirection.DOWNSTREAM)
        await capture.process_frame(
            LLMContextAssistantTurnFrame(
                text="I will remember blue.",
                timestamp=service._state.pending_turns[0].started_at.isoformat(),
            ),
            FrameDirection.DOWNSTREAM,
        )
        await capture.process_frame(EndFrame(), FrameDirection.DOWNSTREAM)
        keys.append(client.ingest_calls[0]["idempotency_key"])

    assert keys[0] == keys[1]


@pytest.mark.asyncio
async def test_speculative_context_and_recall_timeout_fail_open():
    client = _FakeClient(search_delay=0.05)
    service = _service(client, search_timeout_seconds=0.001)
    recall = _wire(service.recall_processor())
    capture = _wire(service.capture_processor())

    speculative_context = LLMContext([{"role": "user", "content": "not final"}])
    await recall.process_frame(
        LLMContextFrame(context=speculative_context, speculation=True),
        FrameDirection.DOWNSTREAM,
    )
    assert client.search_calls == []

    context = LLMContext([{"role": "user", "content": "final query"}])
    await recall.process_frame(LLMContextFrame(context=context), FrameDirection.DOWNSTREAM)
    assert len(client.search_calls) == 1
    assert context.get_messages() == [{"role": "user", "content": "final query"}]

    # A speculative turn never creates a pending capture pair.
    other = _service(_FakeClient())
    other_recall = _wire(other.recall_processor())
    other_capture = _wire(other.capture_processor())
    await other_recall.process_frame(
        LLMContextFrame(context=speculative_context, speculation=True),
        FrameDirection.DOWNSTREAM,
    )
    await other_capture.process_frame(
        LLMContextAssistantTurnFrame(text="provisional", timestamp="now"),
        FrameDirection.DOWNSTREAM,
    )
    await other_capture.process_frame(EndFrame(), FrameDirection.DOWNSTREAM)
    assert other._state.client.ingest_calls == []

    await capture.process_frame(EndFrame(), FrameDirection.DOWNSTREAM)


@pytest.mark.asyncio
async def test_cancelled_recall_discards_unanswered_pending_turn():
    client = _FakeClient(search_delay=1.0)
    service = _service(client)
    recall = _wire(service.recall_processor())
    context = LLMContext([{"role": "user", "content": "cancel this recall"}])

    task = asyncio.create_task(
        recall.process_frame(LLMContextFrame(context=context), FrameDirection.DOWNSTREAM)
    )
    await asyncio.sleep(0)
    assert client.search_calls
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert not service._state.pending_turns


@pytest.mark.asyncio
async def test_cancel_frame_discards_active_turn_cancels_writes_and_forwards_promptly():
    client = _FakeClient(ingest_delay=1.0)
    service = _service(
        client,
        ingest_timeout_seconds=2.0,
        shutdown_timeout_seconds=1.0,
    )
    recall = _wire(service.recall_processor())
    capture = _wire(service.capture_processor())
    await _claim_shared_state(recall, capture)
    sink = _RecordingSink()
    capture.link(sink)

    first = LLMContext([{"role": "user", "content": "completed user"}])
    await recall.process_frame(LLMContextFrame(context=first), FrameDirection.DOWNSTREAM)
    await capture.process_frame(
        LLMContextAssistantTurnFrame(text="completed assistant", timestamp="t1"),
        FrameDirection.DOWNSTREAM,
    )
    second = LLMContext(
        [
            {"role": "user", "content": "completed user"},
            {"role": "assistant", "content": "completed assistant"},
            {"role": "user", "content": "cancelled user"},
        ]
    )
    await recall.process_frame(LLMContextFrame(context=second), FrameDirection.DOWNSTREAM)
    second_started_at = service._state.pending_turns[-1].started_at.isoformat()
    await capture.process_frame(
        LLMContextAssistantTurnFrame(
            text="cancelled partial assistant", timestamp=second_started_at
        ),
        FrameDirection.DOWNSTREAM,
    )
    await asyncio.sleep(0)
    assert len(client.ingest_calls) == 1

    loop = asyncio.get_running_loop()
    started = loop.time()
    await asyncio.wait_for(
        capture.process_frame(CancelFrame(reason="test"), FrameDirection.DOWNSTREAM),
        timeout=0.2,
    )
    elapsed = loop.time() - started
    await asyncio.sleep(0)

    assert elapsed < 0.1
    assert isinstance(sink.frames[-1], CancelFrame)
    assert client.ingest_cancelled
    assert not service._state.pending_turns
    await asyncio.gather(recall.cleanup(), capture.cleanup())
    assert len(client.ingest_calls) == 1
    assert client.ingest_calls[0]["user_query"] == "completed user"


@pytest.mark.asyncio
async def test_interruption_discards_partial_turn_but_later_complete_turn_ingests():
    client = _FakeClient()
    service = _service(client)
    recall = _wire(service.recall_processor())
    capture = _wire(service.capture_processor())

    first = LLMContext([{"role": "user", "content": "interrupted user"}])
    await recall.process_frame(LLMContextFrame(context=first), FrameDirection.DOWNSTREAM)
    await capture.process_frame(
        LLMContextAssistantTurnFrame(text="interrupted partial", timestamp="t1"),
        FrameDirection.DOWNSTREAM,
    )
    second = LLMContext(
        [
            {"role": "user", "content": "interrupted user"},
            {"role": "assistant", "content": "interrupted partial"},
            {"role": "user", "content": "complete user"},
        ]
    )
    await recall.process_frame(LLMContextFrame(context=second), FrameDirection.DOWNSTREAM)
    await capture.process_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)

    assert len(service._state.pending_turns) == 1
    assert service._state.pending_turns[0].user_text == "complete user"
    await capture.process_frame(
        LLMContextAssistantTurnFrame(text="complete assistant", timestamp="t2"),
        FrameDirection.DOWNSTREAM,
    )
    await capture.process_frame(EndFrame(), FrameDirection.DOWNSTREAM)

    assert len(client.ingest_calls) == 1
    assert client.ingest_calls[0]["user_query"] == "complete user"
    assert client.ingest_calls[0]["agent_response"] == "complete assistant"


@pytest.mark.asyncio
async def test_delayed_assistant_timestamp_never_cross_pairs_with_next_user():
    client = _FakeClient()
    service = _service(client)
    recall = _wire(service.recall_processor())
    capture = _wire(service.capture_processor())

    first = LLMContext([{"role": "user", "content": "user one"}])
    await recall.process_frame(LLMContextFrame(context=first), FrameDirection.DOWNSTREAM)
    first_started_at = service._state.pending_turns[0].started_at.isoformat()

    second = LLMContext(
        [
            {"role": "user", "content": "user one"},
            {"role": "user", "content": "user two"},
        ]
    )
    await recall.process_frame(LLMContextFrame(context=second), FrameDirection.DOWNSTREAM)
    await capture.process_frame(
        LLMContextAssistantTurnFrame(text="delayed answer one", timestamp=first_started_at),
        FrameDirection.DOWNSTREAM,
    )
    await capture.process_frame(EndFrame(), FrameDirection.DOWNSTREAM)

    assert len(client.ingest_calls) == 1
    assert client.ingest_calls[0]["user_query"] == "user one"
    assert client.ingest_calls[0]["agent_response"] == "delayed answer one"


@pytest.mark.asyncio
async def test_replyless_older_generation_does_not_capture_newer_answer():
    client = _FakeClient()
    service = _service(client)
    recall = _wire(service.recall_processor())
    capture = _wire(service.capture_processor())

    await recall.process_frame(
        LLMContextFrame(context=LLMContext([{"role": "user", "content": "no reply"}])),
        FrameDirection.DOWNSTREAM,
    )
    second = LLMContext(
        [
            {"role": "user", "content": "no reply"},
            {"role": "user", "content": "new user"},
        ]
    )
    await recall.process_frame(LLMContextFrame(context=second), FrameDirection.DOWNSTREAM)
    second_started_at = service._state.pending_turns[-1].started_at.isoformat()
    await capture.process_frame(
        LLMContextAssistantTurnFrame(text="new answer", timestamp=second_started_at),
        FrameDirection.DOWNSTREAM,
    )
    await capture.process_frame(EndFrame(), FrameDirection.DOWNSTREAM)

    assert len(client.ingest_calls) == 1
    assert client.ingest_calls[0]["user_query"] == "new user"
    assert client.ingest_calls[0]["agent_response"] == "new answer"


@pytest.mark.asyncio
async def test_cleanup_without_terminal_frame_flushes_staged_turn_once():
    client = _FakeClient()
    service = _service(client)
    recall = _wire(service.recall_processor())
    capture = _wire(service.capture_processor())
    await _claim_shared_state(recall, capture)
    context = LLMContext([{"role": "user", "content": "flush me"}])

    await recall.process_frame(LLMContextFrame(context=context), FrameDirection.DOWNSTREAM)
    await capture.process_frame(
        LLMContextAssistantTurnFrame(text="flushed answer", timestamp="now"),
        FrameDirection.DOWNSTREAM,
    )
    await asyncio.gather(recall.cleanup(), capture.cleanup())
    await asyncio.gather(recall.cleanup(), capture.cleanup())

    assert len(client.ingest_calls) == 1
    assert client.ingest_calls[0]["agent_response"] == "flushed answer"


@pytest.mark.asyncio
async def test_cleanup_fallback_ingest_uses_shutdown_not_ingest_timeout():
    client = _FakeClient(ingest_delay=1.0)
    service = MemcodeMemoryService(
        client=client,  # type: ignore[arg-type]
        close_client=True,
        session_id="call-123",
        config=MemcodeMemoryConfig(
            ingest_timeout_seconds=2.0,
            shutdown_timeout_seconds=0.02,
        ),
    )
    recall = _wire(service.recall_processor())
    capture = _wire(service.capture_processor())
    await _claim_shared_state(recall, capture)
    context = LLMContext([{"role": "user", "content": "bounded cleanup"}])
    await recall.process_frame(LLMContextFrame(context=context), FrameDirection.DOWNSTREAM)
    await capture.process_frame(
        LLMContextAssistantTurnFrame(text="bounded answer", timestamp="now"),
        FrameDirection.DOWNSTREAM,
    )

    loop = asyncio.get_running_loop()
    started = loop.time()
    await asyncio.wait_for(
        asyncio.gather(recall.cleanup(), capture.cleanup()),
        timeout=0.2,
    )
    elapsed = loop.time() - started

    assert elapsed < 0.1
    assert client.ingest_cancelled
    assert client.close_calls == 1


@pytest.mark.asyncio
async def test_cancelled_cleanup_still_drains_and_closes_owned_client_once():
    client = _FakeClient(ingest_delay=1.0, close_delay=0.001)
    service = MemcodeMemoryService(
        client=client,  # type: ignore[arg-type]
        close_client=True,
        session_id="call-123",
        config=MemcodeMemoryConfig(
            ingest_timeout_seconds=2.0,
            shutdown_timeout_seconds=0.03,
        ),
    )
    recall = _wire(service.recall_processor())
    capture = _wire(service.capture_processor())
    await _claim_shared_state(recall, capture)
    context = LLMContext([{"role": "user", "content": "cancel cleanup"}])
    await recall.process_frame(LLMContextFrame(context=context), FrameDirection.DOWNSTREAM)
    await capture.process_frame(
        LLMContextAssistantTurnFrame(text="cleanup answer", timestamp="now"),
        FrameDirection.DOWNSTREAM,
    )

    recall_cleanup_task = asyncio.create_task(recall.cleanup())
    capture_cleanup_task = asyncio.create_task(capture.cleanup())
    await client.ingest_started.wait()
    capture_cleanup_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await capture_cleanup_task
    await recall_cleanup_task

    assert client.ingest_cancelled
    assert client.close_calls == 1
    assert service._state._close_task is not None
    assert service._state._close_task.done()

    await asyncio.gather(recall.cleanup(), capture.cleanup())
    assert client.close_calls == 1


@pytest.mark.asyncio
async def test_access_token_provider_builds_one_owned_sdk_client(monkeypatch):
    created: list[dict[str, Any]] = []
    fake = _FakeClient()

    def factory(**kwargs):
        created.append(kwargs)
        return fake

    class Provider:
        async def get_access_token(self):
            return "access-token"

        async def refresh_access_token(self, *, failed_token=None):
            return "refreshed-token"

    monkeypatch.setattr(memory_module, "AsyncMemcodeClient", factory)
    provider = Provider()
    service = MemcodeMemoryService(
        api_url="https://memory.example.test",
        access_token_provider=provider,  # type: ignore[arg-type]
        session_id="call-123",
    )
    await _claim_shared_state(service.recall_processor(), service.capture_processor())

    assert created == [
        {
            "api_url": "https://memory.example.test",
            "access_token_provider": provider,
        }
    ]
    await asyncio.gather(
        service.recall_processor().cleanup(),
        service.capture_processor().cleanup(),
    )
    assert fake.close_calls == 1


@pytest.mark.asyncio
async def test_unexpected_tracked_task_exception_is_observed(caplog):
    client = _FakeClient()
    service = _service(client)

    async def fail():
        raise RuntimeError("unexpected-test-error")

    task = asyncio.create_task(fail())
    service._state.track_ingest(task)
    await asyncio.gather(task, return_exceptions=True)
    await asyncio.sleep(0)

    assert not service._state.ingest_tasks
    assert "Unexpected Memcode ingest task failure" in caplog.text


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"search_top_k": 0}, "search_top_k"),
        ({"search_minimum_score": 2}, "search_minimum_score"),
        ({"search_timeout_seconds": 0}, "search_timeout_seconds"),
        ({"max_context_characters": 100}, "max_context_characters"),
        ({"context_header": "  "}, "context_header"),
    ],
)
def test_config_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        MemcodeMemoryConfig(**kwargs)
