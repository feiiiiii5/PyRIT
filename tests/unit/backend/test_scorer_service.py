# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for the scorer backend service."""

import asyncio
import threading

import pytest

from pyrit.backend.models.scorers import CreateScorerRequest
from pyrit.backend.services.scorer_service import ScorerService
from pyrit.models import ComponentIdentifier, Scorable, Score, ScoringExpectation
from pyrit.models.catalog.scorer import ScorerInstance
from pyrit.registry import ScorerRegistry
from pyrit.score.scorer import Scorer


class _ServiceScorer(Scorer):
    """A real registered scorer with an observable constructor."""

    constructions = 0

    def __init__(self, *, label: str = "service") -> None:
        super().__init__()
        type(self).constructions += 1
        self.label = label

    def _build_identifier(self) -> ComponentIdentifier:
        return self._create_identifier(params={"label": self.label})

    async def _score_scorable_async(self, *, scorable: Scorable, expectation: ScoringExpectation | None) -> list[Score]:
        raise AssertionError("listing scorer metadata must not score")

    def validate_return_scores(self, scores: list[Score]) -> None:
        return None

    def get_scorer_metrics(self):
        return None


@pytest.fixture(autouse=True)
def reset_scorer_registry():
    ScorerRegistry.reset_registry_singleton()
    _ServiceScorer.constructions = 0
    yield
    ScorerRegistry.reset_registry_singleton()


async def test_types_are_metadata_only_and_empty_instance_list_is_valid() -> None:
    registry = ScorerRegistry.get_registry_singleton()
    registry.register_class(_ServiceScorer)
    service = ScorerService()

    types = await service.list_scorer_types_async()
    instances = await service.list_scorers_async()
    missing = await service.get_scorer_async(scorer_registry_name="missing")

    service_type = next(item for item in types.items if item.scorer_type == "_ServiceScorer")
    assert any(parameter.name == "label" for parameter in service_type.parameters)
    assert _ServiceScorer.constructions == 0
    assert instances.items == []
    assert instances.pagination.has_more is False
    assert missing is None


async def test_list_scorers_uses_sorted_names_and_cursor_pages() -> None:
    registry = ScorerRegistry.get_registry_singleton()
    registry.register_class(_ServiceScorer)
    registry.create_named_instance(name="z-last", type_name="_ServiceScorer", params={})
    registry.create_named_instance(name="a-first", type_name="_ServiceScorer", params={})
    service = ScorerService()

    first = await service.list_scorers_async(limit=1)
    second = await service.list_scorers_async(limit=1, cursor=first.pagination.next_cursor)

    assert [item.scorer_registry_name for item in first.items] == ["a-first"]
    assert first.pagination.has_more is True
    assert first.pagination.next_cursor == "a-first"
    assert [item.scorer_registry_name for item in second.items] == ["z-last"]
    assert second.pagination.has_more is False
    assert second.pagination.next_cursor is None


async def test_concurrent_same_name_creates_are_serialized(monkeypatch: pytest.MonkeyPatch) -> None:
    """One same-name create wins without replacing the scorer already registered."""
    registry = ScorerRegistry.get_registry_singleton()
    registry.register_class(_ServiceScorer)
    service = ScorerService()
    first_validation = threading.Event()
    release_first = threading.Event()
    competing_validation = threading.Event()
    lock_attempt = threading.Event()
    observed = threading.Event()
    first_thread: list[int] = []
    calls = threading.local()
    validate_name = registry.instances.validate_name_available

    def controlled_validate(name: str) -> None:
        count = getattr(calls, "count", 0) + 1
        calls.count = count
        validate_name(name)
        if name != "contended" or count != 2:
            return
        thread_id = threading.get_ident()
        if not first_thread:
            first_thread.append(thread_id)
            first_validation.set()
            if not release_first.wait(timeout=5):
                raise TimeoutError("test did not release the first scorer creation")
        elif thread_id != first_thread[0]:
            competing_validation.set()
            observed.set()

    class ObservedRLock:
        """Signal that a competing service operation reached its lock."""

        def __init__(self, lock: object) -> None:
            self._lock = lock

        def __enter__(self) -> "ObservedRLock":
            if first_validation.is_set() and threading.get_ident() != first_thread[0]:
                lock_attempt.set()
                observed.set()
            self._lock.acquire()
            return self

        def __exit__(self, *_: object) -> None:
            self._lock.release()

    monkeypatch.setattr(registry.instances, "validate_name_available", controlled_validate)
    if hasattr(service, "_registry_lock"):
        service._registry_lock = ObservedRLock(service._registry_lock)

    first_task: asyncio.Task | None = None
    second_task: asyncio.Task | None = None
    tasks: list[asyncio.Task] = []
    try:
        first_task = asyncio.create_task(
            service.create_scorer_async(
                request=CreateScorerRequest(name="contended", type="_ServiceScorer", params={"label": "first"})
            )
        )
        tasks.append(first_task)
        assert await asyncio.to_thread(first_validation.wait, 5), "first create did not reach its final name check"

        second_task = asyncio.create_task(
            service.create_scorer_async(
                request=CreateScorerRequest(name="contended", type="_ServiceScorer", params={"label": "second"})
            )
        )
        tasks.append(second_task)
        assert await asyncio.to_thread(observed.wait, 5), "second create neither blocked on the service lock nor raced"

        if competing_validation.is_set():
            second = await second_task
            release_first.set()
            first = await first_task
            pytest.fail(
                "unprotected create_named_instance allowed two same-name successes "
                f"({first.identifier.class_name!r}, {second.identifier.class_name!r}); "
                f"registry now contains label={registry.instances.get('contended').label!r}"
            )

        assert lock_attempt.is_set(), "the competing operation should wait at the service boundary"
        assert not second_task.done(), "a second create must wait until the first registration finishes"
        release_first.set()
        first, second = await asyncio.gather(first_task, second_task, return_exceptions=True)
        assert isinstance(first, ScorerInstance)
        assert isinstance(second, ValueError)
        assert registry.instances.get("contended").label == "first"
    finally:
        release_first.set()
        pending = [task for task in tasks if not task.done()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
