# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for skipped-technique diagnostics in ``ScenarioRunService`` run summaries."""

import pytest

from pyrit.backend.services.scenario_run_service import ScenarioRunService
from pyrit.models.identifiers.scenario_identifier import ScenarioIdentifier
from pyrit.models.results.scenario_result import ScenarioResult


def _service() -> ScenarioRunService:
    from unittest.mock import MagicMock

    service = object.__new__(ScenarioRunService)
    service._active_tasks = {}
    # The error fallback path queries persisted error results; none exist here.
    service._memory = MagicMock()
    service._memory.get_attack_results.return_value = []
    return service


def _result(*, techniques, skipped, display_group_map=None) -> ScenarioResult:
    return ScenarioResult.model_construct(
        scenario_identifier=ScenarioIdentifier(class_name="scenario", techniques=techniques),
        attack_results={},
        display_group_map=display_group_map or {},
        metadata={"skipped_techniques": sorted(skipped)},
    )


@pytest.mark.usefixtures("patch_central_database")
class TestBuildResponseSkippedTechniques:
    """``skipped_techniques`` comes from the resolution-time record the scenario persists."""

    def test_persisted_skips_are_reported_verbatim(self):
        result = _result(techniques=["alpha", "ghost"], skipped=["ghost"], display_group_map={"cell": "alpha"})

        summary = _service()._build_response_from_db(scenario_result=result)

        assert summary.skipped_techniques == ["ghost"]

    def test_display_group_labels_do_not_affect_reporting(self):
        """Custom display-group functions rename cells; skips must still surface verbatim."""
        result = _result(
            techniques=["alpha", "ghost"],
            skipped=["ghost"],
            display_group_map={"cell-a": "alpha (hard mode)", "cell-b": "ghost (hard mode)"},
        )

        summary = _service()._build_response_from_db(scenario_result=result)

        assert summary.skipped_techniques == ["ghost"]

    def test_no_selection_reports_no_skips(self):
        result = _result(techniques=None, skipped=[])

        summary = _service()._build_response_from_db(scenario_result=result)

        assert summary.skipped_techniques == []

    def test_skips_are_sorted_and_deduplicated(self):
        result = _result(techniques=["zeta", "alpha", "alpha"], skipped=["zeta", "alpha", "alpha"])

        summary = _service()._build_response_from_db(scenario_result=result)

        assert summary.skipped_techniques == ["alpha", "zeta"]

    def test_legacy_results_without_metadata_report_no_skips(self):
        result = ScenarioResult.model_construct(
            scenario_identifier=ScenarioIdentifier(class_name="scenario", techniques=["x"]),
            attack_results={},
            metadata={},
        )

        summary = _service()._build_response_from_db(scenario_result=result)

        assert summary.skipped_techniques == []
