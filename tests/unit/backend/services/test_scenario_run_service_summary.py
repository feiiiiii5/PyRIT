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


def _result(*, techniques, display_groups) -> ScenarioResult:
    return ScenarioResult(
        scenario_identifier=ScenarioIdentifier(name="scenario", techniques=techniques),
        attack_results={},
        display_group_map=display_groups,
    )


@pytest.mark.usefixtures("patch_central_database")
class TestBuildResponseSkippedTechniques:
    """Selected techniques with no built attack cell surface as ``skipped_techniques``."""

    def test_selected_without_built_label_is_reported_skipped(self):
        result = _result(techniques=["alpha", "ghost"], display_groups={"alpha::ds": "alpha"})

        summary = _service()._build_response_from_db(scenario_result=result)

        assert summary.skipped_techniques == ["ghost"]
        assert summary.techniques_used is not None

    def test_decorated_display_label_still_counts_as_built(self):
        # Custom ``display_group_fn`` may decorate technique names; a label that
        # contains the technique name must not be reported as skipped.
        result = _result(techniques=["alpha"], display_groups={"cell-1": "alpha (hard mode)"})

        summary = _service()._build_response_from_db(scenario_result=result)

        assert summary.skipped_techniques == []

    def test_no_selection_reports_no_skips(self):
        result = _result(techniques=None, display_groups={})

        summary = _service()._build_response_from_db(scenario_result=result)

        assert summary.skipped_techniques == []

    def test_skips_are_sorted_and_deduplicated(self):
        result = _result(
            techniques=["zeta", "alpha", "alpha"],
            display_groups={"mid::ds": "mid"},
        )

        summary = _service()._build_response_from_db(scenario_result=result)

        assert summary.skipped_techniques == ["alpha", "zeta"]
