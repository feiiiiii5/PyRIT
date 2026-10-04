# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for the configured objective-scorer catalog.

Uses fake scorer classes registered directly into the registry, so nothing here needs a
model target, a network call or a paid service.
"""

import pytest

from pyrit.models.catalog import (
    ObjectiveScorerCatalog,
    ObjectiveScorerIncompatibility,
    ObjectiveScorerSummary,
)
from pyrit.registry.components.scorer_catalog import build_objective_scorer_catalog
from pyrit.registry.instance_registry import RegistryEntry


class _FakeScorer:
    """Minimal stand-in exposing only what the projection reads."""

    def __init__(self, scorer_type="true_false", chat_target=None, blocks=False, doc=None):
        self.scorer_type = scorer_type
        self._prompt_target = chat_target
        self.raise_if_scorer_blocks = blocks
        if doc is not None:
            self.__doc__ = doc

    def get_chat_target(self):
        """Same pure attribute read the Scorer base provides - no model call."""
        return self._prompt_target


class _FakeRegistry:
    """Stands in for ScorerRegistry; only ``instances`` is touched."""

    def __init__(self, entries):
        self._entries = entries

    @property
    def instances(self):
        return self

    def get_all_instances(self):
        return self._entries


def _entry(name, tags=None, **kwargs):
    return RegistryEntry(name=name, instance=_FakeScorer(**kwargs), tags=tags or {})


def _catalog(entries):
    return build_objective_scorer_catalog(_FakeRegistry(entries))


def test_lists_configured_instances_with_stable_names():
    catalog = _catalog([_entry("refusal_gpt5_4", {"refusal": ""}, doc="Refuses.")])

    assert [s.scorer_name for s in catalog.scorers] == ["refusal_gpt5_4"]
    assert catalog.scorers[0].description == "Refuses."


def test_reports_the_result_family_rather_than_assuming_one():
    catalog = _catalog([_entry("s", scorer_type="true_false")])

    assert catalog.scorers[0].scorer_type == "true_false"


def test_incompatible_result_family_is_listed_with_a_reason():
    catalog = _catalog([_entry("graded", scorer_type="float_scale")])

    assert catalog.scorers == []
    assert len(catalog.incompatible) == 1
    entry = catalog.incompatible[0]
    assert entry.scorer_name == "graded"
    assert entry.scorer_type == "float_scale"
    assert "cannot decide success" in entry.reason


def test_unknown_result_family_is_also_incompatible():
    catalog = _catalog([_entry("mystery", scorer_type="unknown")])

    assert catalog.scorers == []
    assert catalog.incompatible[0].scorer_type == "unknown"


def test_empty_registry_yields_an_empty_catalog_not_an_error():
    catalog = _catalog([])

    assert isinstance(catalog, ObjectiveScorerCatalog)
    assert catalog.scorers == []
    assert catalog.incompatible == []
    assert catalog.default_objective_scorer is None


def test_default_tag_is_reported_and_does_not_remove_the_preset():
    catalog = _catalog([_entry("chosen", {"default_objective_scorer": ""}), _entry("other")])

    assert catalog.default_objective_scorer == "chosen"
    assert {s.scorer_name for s in catalog.scorers} == {"chosen", "other"}
    chosen = next(s for s in catalog.scorers if s.scorer_name == "chosen")
    assert chosen.is_default_objective_scorer is True


def test_non_default_presets_are_marked_as_such():
    catalog = _catalog([_entry("plain")])

    assert catalog.scorers[0].is_default_objective_scorer is False
    assert catalog.default_objective_scorer is None


def test_tags_are_carried_through_verbatim():
    catalog = _catalog([_entry("s", {"refusal": "", "best_refusal": "true"})])

    assert catalog.scorers[0].tags == {"refusal": "", "best_refusal": "true"}


def test_refusal_polarity_is_reported():
    catalog = _catalog([_entry("blocking", blocks=True), _entry("scoring", blocks=False)])

    polarity = {s.scorer_name: s.blocks_instead_of_false for s in catalog.scorers}
    assert polarity == {"blocking": True, "scoring": False}


def test_missing_chat_target_is_reported_rather_than_discovered():
    catalog = _catalog([_entry("needs_target", chat_target=object()), _entry("offline")])

    needs = {s.scorer_name: s.requires_chat_target for s in catalog.scorers}
    assert needs == {"needs_target": True, "offline": False}


def test_listing_does_not_score_or_call_the_target():
    """Listing reads metadata only: never scores, never prompts a target."""

    class _Exploding:
        scorer_type = "true_false"

        def get_chat_target(self):
            # A pure attribute read, exactly as Scorer.get_chat_target is; what must
            # never happen is an actual prompt.
            return None

        def __call__(self, *args, **kwargs):
            raise AssertionError("listing must not score")

        def score_async(self, *args, **kwargs):
            raise AssertionError("listing must not score")

    catalog = build_objective_scorer_catalog(_FakeRegistry([RegistryEntry(name="s", instance=_Exploding(), tags={})]))

    assert [s.scorer_name for s in catalog.scorers] == ["s"]


def test_response_carries_no_credential_values():
    """Tags are the only free-form field, so nothing else can leak configuration."""

    class _Secretive:
        scorer_type = "true_false"
        api_key = "sk-should-not-appear"

        def get_chat_target(self):
            return None

    catalog = build_objective_scorer_catalog(
        _FakeRegistry([RegistryEntry(name="s", instance=_Secretive(), tags={"refusal": ""})])
    )

    dumped = catalog.model_dump_json()
    assert "sk-should-not-appear" not in dumped
    assert "api_key" not in dumped


def test_summary_model_is_importable_from_the_catalog_package():
    assert ObjectiveScorerSummary(scorer_name="s", scorer_type="true_false").scorer_name == "s"
    assert ObjectiveScorerIncompatibility(scorer_name="s", reason="r").reason == "r"


def test_projection_defaults_to_the_registry_singleton(monkeypatch):
    """Called with no argument it must read the process-wide registry."""
    import pyrit.registry.components.scorer_catalog as mod
    from pyrit.registry.components.scorer_registry import ScorerRegistry

    monkeypatch.setattr(
        ScorerRegistry,
        "get_registry_singleton",
        classmethod(lambda cls: _FakeRegistry([_entry("from_singleton", {"default_objective_scorer": ""})])),
    )

    catalog = mod.build_objective_scorer_catalog()

    assert [s.scorer_name for s in catalog.scorers] == ["from_singleton"]
    assert catalog.default_objective_scorer == "from_singleton"


@pytest.mark.parametrize(
    "scorer_type, expected_compatible",
    [("true_false", True), ("float_scale", False), ("unknown", False)],
)
def test_only_true_false_scorers_are_offered(scorer_type, expected_compatible):
    catalog = _catalog([_entry("s", scorer_type=scorer_type)])

    assert bool(catalog.scorers) is expected_compatible
