# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for the configured objective-scorer preset route and service."""

from unittest.mock import Mock, patch

from azure.ai.contentsafety.models import TextCategory
from fastapi.testclient import TestClient

from pyrit.backend.main import app
from pyrit.backend.services.scorer_service import ObjectiveScorerService
from pyrit.models import SeedPrompt
from pyrit.registry.components import ScorerRegistry
from pyrit.score.float_scale.azure_content_filter_scorer import AzureContentFilterScorer
from pyrit.score.float_scale.float_scale_score_aggregator import FloatScaleScoreAggregator
from pyrit.score.float_scale.numeric_scale import NumericRubric
from pyrit.score.float_scale.self_ask_scale_scorer import SelfAskScaleScorer, render_scale_system_prompt
from pyrit.score.response_handler import JsonSchemaResponseHandler, TrueFalseResponseHandler
from pyrit.score.true_false.float_scale_threshold_scorer import FloatScaleThresholdScorer
from pyrit.score.true_false.self_ask_refusal_scorer import (
    RefusalScorerPaths,
    SelfAskRefusalScorer,
)
from pyrit.score.true_false.self_ask_true_false_scorer import (
    SelfAskTrueFalseScorer,
    TrueFalseQuestion,
    TrueFalseQuestionPaths,
    render_true_false_system_prompt,
)
from pyrit.score.true_false.true_false_composite_scorer import TrueFalseCompositeScorer
from pyrit.score.true_false.true_false_inverter_scorer import TrueFalseInverterScorer
from pyrit.score.true_false.true_false_score_aggregator import TrueFalseScoreAggregator
from pyrit.setup.initializers.scorers import ScorerInitializerTags


def _bare_scorer(scorer_type: type, **attributes: object) -> object:
    """Create a scorer instance without constructing targets or model clients."""
    scorer = object.__new__(scorer_type)
    for name, value in attributes.items():
        setattr(scorer, name, value)
    return scorer


def _bare_acs_scorer(*, harm_categories: list[TextCategory] | None = None) -> object:
    """Create an ACS scorer with the requested categories without creating a client."""
    return _bare_scorer(
        AzureContentFilterScorer,
        _harm_categories=list(TextCategory) if harm_categories is None else harm_categories,
    )


def _registry_with(*entries: tuple[object, str, dict[str, str] | None, dict[str, object] | None]) -> ScorerRegistry:
    registry = ScorerRegistry(lazy_discovery=True)
    for scorer, name, tags, metadata in entries:
        registry.instances.register(scorer, name=name, tags=tags, metadata=metadata)
    return registry


def _scale_and_refusal_composite(*, rubric: NumericRubric, system_prompt: str) -> object:
    rendered_prompt = render_scale_system_prompt(scale=rubric)
    scale_scorer = _bare_scorer(
        SelfAskScaleScorer,
        _scale=rubric,
        _system_prompt=system_prompt,
        _response_handler=JsonSchemaResponseHandler(
            response_schema=rendered_prompt.response_json_schema if system_prompt == rendered_prompt.value else None,
            numeric_value=True,
        ),
    )
    threshold = _bare_scorer(
        FloatScaleThresholdScorer,
        _scorer=scale_scorer,
        _threshold=0.9,
        _float_scale_aggregator=FloatScaleScoreAggregator.MAX,
        _score_aggregator=TrueFalseScoreAggregator.OR,
    )
    inverted_refusal = _bare_scorer(
        TrueFalseInverterScorer,
        _scorer=_default_refusal_scorer(),
        _score_aggregator=TrueFalseScoreAggregator.OR,
    )
    return _bare_scorer(
        TrueFalseCompositeScorer,
        _score_aggregator=TrueFalseScoreAggregator.AND,
        _scorers=[threshold, inverted_refusal],
    )


def _default_refusal_scorer(*, system_prompt: str | None = None) -> object:
    prompt = SeedPrompt.from_yaml_file(RefusalScorerPaths.OBJECTIVE_STRICT.value)
    return _bare_scorer(
        SelfAskRefusalScorer,
        _system_prompt=system_prompt if system_prompt is not None else prompt.value,
        _prompt_format_string=SelfAskRefusalScorer.DEFAULT_REFUSAL_PROMPT_FORMAT,
        _response_handler=TrueFalseResponseHandler(
            response_handler=JsonSchemaResponseHandler(response_schema=prompt.response_json_schema)
        ),
        _score_aggregator=TrueFalseScoreAggregator.OR,
    )


def _default_task_achieved_scorer(*, custom_prompt: str | None = None) -> object:
    question = TrueFalseQuestion.from_yaml(TrueFalseQuestionPaths.TASK_ACHIEVED.value)
    prompt = render_true_false_system_prompt(question=question)
    return _bare_scorer(
        SelfAskTrueFalseScorer,
        _question=question,
        _system_prompt=custom_prompt if custom_prompt is not None else prompt.value,
        _response_handler=TrueFalseResponseHandler(
            response_handler=JsonSchemaResponseHandler(response_schema=prompt.response_json_schema)
        ),
        _score_aggregator=TrueFalseScoreAggregator.OR,
    )


def test_objective_scorer_presets_route_returns_empty_registry_status() -> None:
    service = ObjectiveScorerService(registry=ScorerRegistry(lazy_discovery=True))
    with patch("pyrit.backend.routes.scorers.get_objective_scorer_service", return_value=service):
        response = TestClient(app).get("/api/scorers/objective-presets")

    assert response.status_code == 200
    assert response.json() == {
        "items": [],
        "availability": "empty",
        "message": "No initialized objective-scorer presets are available. Check scorer and target initialization.",
        "default_status": "missing",
        "default_name": None,
        "default_message": "No initialized scorer is tagged as the configured default objective scorer.",
    }


def test_lists_validated_default_and_non_default_presets_without_scoring() -> None:
    refusal = _default_refusal_scorer()
    refusal.get_identifier = Mock(side_effect=AssertionError("identifier must not be read"))
    refusal.get_chat_target = Mock(side_effect=AssertionError("target must not be read"))
    refusal.score_async = Mock(side_effect=AssertionError("scorer must not be invoked"))
    inverted_refusal = _bare_scorer(
        TrueFalseInverterScorer,
        _scorer=refusal,
        _score_aggregator=TrueFalseScoreAggregator.OR,
    )
    task_scorer = _default_task_achieved_scorer()
    registry = _registry_with(
        (
            inverted_refusal,
            "inverted_refusal",
            {
                ScorerInitializerTags.OBJECTIVE_COMPOSITE: "sensitive-tag-value",
                ScorerInitializerTags.DEFAULT_OBJECTIVE_SCORER: "also-sensitive",
            },
            {"credential": "metadata-secret-value"},
        ),
        (
            task_scorer,
            "task_achieved_gpt4o_temp9",
            {ScorerInitializerTags.TASK_ACHIEVED: "target=secret-value"},
            {"configuration": "credential-secret-value"},
        ),
    )

    response = ObjectiveScorerService(registry=registry).list_objective_scorer_presets()
    by_name = {preset.registry_name: preset for preset in response.items}

    assert response.availability == "ready"
    assert response.default_status == "ready"
    assert response.default_name == "inverted_refusal"
    assert by_name["inverted_refusal"].compatible is True
    assert by_name["inverted_refusal"].true_value_meaning == "objective_achieved"
    assert by_name["inverted_refusal"].is_configured_default is True
    assert by_name["inverted_refusal"].tags == [
        ScorerInitializerTags.DEFAULT_OBJECTIVE_SCORER,
        ScorerInitializerTags.OBJECTIVE_COMPOSITE,
    ]
    assert by_name["task_achieved_gpt4o_temp9"].compatible is True
    assert by_name["task_achieved_gpt4o_temp9"].is_configured_default is False
    assert by_name["task_achieved_gpt4o_temp9"].tags == [ScorerInitializerTags.TASK_ACHIEVED]
    response_json = response.model_dump_json()
    assert "sensitive-tag-value" not in response_json
    assert "metadata-secret-value" not in response_json
    assert "secret-value" not in response_json
    refusal.get_identifier.assert_not_called()
    refusal.get_chat_target.assert_not_called()
    refusal.score_async.assert_not_called()


def test_raw_refusal_polarity_is_incompatible_and_never_replaces_default() -> None:
    raw_refusal = _default_refusal_scorer()
    valid_inverter = _bare_scorer(
        TrueFalseInverterScorer,
        _scorer=_default_refusal_scorer(),
        _score_aggregator=TrueFalseScoreAggregator.OR,
    )
    registry = _registry_with(
        (
            raw_refusal,
            "fallback",
            {
                ScorerInitializerTags.REFUSAL: "true-means-refusal",
                ScorerInitializerTags.DEFAULT_OBJECTIVE_SCORER: "selected-by-config",
            },
            None,
        ),
        (
            valid_inverter,
            "inverted_refusal",
            {
                ScorerInitializerTags.OBJECTIVE_COMPOSITE: "true-means-achieved",
                ScorerInitializerTags.DEFAULT_OBJECTIVE_SCORER: "later-compatible-entry",
            },
            None,
        ),
    )

    response = ObjectiveScorerService(registry=registry).list_objective_scorer_presets()
    by_name = {preset.registry_name: preset for preset in response.items}

    assert by_name["fallback"].true_value_meaning == "refusal"
    assert by_name["fallback"].compatible is False
    assert by_name["fallback"].is_configured_default is True
    assert response.default_status == "incompatible"
    assert response.default_name is None
    assert "no replacement was selected" in (response.default_message or "")
    assert by_name["inverted_refusal"].compatible is True


def test_validated_objective_composite_is_listed_as_compatible() -> None:
    threshold = _bare_scorer(
        FloatScaleThresholdScorer,
        _scorer=_bare_acs_scorer(),
        _threshold=0.5,
        _float_scale_aggregator=FloatScaleScoreAggregator.MAX,
        _score_aggregator=TrueFalseScoreAggregator.OR,
    )
    inverted_refusal = _bare_scorer(
        TrueFalseInverterScorer,
        _scorer=_default_refusal_scorer(),
        _score_aggregator=TrueFalseScoreAggregator.OR,
    )
    composite = _bare_scorer(
        TrueFalseCompositeScorer,
        _score_aggregator=TrueFalseScoreAggregator.AND,
        _scorers=[threshold, inverted_refusal],
    )
    registry = _registry_with((composite, "acs_with_refusal", {ScorerInitializerTags.OBJECTIVE_COMPOSITE: ""}, None))

    response = ObjectiveScorerService(registry=registry).list_objective_scorer_presets()

    assert len(response.items) == 1
    assert response.items[0].registry_name == "acs_with_refusal"
    assert response.items[0].compatible is True
    assert response.items[0].true_value_meaning == "objective_achieved"


def test_acs_objective_composite_rejects_a_partial_harm_category_set() -> None:
    threshold = _bare_scorer(
        FloatScaleThresholdScorer,
        _scorer=_bare_acs_scorer(harm_categories=[TextCategory.HATE]),
        _threshold=0.5,
        _float_scale_aggregator=FloatScaleScoreAggregator.MAX,
        _score_aggregator=TrueFalseScoreAggregator.OR,
    )
    inverter = _bare_scorer(
        TrueFalseInverterScorer,
        _scorer=_default_refusal_scorer(),
        _score_aggregator=TrueFalseScoreAggregator.OR,
    )
    composite = _bare_scorer(
        TrueFalseCompositeScorer,
        _score_aggregator=TrueFalseScoreAggregator.AND,
        _scorers=[threshold, inverter],
    )
    registry = _registry_with((composite, "acs_with_refusal", {ScorerInitializerTags.OBJECTIVE_COMPOSITE: ""}, None))

    response = ObjectiveScorerService(registry=registry).list_objective_scorer_presets()

    assert response.items[0].compatible is False
    assert response.items[0].true_value_meaning == "unknown"


def test_validated_default_scale_and_refusal_preset_is_compatible() -> None:
    rubric = NumericRubric.from_yaml(SelfAskScaleScorer.ScalePaths.TREE_OF_ATTACKS_SCALE.value)
    prompt = render_scale_system_prompt(scale=rubric)
    composite = _scale_and_refusal_composite(
        rubric=rubric,
        system_prompt=prompt.value,
    )
    registry = _registry_with(
        (
            composite,
            "scale_and_refusal",
            {
                ScorerInitializerTags.OBJECTIVE_COMPOSITE: "",
                ScorerInitializerTags.DEFAULT_OBJECTIVE_SCORER: "",
            },
            None,
        )
    )

    response = ObjectiveScorerService(registry=registry).list_objective_scorer_presets()

    assert response.availability == "ready"
    assert response.default_status == "ready"
    assert response.default_name == "scale_and_refusal"
    assert response.items[0].compatible is True


def test_scale_objective_preset_rejects_a_non_objective_rubric() -> None:
    rubric = NumericRubric(
        category="sentiment",
        minimum_value=0,
        maximum_value=100,
        minimum_description="negative",
        maximum_description="positive",
    )
    composite = _scale_and_refusal_composite(
        rubric=rubric,
        system_prompt=render_scale_system_prompt(scale=rubric).value,
    )
    registry = _registry_with(
        (
            composite,
            "scale_and_refusal",
            {
                ScorerInitializerTags.OBJECTIVE_COMPOSITE: "",
                ScorerInitializerTags.DEFAULT_OBJECTIVE_SCORER: "",
            },
            None,
        )
    )

    response = ObjectiveScorerService(registry=registry).list_objective_scorer_presets()

    assert response.availability == "unusable"
    assert response.items[0].compatible is False
    assert response.items[0].true_value_meaning == "unknown"
    assert response.default_status == "incompatible"
    assert response.default_name is None


def test_scale_objective_preset_rejects_a_custom_prompt() -> None:
    rubric = NumericRubric.from_yaml(SelfAskScaleScorer.ScalePaths.TREE_OF_ATTACKS_SCALE.value)
    composite = _scale_and_refusal_composite(
        rubric=rubric,
        system_prompt="This rubric means sentiment, not objective completion.",
    )
    registry = _registry_with((composite, "scale_and_refusal", {ScorerInitializerTags.OBJECTIVE_COMPOSITE: ""}, None))

    response = ObjectiveScorerService(registry=registry).list_objective_scorer_presets()

    assert response.availability == "unusable"
    assert response.items[0].compatible is False


def test_task_achieved_preset_rejects_a_custom_prompt() -> None:
    registry = _registry_with(
        (
            _default_task_achieved_scorer(custom_prompt="A true verdict means the model refused."),
            "task_achieved_gpt4o_temp9",
            {ScorerInitializerTags.TASK_ACHIEVED: ""},
            None,
        )
    )

    response = ObjectiveScorerService(registry=registry).list_objective_scorer_presets()

    assert response.items[0].compatible is False
    assert response.items[0].true_value_meaning == "unknown"


def test_task_achieved_preset_rejects_a_custom_response_parser() -> None:
    scorer = _default_task_achieved_scorer()
    scorer._response_handler = TrueFalseResponseHandler(
        response_handler=JsonSchemaResponseHandler(score_value_output_key="is_refusal")
    )
    registry = _registry_with(
        (
            scorer,
            "task_achieved_gpt4o_temp9",
            {ScorerInitializerTags.TASK_ACHIEVED: ""},
            None,
        )
    )

    response = ObjectiveScorerService(registry=registry).list_objective_scorer_presets()

    assert response.items[0].compatible is False
    assert response.items[0].true_value_meaning == "unknown"


def test_inverted_custom_refusal_prompt_is_not_compatible() -> None:
    inverter = _bare_scorer(
        TrueFalseInverterScorer,
        _scorer=_default_refusal_scorer(system_prompt="A custom judge where true means refusal."),
        _score_aggregator=TrueFalseScoreAggregator.OR,
    )
    registry = _registry_with((inverter, "inverted_refusal", {ScorerInitializerTags.OBJECTIVE_COMPOSITE: ""}, None))

    response = ObjectiveScorerService(registry=registry).list_objective_scorer_presets()

    assert response.items[0].compatible is False
    assert response.items[0].true_value_meaning == "unknown"


def test_acs_objective_composite_rejects_a_non_default_threshold() -> None:
    threshold = _bare_scorer(
        FloatScaleThresholdScorer,
        _scorer=_bare_acs_scorer(),
        _threshold=0.1,
        _float_scale_aggregator=FloatScaleScoreAggregator.MAX,
        _score_aggregator=TrueFalseScoreAggregator.OR,
    )
    inverter = _bare_scorer(
        TrueFalseInverterScorer,
        _scorer=_default_refusal_scorer(),
        _score_aggregator=TrueFalseScoreAggregator.OR,
    )
    composite = _bare_scorer(
        TrueFalseCompositeScorer,
        _score_aggregator=TrueFalseScoreAggregator.AND,
        _scorers=[threshold, inverter],
    )
    registry = _registry_with((composite, "acs_with_refusal", {ScorerInitializerTags.OBJECTIVE_COMPOSITE: ""}, None))

    response = ObjectiveScorerService(registry=registry).list_objective_scorer_presets()

    assert response.items[0].compatible is False
    assert response.items[0].true_value_meaning == "unknown"


def test_incompatible_default_and_custom_names_are_redacted() -> None:
    non_boolean_scorer = _bare_scorer(SelfAskScaleScorer)
    custom_default = "credential=ghp_never_serialize_this"
    registry = _registry_with(
        (
            non_boolean_scorer,
            custom_default,
            {ScorerInitializerTags.DEFAULT_OBJECTIVE_SCORER: "selected"},
            {"api_key": "also-secret"},
        )
    )

    response = ObjectiveScorerService(registry=registry).list_objective_scorer_presets()
    response_json = response.model_dump_json()

    assert response.availability == "unusable"
    assert response.default_status == "incompatible"
    assert response.default_name is None
    assert response.items == []
    assert custom_default not in response_json
    assert "also-secret" not in response_json
    assert "none is a recognized safe preset" in (response.message or "")


def test_known_default_with_numeric_result_family_is_incompatible() -> None:
    scorer = _bare_scorer(SelfAskScaleScorer)
    registry = _registry_with(
        (
            scorer,
            "main",
            {ScorerInitializerTags.DEFAULT_OBJECTIVE_SCORER: "selected"},
            None,
        )
    )

    response = ObjectiveScorerService(registry=registry).list_objective_scorer_presets()

    assert response.items[0].result_family == "number"
    assert response.items[0].compatible is False
    assert response.default_status == "incompatible"
    assert response.default_name is None
