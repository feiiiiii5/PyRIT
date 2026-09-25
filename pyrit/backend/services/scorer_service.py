# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Read-only projections of initialized objective-scorer instances."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Literal

from azure.ai.contentsafety.models import TextCategory

from pyrit.backend.models.objective_scorers import ObjectiveScorerPreset, ObjectiveScorerPresetListResponse
from pyrit.models import SeedPrompt
from pyrit.registry.components import ScorerRegistry
from pyrit.score.float_scale.azure_content_filter_scorer import AzureContentFilterScorer
from pyrit.score.float_scale.float_scale_score_aggregator import FloatScaleScoreAggregator
from pyrit.score.float_scale.float_scale_scorer import FloatScaleScorer
from pyrit.score.float_scale.numeric_scale import NumericRubric
from pyrit.score.float_scale.self_ask_scale_scorer import SelfAskScaleScorer, render_scale_system_prompt
from pyrit.score.response_handler import JsonSchemaResponseHandler, TrueFalseResponseHandler
from pyrit.score.true_false.float_scale_threshold_scorer import FloatScaleThresholdScorer
from pyrit.score.true_false.self_ask_refusal_scorer import RefusalScorerPaths, SelfAskRefusalScorer
from pyrit.score.true_false.self_ask_true_false_scorer import (
    SelfAskTrueFalseScorer,
    TrueFalseQuestion,
    TrueFalseQuestionPaths,
    render_true_false_system_prompt,
)
from pyrit.score.true_false.true_false_composite_scorer import TrueFalseCompositeScorer
from pyrit.score.true_false.true_false_inverter_scorer import TrueFalseInverterScorer
from pyrit.score.true_false.true_false_score_aggregator import TrueFalseScoreAggregator
from pyrit.score.true_false.true_false_scorer import TrueFalseScorer
from pyrit.setup.initializers.scorers import ScorerInitializerTags

if TYPE_CHECKING:
    from pathlib import Path

    from pyrit.registry.instance_registry import RegistryEntry
    from pyrit.score.scorer import Scorer

_OBJECTIVE_TAGS = frozenset(
    {
        ScorerInitializerTags.BEST_OBJECTIVE,
        ScorerInitializerTags.DEFAULT_OBJECTIVE_SCORER,
        ScorerInitializerTags.OBJECTIVE_COMPOSITE,
        ScorerInitializerTags.TASK_ACHIEVED,
        ScorerInitializerTags.REFUSAL,
    }
)
_INVERTED_REFUSAL_NAMES = frozenset({"main", "fallback", "inverted_refusal"})
_TASK_ACHIEVED_QUESTIONS = {
    "task_achieved_gpt4o_temp9": TrueFalseQuestionPaths.TASK_ACHIEVED.value,
    "task_achieved_refined_gpt4o_temp9": TrueFalseQuestionPaths.TASK_ACHIEVED_REFINED.value,
}
_OBJECTIVE_COMPOSITES = frozenset({"acs_with_refusal", "scale_and_refusal"})
_RAW_REFUSAL_PRESETS = frozenset(
    {
        "refusal_gpt4o_objective_strict",
        "refusal_gpt4o_objective_lenient",
        "refusal_gpt4o_no_objective_strict",
        "refusal_gpt4o_no_objective_lenient",
        "refusal_gpt5_4",
        "refusal_gpt5_1",
        "refusal_gpt4o_unsafe",
    }
)
_PRESET_DESCRIPTIONS = {
    "main": "Default objective scorer with refusal results inverted to objective achievement.",
    "fallback": "Fallback objective scorer with refusal results inverted to objective achievement.",
    "inverted_refusal": "Refusal results inverted so true means the response did not refuse.",
    "task_achieved_gpt4o_temp9": "Evaluates whether the response fulfills the task-achievement rubric.",
    "task_achieved_refined_gpt4o_temp9": "Evaluates task fulfillment with the refined task-achievement rubric.",
    "acs_with_refusal": "Combines an ACS threshold with a non-refusal verdict.",
    "scale_and_refusal": "Combines a scale threshold with a non-refusal verdict.",
}
_SAFE_PRESET_NAMES = frozenset(_PRESET_DESCRIPTIONS) | _RAW_REFUSAL_PRESETS
_PREREQUISITES = {
    "main": ["The configured main scorer target must be initialized."],
    "fallback": ["The configured fallback scorer target must be initialized."],
    "inverted_refusal": ["A refusal scorer must be initialized."],
    "task_achieved_gpt4o_temp9": ["A scorer target must be initialized."],
    "task_achieved_refined_gpt4o_temp9": ["A scorer target must be initialized."],
    "acs_with_refusal": ["An ACS threshold scorer and a refusal scorer must be initialized."],
    "scale_and_refusal": ["A scale threshold scorer and a refusal scorer must be initialized."],
}


@lru_cache(maxsize=1)
def _load_tree_of_attacks_rubric() -> NumericRubric:
    """
    Load the bundled objective scale used by the default initializer.

    Returns:
        NumericRubric: The canonical tree-of-attacks rubric.
    """
    return NumericRubric.from_yaml(SelfAskScaleScorer.ScalePaths.TREE_OF_ATTACKS_SCALE.value)


@lru_cache(maxsize=1)
def _render_tree_of_attacks_prompt() -> SeedPrompt:
    """
    Render the canonical prompt for the default objective scale.

    Returns:
        SeedPrompt: The bundled default scale prompt and response schema.
    """
    return render_scale_system_prompt(scale=_load_tree_of_attacks_rubric())


@lru_cache(maxsize=2)
def _load_task_achieved_question(path: Path) -> TrueFalseQuestion:
    """
    Load a bundled task rubric for a structural, non-scoring compatibility check.

    Returns:
        TrueFalseQuestion: The bundled task-achievement question.
    """
    return TrueFalseQuestion.from_yaml(path)


@lru_cache(maxsize=2)
def _load_task_achieved_prompt(path: Path) -> SeedPrompt:
    """
    Render the bundled task-achievement prompt and its response schema.

    Returns:
        SeedPrompt: The canonical prompt and parser schema for the task rubric.
    """
    return render_true_false_system_prompt(question=_load_task_achieved_question(path))


@lru_cache(maxsize=1)
def _load_builtin_refusal_prompts() -> tuple[SeedPrompt, ...]:
    """
    Load the built-in refusal prompts recognized by the scorer initializer.

    Returns:
        tuple[SeedPrompt, ...]: The bundled refusal prompt variants.
    """
    return tuple(SeedPrompt.from_yaml_file(path.value) for path in RefusalScorerPaths)


def _has_builtin_response_handler(
    scorer: object,
    *,
    response_schema: object,
    numeric_value: bool,
) -> bool:
    """
    Require the default JSON parser and score-value contract for a built-in scorer.

    Returns:
        bool: Whether the configured parser matches the built-in contract.
    """
    response_handler = getattr(scorer, "_response_handler", None)
    if not numeric_value:
        if type(response_handler) is not TrueFalseResponseHandler:
            return False
        response_handler = getattr(response_handler, "_response_handler", None)
    if type(response_handler) is not JsonSchemaResponseHandler:
        return False
    expected_keys = {
        "_score_value_output_key": "score_value",
        "_rationale_output_key": "rationale",
        "_description_output_key": "description",
        "_metadata_output_key": "metadata",
        "_category_output_key": "category",
    }
    if any(getattr(response_handler, field, None) != value for field, value in expected_keys.items()):
        return False
    if getattr(response_handler, "_response_schema", None) != response_schema:
        return False
    return getattr(response_handler, "_numeric_value", None) is numeric_value


def _is_builtin_refusal_scorer(scorer: object) -> bool:
    """
    Validate the built-in refusal rubric, input template, aggregation, and parser.

    Returns:
        bool: Whether the scorer has validated refusal semantics.
    """
    if type(scorer) is not SelfAskRefusalScorer:
        return False
    system_prompt = getattr(scorer, "_system_prompt", None)
    if not isinstance(system_prompt, str):
        return False
    prompt = next(
        (candidate for candidate in _load_builtin_refusal_prompts() if candidate.value == system_prompt), None
    )
    if prompt is None:
        return False
    if getattr(scorer, "_prompt_format_string", None) != SelfAskRefusalScorer.DEFAULT_REFUSAL_PROMPT_FORMAT:
        return False
    if getattr(scorer, "_score_aggregator", None) is not TrueFalseScoreAggregator.OR:
        return False
    return _has_builtin_response_handler(
        scorer,
        response_schema=prompt.response_json_schema,
        numeric_value=False,
    )


def _is_inverted_refusal(scorer: object) -> bool:
    """
    Check whether an exact built-in inverter wraps a semantically validated refusal scorer.

    Returns:
        bool: Whether the configured inverter produces a validated non-refusal result.
    """
    return (
        type(scorer) is TrueFalseInverterScorer
        and getattr(scorer, "_score_aggregator", None) is TrueFalseScoreAggregator.OR
        and _is_builtin_refusal_scorer(getattr(scorer, "_scorer", None))
    )


def _is_valid_task_achieved_preset(*, name: str, scorer: object, tags: set[str]) -> bool:
    """
    Require a built-in task rubric, its matching registry tag, and the bundled question.

    Returns:
        bool: Whether the initialized scorer matches the built-in task preset.
    """
    if name not in _TASK_ACHIEVED_QUESTIONS or ScorerInitializerTags.TASK_ACHIEVED not in tags:
        return False
    if type(scorer) is not SelfAskTrueFalseScorer:
        return False
    question = getattr(scorer, "_question", None)
    if not isinstance(question, TrueFalseQuestion):
        return False
    if question != _load_task_achieved_question(_TASK_ACHIEVED_QUESTIONS[name]):
        return False
    prompt = _load_task_achieved_prompt(_TASK_ACHIEVED_QUESTIONS[name])
    if getattr(scorer, "_system_prompt", None) != prompt.value:
        return False
    if getattr(scorer, "_score_aggregator", None) is not TrueFalseScoreAggregator.OR:
        return False
    return _has_builtin_response_handler(
        scorer,
        response_schema=prompt.response_json_schema,
        numeric_value=False,
    )


def _is_valid_objective_composite(*, name: str, scorer: object, tags: set[str]) -> bool:
    """
    Validate the built-in AND composite shape without reading scorer identifiers.

    Returns:
        bool: Whether the composite has the validated objective-scoring shape.
    """
    if name not in _OBJECTIVE_COMPOSITES or ScorerInitializerTags.OBJECTIVE_COMPOSITE not in tags:
        return False
    if type(scorer) is not TrueFalseCompositeScorer:
        return False
    if scorer._score_aggregator is not TrueFalseScoreAggregator.AND or len(scorer._scorers) != 2:
        return False

    threshold_scorer, refusal_scorer = scorer._scorers
    if type(threshold_scorer) is not FloatScaleThresholdScorer or not _is_inverted_refusal(refusal_scorer):
        return False
    if threshold_scorer._score_aggregator is not TrueFalseScoreAggregator.OR:
        return False
    if threshold_scorer._float_scale_aggregator is not FloatScaleScoreAggregator.MAX:
        return False
    threshold_input = getattr(threshold_scorer, "_scorer", None)
    if name == "acs_with_refusal":
        categories = getattr(threshold_input, "_harm_categories", None)
        return (
            type(threshold_input) is AzureContentFilterScorer
            and threshold_scorer.threshold == 0.5
            and isinstance(categories, list)
            and len(categories) == len(TextCategory)
            and all(category in categories for category in TextCategory)
        )
    if type(threshold_input) is not SelfAskScaleScorer:
        return False
    if threshold_scorer.threshold != 0.9:
        return False
    scale = getattr(threshold_input, "_scale", None)
    if not isinstance(scale, NumericRubric) or scale != _load_tree_of_attacks_rubric():
        return False
    prompt = _render_tree_of_attacks_prompt()
    if getattr(threshold_input, "_system_prompt", None) != prompt.value:
        return False
    return _has_builtin_response_handler(
        threshold_input,
        response_schema=prompt.response_json_schema,
        numeric_value=True,
    )


def _is_valid_objective_preset(*, name: str, scorer: object, tags: set[str]) -> bool:
    """
    Validate only the built-in shapes whose true result has objective meaning.

    Returns:
        bool: Whether the scorer's true result means objective achievement.
    """
    if name in _INVERTED_REFUSAL_NAMES:
        return _is_inverted_refusal(scorer)
    if name in _TASK_ACHIEVED_QUESTIONS:
        return _is_valid_task_achieved_preset(name=name, scorer=scorer, tags=tags)
    return _is_valid_objective_composite(name=name, scorer=scorer, tags=tags)


def _result_family(
    scorer: object,
) -> tuple[Literal["true_false", "float_scale", "other"], Literal["boolean", "number", "unknown"]]:
    """
    Describe the scorer/result family without asking it to build an identifier.

    Returns:
        tuple[str, str]: The scorer family and result family labels.
    """
    if isinstance(scorer, TrueFalseScorer):
        return "true_false", "boolean"
    if isinstance(scorer, FloatScaleScorer):
        return "float_scale", "number"
    return "other", "unknown"


def _project_entry(
    *,
    entry: RegistryEntry[Scorer],
    is_configured_default: bool,
) -> ObjectiveScorerPreset:
    """
    Project safe registry facts, deliberately excluding tag values and metadata.

    Returns:
        ObjectiveScorerPreset: The public, redacted preset description.
    """
    scorer = entry.instance
    tag_keys = set(entry.tags)
    scorer_family, result_family = _result_family(scorer)
    raw_refusal = _is_builtin_refusal_scorer(scorer)

    if raw_refusal:
        true_value_meaning = "refusal"
        compatible = False
        reason = "A true value means refusal, not objective achievement."
        description = "Raw refusal scorer; true means the response contains a refusal."
    elif result_family != "boolean":
        true_value_meaning = "unknown"
        compatible = False
        reason = "Objective scorers must produce a boolean result with validated objective semantics."
        description = "Configured scorer with a non-boolean result family."
    elif _is_valid_objective_preset(name=entry.name, scorer=scorer, tags=tag_keys):
        true_value_meaning = "objective_achieved"
        compatible = True
        reason = None
        description = _PRESET_DESCRIPTIONS[entry.name]
    else:
        true_value_meaning = "unknown"
        compatible = False
        reason = "The scorer's true/false polarity has not been validated for objective scoring."
        description = "Configured true/false scorer with unverified objective semantics."

    return ObjectiveScorerPreset(
        registry_name=entry.name,
        scorer_type=type(scorer).__name__,
        scorer_family=scorer_family,
        result_family=result_family,
        description=description,
        true_value_meaning=true_value_meaning,
        compatible=compatible,
        compatibility_reason=reason,
        tags=sorted(tag for tag in tag_keys if tag in _OBJECTIVE_TAGS),
        prerequisites=list(_PREREQUISITES.get(entry.name, [])),
        is_configured_default=is_configured_default,
    )


class ObjectiveScorerService:
    """List initialized objective scorer instances without constructing or scoring them."""

    def __init__(self, *, registry: ScorerRegistry | None = None) -> None:
        """
        Initialize the service with a registry, defaulting to the process singleton.

        Args:
            registry: The registry containing initialized scorer instances.
        """
        self._registry = registry if registry is not None else ScorerRegistry.get_registry_singleton()

    def list_objective_scorer_presets(self) -> ObjectiveScorerPresetListResponse:
        """
        Return validated preset metadata from the current instance registry only.

        Returns:
            ObjectiveScorerPresetListResponse: Redacted preset availability and metadata.
        """
        entries = self._registry.instances.get_all_instances()
        relevant_entries = [
            entry
            for entry in entries
            if entry.name in _SAFE_PRESET_NAMES
            or bool(set(entry.tags) & _OBJECTIVE_TAGS)
            or ScorerInitializerTags.DEFAULT_OBJECTIVE_SCORER in entry.tags
        ]
        candidates = [
            entry
            for entry in relevant_entries
            if entry.name in _SAFE_PRESET_NAMES
            and (entry.name in {"main", "fallback"} or bool(set(entry.tags) & _OBJECTIVE_TAGS))
        ]
        default_entries = self._registry.instances.get_by_tag(tag=ScorerInitializerTags.DEFAULT_OBJECTIVE_SCORER)
        configured_default = default_entries[0] if default_entries else None
        items = [
            _project_entry(
                entry=entry,
                is_configured_default=configured_default is not None and entry.name == configured_default.name,
            )
            for entry in candidates
        ]
        default_preset = next(
            (
                item
                for item in items
                if configured_default is not None and item.registry_name == configured_default.name
            ),
            None,
        )

        if configured_default is None:
            default_status = "missing"
            default_name = None
            default_message = "No initialized scorer is tagged as the configured default objective scorer."
        elif default_preset is None or not default_preset.compatible:
            default_status = "incompatible"
            default_name = None
            default_message = (
                "The first configured default is not a validated objective scorer; no replacement was selected."
            )
        else:
            default_status = "ready"
            default_name = configured_default.name
            default_message = None

        if not relevant_entries:
            availability = "empty"
            message = "No initialized objective-scorer presets are available. Check scorer and target initialization."
        elif not items:
            availability = "unusable"
            message = "Initialized objective-related scorers were found, but none is a recognized safe preset."
        elif not any(item.compatible for item in items):
            availability = "unusable"
            message = "Initialized objective-related scorers were found, but none has validated objective semantics."
        else:
            availability = "ready"
            message = None

        return ObjectiveScorerPresetListResponse(
            items=items,
            availability=availability,
            message=message,
            default_status=default_status,
            default_name=default_name,
            default_message=default_message,
        )


@lru_cache(maxsize=1)
def get_objective_scorer_service() -> ObjectiveScorerService:
    """
    Get the process-wide objective scorer service.

    Returns:
        ObjectiveScorerService: The cached objective-scorer service.
    """
    return ObjectiveScorerService()
