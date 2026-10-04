# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Catalog models for configured objective scorers.

A scenario runs with one *objective scorer*: a pre-configured scorer instance that
decides whether the attack succeeded. The scorer's registry holds two different
things -- the buildable catalog of scorer *classes*, and the configured *instances*
that initializers registered with everything already wired up. Only the second kind
can be offered as a ready choice, so these models describe instances only.

The projection lives beside the registry (``pyrit.registry.components.scorer_catalog``)
so this module stays a set of wire types with no registry knowledge, matching
``initializer.py`` and ``target.py``.
"""

from pydantic import BaseModel, Field

from pyrit.models.score import ScoreType


class ObjectiveScorerSummary(BaseModel):
    """A configured objective-scorer preset a scenario can select."""

    #: Stable registry name, e.g. ``"refusal_gpt5_4"``. Safe to persist and send back.
    scorer_name: str = Field(..., description="Registry name of the configured scorer instance")
    #: Human-readable description, taken from the scorer's docstring when it has one.
    description: str = Field("", description="Human-readable description of the scorer")
    #: The result family this scorer produces, e.g. ``"true_false"`` or ``"float_scale"``.
    #: A scenario's objective scorer must be ``"true_false"``; anything else cannot decide
    #: success on its own, which is why the family is reported rather than assumed.
    scorer_type: ScoreType = Field(..., description="Result family produced by this scorer")
    #: Category and selection tags from the registry entry, e.g. ``{"refusal": ""}``.
    #: Empty-valued keys are the category tags; valued keys are configuration.
    tags: dict[str, str] = Field(default_factory=dict, description="Registry tags attached to the scorer instance")
    #: Whether this instance carries the ``default_objective_scorer`` tag, i.e. the one a
    #: scenario picks when the caller does not choose.
    is_default_objective_scorer: bool = Field(
        False, description="True when this is the configured default objective scorer"
    )
    #: Whether the scorer calls an LLM. Scorers that do cannot be listed without a
    #: configured chat target, which is reported here instead of being discovered.
    requires_chat_target: bool = Field(False, description="True when the scorer needs a configured LLM target")
    #: Refusal polarity: when True the scorer *blocks* (raises) instead of returning a
    #: false verdict. A scenario runs with ``RAISE_IF_DEFAULT_SCORER_BLOCKS`` set from its
    #: own policy, so the polarity of the preset decides how it reports a match.
    blocks_instead_of_false: bool = Field(
        False, description="True when a match raises rather than returning a false verdict"
    )


class ObjectiveScorerIncompatibility(BaseModel):
    """A configured scorer that cannot be selected as a scenario's objective scorer."""

    scorer_name: str = Field(..., description="Registry name of the configured scorer instance")
    #: Why it is not offered, e.g. ``"result family 'float_scale' cannot decide success"``.
    reason: str = Field(..., description="Why this scorer is not a compatible objective scorer")
    #: The result family that made it incompatible, when that is the reason.
    scorer_type: ScoreType | None = Field(None, description="Result family produced by the scorer, when known")


class ObjectiveScorerCatalog(BaseModel):
    """The configured objective scorers available to a scenario."""

    #: Every compatible preset, ordered as the registry holds them.
    scorers: list[ObjectiveScorerSummary] = Field(
        default_factory=list, description="Compatible objective-scorer presets"
    )
    #: Registry names of instances that exist but cannot serve as an objective scorer,
    #: each with the reason. Lets a caller tell "not offered" from "nothing configured"
    #: without the API having to decide what to do about the difference.
    incompatible: list[ObjectiveScorerIncompatibility] = Field(
        default_factory=list, description="Configured scorers that cannot be used as objective scorers"
    )
    #: Name of the default preset, or None when the registry has none configured.
    default_objective_scorer: str | None = Field(
        None, description="Registry name of the configured default objective scorer, if any"
    )
