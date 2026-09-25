# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Typed API responses for configured objective-scorer presets."""

from typing import Literal

from pydantic import BaseModel, Field


class ObjectiveScorerPreset(BaseModel):
    """A safe, read-only projection of one initialized scorer registry entry."""

    registry_name: str = Field(..., description="Stable name of the initialized scorer instance")
    scorer_type: str = Field(..., description="Scorer class name")
    scorer_family: Literal["true_false", "float_scale", "other"] = Field(
        ..., description="Scorer family exposed by the initialized instance"
    )
    result_family: Literal["boolean", "number", "unknown"] = Field(
        ..., description="Result family produced by the scorer"
    )
    description: str = Field(..., description="Safe summary of the scorer's objective-scoring role")
    true_value_meaning: Literal["objective_achieved", "refusal", "unknown"] = Field(
        ..., description="Meaning of a true boolean result, when known"
    )
    compatible: bool = Field(..., description="Whether this preset is validated for objective scoring")
    compatibility_reason: str | None = Field(
        default=None,
        description="Why the preset cannot safely be used as an objective scorer",
    )
    tags: list[str] = Field(default_factory=list, description="Relevant tag keys; tag values are not exposed")
    prerequisites: list[str] = Field(
        default_factory=list,
        description="Generic configuration prerequisites for the built-in preset",
    )
    is_configured_default: bool = Field(
        default=False,
        description="Whether this is the first configured default objective scorer by registry order",
    )


class ObjectiveScorerPresetListResponse(BaseModel):
    """Response for listing initialized objective-scorer presets."""

    items: list[ObjectiveScorerPreset] = Field(..., description="Configured scorer instances relevant to objectives")
    availability: Literal["ready", "empty", "unusable"] = Field(
        ..., description="Whether compatible objective presets are available"
    )
    message: str | None = Field(default=None, description="Setup guidance when no compatible preset is available")
    default_status: Literal["ready", "missing", "incompatible"] = Field(
        ..., description="Whether the configured default is safe to use"
    )
    default_name: str | None = Field(
        default=None,
        description="Configured default registry name, set only when its objective semantics are validated",
    )
    default_message: str | None = Field(default=None, description="Why a configured default is unavailable")
