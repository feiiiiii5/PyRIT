# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Read-only scorer catalog API routes."""

from fastapi import APIRouter

from pyrit.backend.models.objective_scorers import ObjectiveScorerPresetListResponse
from pyrit.backend.services.scorer_service import get_objective_scorer_service

router = APIRouter(prefix="/scorers", tags=["scorers"])


@router.get(
    "/objective-presets",
    response_model=ObjectiveScorerPresetListResponse,
)
def list_objective_scorer_presets() -> ObjectiveScorerPresetListResponse:
    """
    List initialized objective scorer presets without constructing or invoking scorers.

    Returns:
        ObjectiveScorerPresetListResponse: Redacted preset availability and metadata.
    """
    return get_objective_scorer_service().list_objective_scorer_presets()
