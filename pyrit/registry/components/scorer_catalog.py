# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Projection of configured objective-scorer presets out of the scorer registry.

Lives beside the registry rather than in ``pyrit.models.catalog`` so the catalog module
stays a set of wire types with no registry knowledge, matching how ``initializer.py`` and
``target.py`` are pure models.

Reading this is a lookup, not a run: it walks ``ScorerRegistry``'s ``instances`` and reads
each entry's name, tags and declared result family. It never scores, never calls a target,
and never puts a scorer's configuration into the result.
"""

from __future__ import annotations

from pyrit.models.catalog.scorer import (
    ObjectiveScorerCatalog,
    ObjectiveScorerIncompatibility,
    ObjectiveScorerSummary,
)

#: Only a true/false scorer can decide whether an attack succeeded on its own.
COMPATIBLE_SCORER_TYPE = "true_false"


def _describe(instance: object) -> str:
    """
    Read a scorer's own docstring, first line only.

    Args:
        instance (object): The scorer instance to describe.

    Returns:
        str: The first line of its docstring, or an empty string when it has none.
    """
    doc = getattr(instance, "__doc__", None)
    if not doc:
        return ""
    return doc.strip().splitlines()[0].strip()


def _blocks_instead_of_false(instance: object) -> bool:
    """
    Check whether a match raises (``raise_if_scorer_blocks``) instead of scoring false.

    Read with ``getattr`` because ``raise_if_scorer_blocks`` is declared on the LLM-backed
    scorers rather than on the ``Scorer`` base; a scorer that does not declare it does not
    block.

    Args:
        instance (object): The scorer instance to inspect.

    Returns:
        bool: True when the scorer blocks instead of returning a false verdict.
    """
    return bool(getattr(instance, "raise_if_scorer_blocks", False))


def build_objective_scorer_catalog(registry: object | None = None) -> ObjectiveScorerCatalog:
    """
    Describe the objective-scorer presets the registry has configured.

    Only *instances* are considered. The registry also holds a buildable catalog of scorer
    classes, which are not usable as a choice: they still need a chat target and any child
    scorers wired up.

    Args:
        registry: The scorer registry to read. Defaults to the process-wide singleton.

    Returns:
        ObjectiveScorerCatalog: The compatible presets, the configured ones that are not
        compatible and why, and the name of the configured default.
    """
    if registry is None:
        from pyrit.registry.components.scorer_registry import ScorerRegistry

        registry = ScorerRegistry.get_registry_singleton()

    catalog = ObjectiveScorerCatalog()
    default_name: str | None = None

    for entry in registry.instances.get_all_instances():
        instance = entry.instance
        scorer_type = getattr(instance, "scorer_type", "unknown")
        is_default = "default_objective_scorer" in entry.tags

        if scorer_type != COMPATIBLE_SCORER_TYPE:
            catalog.incompatible.append(
                ObjectiveScorerIncompatibility(
                    scorer_name=entry.name,
                    scorer_type=scorer_type,
                    reason=(
                        f"result family {scorer_type!r} cannot decide success on its own; "
                        f"an objective scorer must be {COMPATIBLE_SCORER_TYPE!r}"
                    ),
                )
            )
            continue

        catalog.scorers.append(
            ObjectiveScorerSummary(
                scorer_name=entry.name,
                description=_describe(instance),
                scorer_type=scorer_type,
                tags=dict(entry.tags),
                is_default_objective_scorer=is_default,
                requires_chat_target=getattr(instance, "get_chat_target", lambda: None)() is not None,
                blocks_instead_of_false=_blocks_instead_of_false(instance),
            )
        )

        if is_default:
            default_name = entry.name

    catalog.default_objective_scorer = default_name
    return catalog
