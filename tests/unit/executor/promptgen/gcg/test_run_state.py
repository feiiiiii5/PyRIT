# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for typed optimization-iteration state in the GCG attack loop."""

import random
from typing import Any
from unittest.mock import MagicMock

import pytest

attack_manager_mod = pytest.importorskip(
    "pyrit.executor.promptgen.gcg.attack.base.attack_manager",
    reason="attack_manager module not importable",
)
torch = pytest.importorskip("torch", reason="torch not installed")

MultiPromptAttack = attack_manager_mod.MultiPromptAttack
OptimizationRunState = attack_manager_mod.OptimizationRunState
ProgressiveMultiPromptAttack = attack_manager_mod.ProgressiveMultiPromptAttack
ProgressiveScheduleState = attack_manager_mod.ProgressiveScheduleState
StopReason = attack_manager_mod.StopReason


def _bare_multi_prompt_attack(step_results: list[tuple[str, float]]) -> MultiPromptAttack:
    """Build a MultiPromptAttack without __init__ whose step() replays canned results."""
    attack = object.__new__(MultiPromptAttack)
    prompt_manager = MagicMock()
    prompt_manager.control_str = "initial"
    attack.prompts = [prompt_manager]
    attack.workers = [MagicMock()]
    attack.control_str = "initial"
    attack.logfile = None
    attack.step = MagicMock(side_effect=list(step_results))
    return attack


class TestStopReason:
    def test_has_expected_members(self) -> None:
        assert StopReason.MAX_STEPS_REACHED == "max_steps_reached"
        assert StopReason.ALL_PROMPTS_JAILBROKEN == "all_prompts_jailbroken"


class TestOptimizationRunState:
    def test_counters_and_stop_reason_default(self) -> None:
        state = OptimizationRunState(control="c", best_control="c", loss=1e6, best_loss=1e6)

        assert state.steps_completed == 0
        assert state.runtime == 0.0
        assert state.stop_reason is None


class TestProgressiveScheduleState:
    def test_defaults(self) -> None:
        schedule = ProgressiveScheduleState(goals_admitted=1, workers_admitted=2)

        assert schedule.steps_completed == 0
        assert schedule.loss == float("inf")
        assert schedule.stop_inner_on_success is False


class TestMultiPromptRunStateTracking:
    def test_run_sets_max_steps_reached_when_loop_exhausts(self) -> None:
        attack = _bare_multi_prompt_attack([("better", 1.0)])

        control, loss, steps = attack.run(n_steps=1, prev_loss=2.0, stop_on_success=False, anneal=True)

        assert (control, loss, steps) == ("better", 1.0, 1)
        state: OptimizationRunState | None = getattr(attack, "last_run_state", None)
        assert state is not None
        assert state.steps_completed == 1
        assert state.stop_reason == StopReason.MAX_STEPS_REACHED
        assert state.best_control == "better"
        assert state.best_loss == 1.0
        assert state.control == "better"

    def test_run_records_jailbroken_stop_reason_without_counting_final_check(self) -> None:
        attack = _bare_multi_prompt_attack([])
        attack.test = MagicMock(return_value=([[True]], [[1]], [[1.0]]))

        control, loss, steps = attack.run(n_steps=5, stop_on_success=True)

        assert (control, loss, steps) == ("initial", 1e6, 0)
        state: OptimizationRunState = attack.last_run_state
        assert state.steps_completed == 0
        assert state.stop_reason == StopReason.ALL_PROMPTS_JAILBROKEN
        attack.step.assert_not_called()

    def test_rejected_candidate_keeps_active_suffix_but_updates_loss(self) -> None:
        attack = _bare_multi_prompt_attack([("better", 1.0), ("worse", 5.0)])
        random.seed(2026)

        control, loss, steps = attack.run(n_steps=2, prev_loss=2.0, stop_on_success=False, anneal=True)

        # The worse candidate must be rejected by annealing with overwhelming
        # probability under this seed; the active suffix stays "better".
        assert control == "better"
        assert steps == 2
        state: OptimizationRunState = attack.last_run_state
        assert state.best_control == "better"
        assert state.best_loss == 1.0
        assert state.loss == 5.0
        assert state.stop_reason == StopReason.MAX_STEPS_REACHED

    def test_periodic_checkpoint_restores_active_suffix(self) -> None:
        attack = _bare_multi_prompt_attack([("better", 1.0), ("best-yet", 0.25)])
        attack.logfile = "unused-by-test.json"  # gate for periodic checkpoints; log/test_all are mocked
        attack.test_all = MagicMock(return_value=([[False]], [[0]], [[0.5]]))
        attack.log = MagicMock()

        attack.run(
            n_steps=2,
            prev_loss=2.0,
            stop_on_success=False,
            anneal=True,
            test_steps=1,
        )

        # Each periodic checkpoint evaluates the best-known suffix and then
        # restores whatever suffix was active for optimization.
        assert attack.control_str == "best-yet"
        assert attack.log.call_count == 2
        first_log_args = attack.log.call_args_list[0].args
        assert first_log_args[2] == "better"
        second_log_args = attack.log.call_args_list[1].args
        assert second_log_args[2] == "best-yet"

    def test_seeded_runs_produce_identical_trajectories(self) -> None:
        results = []
        for _ in range(2):
            random.seed(1234)
            attack = _bare_multi_prompt_attack([("a", 3.0), ("b", 2.0), ("c", 1.5)])
            results.append(attack.run(n_steps=3, prev_loss=4.0, stop_on_success=False, anneal=True))

        assert results[0] == results[1]
        assert results[0] == ("c", 1.5, 3)


class TestGCGCandidateSelection:
    def test_selects_minimum_within_single_group(self) -> None:
        from pyrit.executor.promptgen.gcg.attack.gcg.gcg_attack import GCGMultiPromptAttack

        attack = object.__new__(GCGMultiPromptAttack)
        next_control, cand_loss = attack._select_best_candidate(
            control_cands=[["aa", "bb"]],
            losses=torch.tensor([0.5, 9.0]),
            batch_size=2,
        )

        assert next_control == "aa"
        assert cand_loss.item() == pytest.approx(0.5)

    def test_decomposes_cross_group_argmin_index(self) -> None:
        from pyrit.executor.promptgen.gcg.attack.gcg.gcg_attack import GCGMultiPromptAttack

        attack = object.__new__(GCGMultiPromptAttack)
        next_control, cand_loss = attack._select_best_candidate(
            control_cands=[["aa", "bb"], ["cc", "dd"]],
            losses=torch.tensor([9.0, 8.0, 7.0, 6.0]),
            batch_size=2,
        )

        assert next_control == "dd"
        assert cand_loss.item() == pytest.approx(6.0)


class TestProgressiveRunScheduleState:
    def _bare_progressive_attack(self, inner_attack: Any) -> ProgressiveMultiPromptAttack:
        progressive = object.__new__(ProgressiveMultiPromptAttack)
        progressive.goals = ["goal"]
        progressive.targets = ["target"]
        progressive.workers = [MagicMock()]
        progressive.test_goals = []
        progressive.test_targets = []
        progressive.test_workers = []
        progressive.test_prefixes = []
        progressive.managers = {"MPA": MagicMock(return_value=inner_attack)}
        progressive.control = "initial"
        progressive.logfile = None
        progressive.progressive_goals = True
        progressive.progressive_models = True
        return progressive

    def test_finalize_phase_logs_final_evaluation_and_stops(self) -> None:
        inner_attack = MagicMock()
        inner_attack.run.return_value = ("ctrl", 0.5, 2)
        model_tests = ([[True]], [[1]], [[1.0]])
        inner_attack.test_all.return_value = model_tests
        progressive = self._bare_progressive_attack(inner_attack)

        control, steps = progressive.run(n_steps=10, stop_on_success=True)

        assert (control, steps) == ("ctrl", 2)
        schedule: ProgressiveScheduleState = progressive.last_schedule_state
        assert schedule.steps_completed == 2
        assert schedule.goals_admitted == 1
        assert schedule.workers_admitted == 1
        inner_attack.test_all.assert_called_once()
        inner_attack.log.assert_called_once_with(2, 10, "ctrl", 0.5, 0.0, model_tests, verbose=True)

    def test_schedule_exhaustion_continues_until_step_budget_spent(self) -> None:
        inner_attack = MagicMock()
        inner_attack.run.return_value = ("ctrl", 0.5, 2)
        progressive = self._bare_progressive_attack(inner_attack)

        control, steps = progressive.run(n_steps=10, stop_on_success=False)

        assert (control, steps) == ("ctrl", 10)
        schedule: ProgressiveScheduleState = progressive.last_schedule_state
        assert schedule.steps_completed == 10
        assert schedule.stop_inner_on_success is False
        inner_attack.run.assert_called_with(
            n_steps=2,
            batch_size=1024,
            topk=256,
            temp=1.0,
            allow_non_ascii=False,
            target_weight=None,
            control_weight=None,
            anneal=True,
            anneal_from=8,
            # The inner result's loss feeds back as the next phase's prev_loss
            # so the annealing temperature schedule stays continuous across
            # progressive admissions.
            prev_loss=0.5,
            stop_on_success=False,
            test_steps=50,
            filter_cand=True,
            verbose=True,
        )
