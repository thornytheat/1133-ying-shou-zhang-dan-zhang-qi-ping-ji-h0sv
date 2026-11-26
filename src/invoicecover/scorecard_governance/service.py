from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from datetime import date
from typing import Any

from .contracts import (
    HIGH_RISK_THRESHOLD,
    ControlReplay,
    ReplayComparison,
)
from .engine import compare_replays, replay_controls
from .ledger import ScorecardLedger, TrainingVersion


def evaluate_card_on_controls(
    controls: Iterable[Any],
    current_probabilities: Mapping[str, float | None] | Callable[[Any], float | None],
    candidate_probabilities: Mapping[str, float | None] | Callable[[Any], float | None],
    *,
    current_card_id: str = "current",
    candidate_card_id: str = "candidate",
    threshold: float = HIGH_RISK_THRESHOLD,
    unknown_as_miss: bool = False,
) -> ReplayComparison:
    current = replay_controls(
        controls,
        current_probabilities,
        card_id=current_card_id,
        threshold=threshold,
        unknown_as_miss=unknown_as_miss,
    )
    candidate = replay_controls(
        controls,
        candidate_probabilities,
        card_id=candidate_card_id,
        threshold=threshold,
        unknown_as_miss=unknown_as_miss,
    )
    return compare_replays(current, candidate)


def decide_switch(comparison: ReplayComparison) -> dict[str, Any]:
    """Pure decision used by CLI/API and useful in tests."""
    return {
        "switch": comparison.allowed,
        "active_card_id": comparison.candidate.card_id if comparison.allowed else comparison.current.card_id,
        "retained_card_id": None if comparison.allowed else comparison.current.card_id,
        "missed_current": comparison.current.missed_count,
        "missed_candidate": comparison.candidate.missed_count,
        "missed_delta": comparison.missed_delta,
        "reason": (
            "对照户高风险漏户未增加，允许换卡"
            if comparison.allowed
            else "对照户高风险漏户增加，保留现用卡"
        ),
    }


def run_governed_switch(
    ledger: ScorecardLedger,
    batch_id: str,
    *,
    comparison: ReplayComparison,
    commit: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Switch when safe; otherwise record rejection and keep current card."""
    if comparison.allowed:
        promotion = ledger.promote(
            batch_id, comparison=comparison, commit=commit
        )
        return {**decide_switch(comparison), **promotion}
    rejection = ledger.reject_switch(batch_id, comparison)
    return {**decide_switch(comparison), **rejection}
