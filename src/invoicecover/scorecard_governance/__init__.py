"""复核结论回流、对照回放与评分卡版本治理。"""

from .contracts import (
    HIGH_RISK_THRESHOLD,
    ControlGateRejected,
    ControlReplay,
    GovernanceError,
    ReplayComparison,
    ReviewConclusion,
    ScorecardSwitchError,
)
from .engine import (
    compare_replays,
    normalize_conclusion,
    normalize_conclusions,
    replay_controls,
    split_maturity,
)
from .ledger import ScorecardLedger, TrainingVersion
from .service import decide_switch, evaluate_card_on_controls, run_governed_switch

# Descriptive aliases for callers naming the same governance workflow.
ScorecardGovernor = ScorecardLedger
ScorecardRegistry = ScorecardLedger
ReviewLedger = ScorecardLedger
promote_scorecard = run_governed_switch

__all__ = [
    "HIGH_RISK_THRESHOLD",
    "ControlGateRejected",
    "ControlReplay",
    "GovernanceError",
    "ReplayComparison",
    "ReviewConclusion",
    "ScorecardLedger",
    "ScorecardSwitchError",
    "TrainingVersion",
    "compare_replays",
    "decide_switch",
    "evaluate_card_on_controls",
    "normalize_conclusion",
    "normalize_conclusions",
    "replay_controls",
    "run_governed_switch",
    "split_maturity",
    "ScorecardGovernor",
    "ScorecardRegistry",
    "ReviewLedger",
    "promote_scorecard",
]
