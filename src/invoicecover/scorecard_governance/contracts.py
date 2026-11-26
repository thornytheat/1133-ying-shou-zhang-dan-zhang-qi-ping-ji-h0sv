from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Mapping


HIGH_RISK_THRESHOLD = 0.72


class GovernanceError(ValueError):
    """复核结论或版本台账不符合业务契约。"""


class ConclusionConflict(GovernanceError):
    """同一结论标识被补录成互相冲突的内容。"""


class ControlGateRejected(GovernanceError):
    """对照户回放显示高风险漏户增加。"""


class ScorecardSwitchError(RuntimeError):
    """换卡提交失败；调用方必须继续使用上一版评分卡。"""


def parse_date(value: date | str | None) -> date | None:
    if value is None or isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def canonical_json(payload: Any) -> str:
    import json

    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if hasattr(value, "_asdict"):
        return value._asdict()
    if hasattr(value, "__dict__"):
        return {k: v for k, v in vars(value).items() if not k.startswith("_")}
    return str(value)


def content_hash(payload: Any, length: int = 14) -> str:
    from hashlib import sha256

    return sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:length]


@dataclass(frozen=True)
class ReviewConclusion:
    """一户在某批复核中的结论。

    ``label is None`` 表示观察期尚未走完或结论未回收；该记录只允许登记，
    不能进入训练。
    """

    debtor: str
    label: int | None
    observed_at: date | None = None
    mature_on: date | None = None
    conclusion_id: str | None = None
    batch_id: str | None = None
    reviewed: bool = True
    source: str = "manual_review"
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.debtor:
            raise GovernanceError("复核结论缺少户名")
        if self.label not in (None, 0, 1):
            raise GovernanceError("复核标签只能是 0、1 或 None")
        if self.mature_on is not None and self.observed_at is not None:
            # 到期日等于特征日的记录不可能已经形成回款结论。
            if self.mature_on < self.observed_at:
                raise GovernanceError("观察期结束日不能早于特征日")

    def is_mature(self, as_of: date | str) -> bool:
        current = parse_date(as_of)
        if current is None:
            raise GovernanceError("缺少判断观察期是否完成的 as_of")
        return self.label is not None and self.mature_on is not None and self.mature_on <= current

    @property
    def stable_id(self) -> str:
        if self.conclusion_id:
            return str(self.conclusion_id)
        return content_hash(
            {
                "batch_id": self.batch_id,
                "debtor": self.debtor,
                "observed_at": self.observed_at,
                "source": self.source,
            },
            length=16,
        )

    def fingerprint(self) -> str:
        return content_hash(
            {
                "debtor": self.debtor,
                "label": self.label,
                "observed_at": self.observed_at,
                "mature_on": self.mature_on,
                "reviewed": self.reviewed,
                "source": self.source,
                "payload": dict(self.payload),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "conclusion_id": self.stable_id,
            "batch_id": self.batch_id,
            "debtor": self.debtor,
            "label": self.label,
            "observed_at": self.observed_at.isoformat() if self.observed_at else None,
            "mature_on": self.mature_on.isoformat() if self.mature_on else None,
            "mature": None,
            "reviewed": self.reviewed,
            "source": self.source,
            "fingerprint": self.fingerprint(),
            "payload": dict(self.payload),
        }


@dataclass(frozen=True)
class ControlReplay:
    card_id: str
    control_count: int
    positive_count: int
    flagged: tuple[str, ...] = ()
    missed: tuple[str, ...] = ()
    out_of_support: tuple[str, ...] = ()
    threshold: float = HIGH_RISK_THRESHOLD

    @property
    def missed_count(self) -> int:
        return len(self.missed)

    @property
    def accepted(self) -> bool:
        return True

    def compared_with(self, current: "ControlReplay") -> "ReplayComparison":
        return ReplayComparison(current=current, candidate=self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "card_id": self.card_id,
            "control_count": self.control_count,
            "positive_count": self.positive_count,
            "threshold": self.threshold,
            "flagged": list(self.flagged),
            "missed": list(self.missed),
            "missed_count": self.missed_count,
            "out_of_support": list(self.out_of_support),
        }


@dataclass(frozen=True)
class ReplayComparison:
    current: ControlReplay
    candidate: ControlReplay

    @property
    def missed_delta(self) -> int:
        return self.candidate.missed_count - self.current.missed_count

    @property
    def allowed(self) -> bool:
        """漏户不增加才允许换卡；持平也允许。"""
        return self.missed_delta <= 0

    def require_allowed(self) -> None:
        if not self.allowed:
            raise ControlGateRejected(
                f"对照户高风险漏户由 {self.current.missed_count} 增至 "
                f"{self.candidate.missed_count}，保留现用卡 {self.current.card_id}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "missed_delta": self.missed_delta,
            "current": self.current.to_dict(),
            "candidate": self.candidate.to_dict(),
        }
