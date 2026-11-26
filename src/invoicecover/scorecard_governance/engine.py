from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from datetime import date
from typing import Any

from .contracts import (
    HIGH_RISK_THRESHOLD,
    ControlReplay,
    ReviewConclusion,
    parse_date,
)


def _first(mapping: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return None


def normalize_conclusion(
    value: ReviewConclusion | Mapping[str, Any] | Any,
    *,
    batch_id: str | None = None,
) -> ReviewConclusion:
    """Accept dataclasses, mappings, or ordinary attribute objects."""
    if isinstance(value, ReviewConclusion):
        if batch_id and value.batch_id != batch_id:
            return ReviewConclusion(
                debtor=value.debtor,
                label=value.label,
                observed_at=value.observed_at,
                mature_on=value.mature_on,
                conclusion_id=value.conclusion_id,
                batch_id=batch_id,
                reviewed=value.reviewed,
                source=value.source,
                payload=value.payload,
            )
        return value

    if isinstance(value, Mapping):
        data = value
        get = data.get
    else:
        def get(name: str, default: Any = None) -> Any:
            return getattr(value, name, default)

    debtor = _first(data if isinstance(value, Mapping) else {}, "debtor", "account", "customer", "户名", "经销商") if isinstance(value, Mapping) else get("debtor") or get("account") or get("customer") or get("户名")
    if not debtor:
        raise ValueError("复核结论缺少 debtor/户名")

    raw_label = None
    if isinstance(value, Mapping):
        raw_label = _first(data, "label", "target", "late", "is_bad", "bad")
        if raw_label is None:
            outcome = _first(data, "outcome", "conclusion", "结论")
            if isinstance(outcome, str):
                text = outcome.strip()
                if text in {"1", "bad", "late", "逾期", "坏"}:
                    raw_label = 1
                elif text in {"0", "good", "paid", "按期", "正常"}:
                    raw_label = 0
    else:
        raw_label = get("label", get("target", get("late")))

    if raw_label is not None and not isinstance(raw_label, int):
        if isinstance(raw_label, bool):
            raw_label = int(raw_label)
        elif str(raw_label).strip() in {"0", "1"}:
            raw_label = int(raw_label)
        elif str(raw_label).strip() in {"True", "true", "逾期"}:
            raw_label = 1
        elif str(raw_label).strip() in {"False", "false", "按期"}:
            raw_label = 0
        else:
            raw_label = None

    observed_raw = None
    mature_raw = None
    cid = None
    source = "manual_review"
    reviewed = True
    payload: dict[str, Any] = {}
    if isinstance(value, Mapping):
        observed_raw = _first(data, "observed_at", "as_of", "feature_date", "observed")
        mature_raw = _first(data, "mature_on", "matures_on", "observation_end", "completed_at", "outcome_date", "outcome")
        cid = _first(data, "conclusion_id", "id", "record_id", "receipt")
        source = _first(data, "source", "channel") or source
        reviewed = bool(_first(data, "reviewed", "in_review") if _first(data, "reviewed", "in_review") is not None else True)
        payload = {
            k: v
            for k, v in data.items()
            if k
            not in {
                "debtor", "account", "customer", "户名", "经销商", "label", "target",
                "late", "is_bad", "bad", "outcome", "conclusion", "observed_at",
                "as_of", "feature_date", "observed", "mature_on", "matures_on",
                "observation_end", "completed_at", "outcome_date", "conclusion_id",
                "id", "record_id", "batch_id", "reviewed", "source", "fingerprint",
                "mature", "payload",
            }
        }
    else:
        observed_raw = get("observed_at") or get("as_of")
        mature_raw = get("mature_on") or get("observation_end") or get("completed_at")
        cid = get("conclusion_id") or get("id")
        source = get("source", source)
        reviewed = bool(get("reviewed", True))

    return ReviewConclusion(
        debtor=str(debtor),
        label=raw_label,
        observed_at=parse_date(observed_raw),
        mature_on=parse_date(mature_raw),
        conclusion_id=str(cid) if cid is not None else None,
        batch_id=batch_id or (str(get("batch_id")) if not isinstance(value, Mapping) and get("batch_id") else data.get("batch_id") if isinstance(value, Mapping) else None),
        reviewed=reviewed,
        source=str(source),
        payload=payload,
    )


def normalize_conclusions(
    values: Iterable[Any], *, batch_id: str | None = None
) -> tuple[ReviewConclusion, ...]:
    return tuple(normalize_conclusion(item, batch_id=batch_id) for item in values)


def split_maturity(
    conclusions: Iterable[ReviewConclusion], as_of: date | str
) -> tuple[tuple[ReviewConclusion, ...], tuple[ReviewConclusion, ...]]:
    mature: list[ReviewConclusion] = []
    pending: list[ReviewConclusion] = []
    for item in conclusions:
        (mature if item.is_mature(as_of) else pending).append(item)
    return tuple(sorted(mature, key=lambda x: x.stable_id)), tuple(pending)


def _record_field(record: Any, *names: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        return _first(record, *names) if _first(record, *names) is not None else default
    for name in names:
        if hasattr(record, name):
            value = getattr(record, name)
            if value is not None:
                return value
    return default


def replay_controls(
    controls: Iterable[Any],
    probabilities: Mapping[str, float | None] | Callable[[Any], float | None] | None = None,
    *,
    card_id: str = "unknown",
    threshold: float = HIGH_RISK_THRESHOLD,
    unknown_as_miss: bool = False,
) -> ControlReplay:
    """Replay a card on labelled but unreviewed control accounts.

    A supported positive below ``threshold`` is a high-risk miss.  An
    out-of-support probability is reported separately because the existing
    review policy pauses such accounts and places them ahead of normal queues.
    """
    if not 0 < threshold < 1:
        raise ValueError("高风险阈值必须位于 0 和 1 之间")

    flagged: list[str] = []
    missed: list[str] = []
    unsupported: list[str] = []
    positive = 0
    total = 0

    for record in controls:
        total += 1
        debtor = str(_record_field(record, "debtor", "account", "customer", "户名", "经销商"))
        raw_label = _record_field(record, "label", "target", "late", "is_bad", "bad")
        label = int(raw_label) if raw_label is not None else 0
        positive += int(label == 1)

        probability: float | None
        if probabilities is None:
            probability = _record_field(record, "probability", "risk", "p")
        elif callable(probabilities):
            probability = probabilities(record)
        else:
            probability = probabilities.get(debtor)

        if probability is None:
            unsupported.append(debtor)
            if unknown_as_miss and label == 1:
                missed.append(debtor)
        elif probability >= threshold:
            flagged.append(debtor)
        elif label == 1:
            missed.append(debtor)

    return ControlReplay(
        card_id=card_id,
        control_count=total,
        positive_count=positive,
        flagged=tuple(sorted(flagged)),
        missed=tuple(sorted(missed)),
        out_of_support=tuple(sorted(unsupported)),
        threshold=threshold,
    )


def compare_replays(current: ControlReplay, candidate: ControlReplay):
    from .contracts import ReplayComparison

    return ReplayComparison(current=current, candidate=candidate)
