from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .contracts import (
    ConclusionConflict,
    ControlGateRejected,
    ControlReplay,
    GovernanceError,
    ReplayComparison,
    ReviewConclusion,
    ScorecardSwitchError,
    content_hash,
    parse_date,
)
from .engine import normalize_conclusion, normalize_conclusions, split_maturity


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class TrainingVersion:
    """同一复核批次对应的唯一逻辑训练版本。"""

    batch_id: str
    version_id: str
    input_hash: str
    mature_count: int
    pending_count: int
    card: dict[str, Any]
    first_built_at: str
    last_built_at: str
    build_count: int = 1
    revision: int = 1
    source_ids: tuple[str, ...] = ()
    prior_input_hashes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["source_ids"] = list(self.source_ids)
        data["prior_input_hashes"] = list(self.prior_input_hashes)
        return data


class ScorecardLedger:
    """复核结论、训练版本与评分卡切换历史的追加型台账。

    可选 ``path`` 指向 JSON 文件。保存采用同目录临时文件 + ``os.replace``，
    因此序列化或临时文件阶段失败时不会破坏旧台账。
    """

    schema = "invoicecover.scorecard_ledger/v1"

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else None
        self.batches: dict[str, dict[str, ReviewConclusion]] = {}
        self.training_versions: dict[str, TrainingVersion] = {}
        self.cards: dict[str, dict[str, Any]] = {}
        self.active_card_id: str | None = None
        self.previous_card_id: str | None = None
        self.switch_history: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []

    # ---------- persistence ----------
    @classmethod
    def load(cls, path: str | Path) -> "ScorecardLedger":
        target = Path(path)
        ledger = cls(target)
        if not target.exists():
            return ledger
        raw = json.loads(target.read_text(encoding="utf-8"))
        if raw.get("schema") != cls.schema:
            raise GovernanceError("评分卡台账版本无法识别")
        for batch_id, records in raw.get("batches", {}).items():
            ledger.batches[batch_id] = {
                item["conclusion_id"]: normalize_conclusion(item, batch_id=batch_id)
                for item in records
            }
        for batch_id, item in raw.get("training_versions", {}).items():
            ledger.training_versions[batch_id] = TrainingVersion(
                batch_id=item["batch_id"],
                version_id=item["version_id"],
                input_hash=item["input_hash"],
                mature_count=item["mature_count"],
                pending_count=item["pending_count"],
                card=item["card"],
                first_built_at=item["first_built_at"],
                last_built_at=item["last_built_at"],
                build_count=item.get("build_count", 1),
                revision=item.get("revision", 1),
                source_ids=tuple(item.get("source_ids", [])),
                prior_input_hashes=tuple(item.get("prior_input_hashes", [])),
            )
        ledger.cards = dict(raw.get("cards", {}))
        ledger.active_card_id = raw.get("active_card_id")
        ledger.previous_card_id = raw.get("previous_card_id")
        ledger.switch_history = list(raw.get("switch_history", []))
        ledger.events = list(raw.get("events", []))
        return ledger

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "batches": {
                batch_id: [item.to_dict() for item in records.values()]
                for batch_id, records in sorted(self.batches.items())
            },
            "training_versions": {
                key: item.to_dict() for key, item in sorted(self.training_versions.items())
            },
            "cards": self.cards,
            "active_card_id": self.active_card_id,
            "previous_card_id": self.previous_card_id,
            "switch_history": self.switch_history,
            "events": self.events,
        }

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=str,
        )
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise

    # ---------- conclusion registration ----------
    def register(
        self,
        batch_id: str,
        conclusions: Iterable[Any],
        *,
        as_of: date | str | None = None,
    ) -> dict[str, int]:
        if not batch_id:
            raise GovernanceError("复核批次必须有 batch_id")
        bucket = self.batches.setdefault(str(batch_id), {})
        inserted = updated = unchanged = 0
        for raw in conclusions:
            item = normalize_conclusion(raw, batch_id=str(batch_id))
            key = item.stable_id
            old = bucket.get(key)
            if old is None:
                bucket[key] = item
                inserted += 1
            elif old.fingerprint() == item.fingerprint():
                unchanged += 1
            else:
                # 同户同特征日的标签不能被后到记录静默覆盖。补录新字段时
                # 应给出新的 conclusion_id 或显式更正批次。
                raise ConclusionConflict(
                    f"批次 {batch_id} 的结论 {key} 与已登记内容冲突"
                )
        mature_count = pending_count = 0
        if as_of is not None:
            mature, pending = split_maturity(bucket.values(), as_of)
            mature_count, pending_count = len(mature), len(pending)
        self.events.append(
            {
                "at": _now(),
                "type": "register",
                "batch_id": batch_id,
                "inserted": inserted,
                "unchanged": unchanged,
                "total": len(bucket),
                "mature": mature_count,
                "pending": pending_count,
            }
        )
        self.save()
        return {
            "inserted": inserted,
            "updated": updated,
            "unchanged": unchanged,
            "total": len(bucket),
            "mature": mature_count,
            "pending": pending_count,
        }

    def conclusions(self, batch_id: str) -> tuple[ReviewConclusion, ...]:
        return tuple(sorted(self.batches[batch_id].values(), key=lambda x: x.stable_id))

    def mature_conclusions(
        self, batch_id: str, as_of: date | str
    ) -> tuple[ReviewConclusion, ...]:
        mature, _ = split_maturity(self.conclusions(batch_id), as_of)
        return mature

    def pending_conclusions(
        self, batch_id: str, as_of: date | str
    ) -> tuple[ReviewConclusion, ...]:
        _, pending = split_maturity(self.conclusions(batch_id), as_of)
        return pending

    # ---------- training version ----------
    def build_version(
        self,
        batch_id: str,
        trainer: Callable[[tuple[ReviewConclusion, ...]], dict[str, Any]],
        *,
        as_of: date | str,
        force: bool = False,
    ) -> TrainingVersion:
        mature = self.mature_conclusions(batch_id, as_of)
        pending = self.pending_conclusions(batch_id, as_of)
        if not mature:
            raise GovernanceError("没有走完观察期的结论可入训；观察期内记录只能登记")

        input_payload = [
            {
                "id": item.stable_id,
                "debtor": item.debtor,
                "label": item.label,
                "observed_at": item.observed_at,
                "mature_on": item.mature_on,
                "fingerprint": item.fingerprint(),
            }
            for item in mature
        ]
        input_hash = content_hash(input_payload, length=16)
        existing = self.training_versions.get(batch_id)
        if existing is not None and not force and existing.input_hash == input_hash:
            existing.build_count += 1
            self.save()
            return existing

        card = trainer(mature)
        if not isinstance(card, dict):
            raise GovernanceError("trainer 必须返回可 JSON 化的评分卡记录字典")
        card.setdefault("card_id", card.get("model_id") or content_hash(card, 14))
        now = _now()
        if existing is None:
            version = TrainingVersion(
                batch_id=batch_id,
                version_id=f"TRAIN-{batch_id}",
                input_hash=input_hash,
                mature_count=len(mature),
                pending_count=len(pending),
                card=card,
                first_built_at=now,
                last_built_at=now,
                source_ids=tuple(item.stable_id for item in mature),
            )
        else:
            prior = tuple(dict.fromkeys(existing.prior_input_hashes + (existing.input_hash,)))
            version = TrainingVersion(
                batch_id=batch_id,
                version_id=existing.version_id,
                input_hash=input_hash,
                mature_count=len(mature),
                pending_count=len(pending),
                card=card,
                first_built_at=existing.first_built_at,
                last_built_at=now,
                build_count=existing.build_count + 1,
                revision=existing.revision + 1,
                source_ids=tuple(item.stable_id for item in mature),
                prior_input_hashes=prior,
            )
        # 同一 batch_id 永远只对应一个逻辑训练版本对象；重跑是幂等返回，
        # 补录则更新该版本而不是新增第二个 batch 版本键。
        self.training_versions[batch_id] = version
        self.events.append(
            {
                "at": now,
                "type": "build_version",
                "batch_id": batch_id,
                "version_id": version.version_id,
                "revision": version.revision,
                "build_count": version.build_count,
                "input_hash": input_hash,
                "mature": len(mature),
                "pending": len(pending),
            }
        )
        self.save()
        return version

    # ---------- promotion / switch ----------
    def promote(
        self,
        card: dict[str, Any] | TrainingVersion | str,
        *,
        comparison: ReplayComparison | None = None,
        commit: Callable[[dict[str, Any]], None] | None = None,
        switched_at: str | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        if comparison is not None:
            comparison.require_allowed()
        elif not force:
            raise GovernanceError("换卡必须附带未复核对照户回放结果，或显式 force")

        card_record, batch_id = self._resolve_card(card)
        card_id = str(card_record.get("card_id") or card_record.get("model_id"))
        if not card_id:
            raise GovernanceError("评分卡记录缺少 card_id/model_id")
        card_record = {**card_record, "card_id": card_id}

        if self.active_card_id == card_id:
            self.cards.setdefault(card_id, card_record)
            self.save()
            return {
                "switched": False,
                "already_active": True,
                "active_card_id": card_id,
                "previous_card_id": self.previous_card_id,
                "comparison": comparison.to_dict() if comparison is not None else None,
            }

        previous_id = self.active_card_id
        if commit is not None:
            try:
                commit(card_record)
            except Exception as exc:
                self.events.append(
                    {
                        "at": _now(),
                        "type": "switch_failed",
                        "attempted_card_id": card_id,
                        "active_card_id": previous_id,
                        "error": str(exc),
                    }
                )
                self.save()
                raise ScorecardSwitchError(f"评分卡切换失败，保留 {previous_id}") from exc

        # 先执行可能失败的外部提交，再修改内部 active 指针。旧卡始终保留在
        # cards 中，previous_card_id 只记录紧邻上一版。
        self.cards[card_id] = card_record
        self.previous_card_id = previous_id
        self.active_card_id = card_id
        event = {
            "at": switched_at or _now(),
            "type": "switch_card",
            "batch_id": batch_id,
            "card_id": card_id,
            "previous_card_id": previous_id,
            "comparison": comparison.to_dict() if comparison is not None else None,
        }
        self.switch_history.append(event)
        self.events.append(event)
        self.save()
        return {
            "switched": True,
            "active_card_id": card_id,
            "previous_card_id": previous_id,
            "comparison": comparison.to_dict() if comparison is not None else None,
        }

    def reject_switch(
        self,
        attempted_card: dict[str, Any] | TrainingVersion | str,
        comparison: ReplayComparison,
        *,
        reason: str | None = None,
    ) -> dict[str, Any]:
        card_record, batch_id = self._resolve_card(attempted_card)
        card_id = str(card_record.get("card_id") or card_record.get("model_id") or "unknown")
        result = {
            "at": _now(),
            "type": "retain_current_card",
            "batch_id": batch_id,
            "attempted_card_id": card_id,
            "active_card_id": self.active_card_id,
            "previous_card_id": self.previous_card_id,
            "missed_delta": comparison.missed_delta,
            "reason": reason
            or f"对照户高风险漏户增加 {comparison.missed_delta} 户，保留现用卡",
            "comparison": comparison.to_dict(),
        }
        self.events.append(result)
        self.save()
        return {
            "switched": False,
            "active_card_id": self.active_card_id,
            "reason": result["reason"],
            "comparison": comparison.to_dict(),
        }

    def _resolve_card(
        self, value: dict[str, Any] | TrainingVersion | str
    ) -> tuple[dict[str, Any], str | None]:
        if isinstance(value, TrainingVersion):
            return value.card, value.batch_id
        if isinstance(value, str):
            version = self.training_versions.get(value)
            if version is not None:
                return version.card, value
            if value in self.cards:
                return self.cards[value], None
            raise GovernanceError(f"找不到评分卡或训练批次 {value}")
        if isinstance(value, dict):
            return value, value.get("batch_id")
        raise GovernanceError("无法识别的评分卡参数")

    def active_card(self) -> dict[str, Any] | None:
        if self.active_card_id is None:
            return None
        return self.cards[self.active_card_id]

    @property
    def versions(self) -> tuple[TrainingVersion, ...]:
        """Logical training versions, one per review batch."""
        return tuple(self.training_versions.values())

    @property
    def history(self) -> list[dict[str, Any]]:
        return self.switch_history

    def card_history(self) -> tuple[dict[str, Any], ...]:
        """Return every card touched by a successful switch, newest last."""
        return tuple(self.cards[event["card_id"]] for event in self.switch_history)
