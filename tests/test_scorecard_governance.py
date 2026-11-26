from datetime import date

import pytest

from invoicecover.scorecard_governance import (
    ScorecardLedger,
    evaluate_card_on_controls,
)
from invoicecover.scorecard_governance.contracts import ScorecardSwitchError


def conclusion(debtor, label, mature_on="2026-09-04", cid=None):
    return {
        "conclusion_id": cid or f"id-{debtor}",
        "debtor": debtor,
        "label": label,
        "observed_at": "2026-08-01",
        "mature_on": mature_on,
    }


def test_observation_period_records_are_registered_but_not_trained(tmp_path):
    ledger = ScorecardLedger(tmp_path / "cards.json")
    ledger.register(
        "B1",
        [
            conclusion("甲", 1, "2026-09-04"),
            conclusion("乙", 0, "2026-12-15"),
            {"debtor": "丙", "label": None, "mature_on": None},
        ],
        as_of=date(2026, 11, 26),
    )

    assert [x.debtor for x in ledger.mature_conclusions("B1", date(2026, 11, 26))] == ["甲"]
    assert len(ledger.pending_conclusions("B1", date(2026, 11, 26))) == 2

    def trainer(records):
        assert len(records) == 1
        return {"model_id": "candidate-1", "n": len(records)}

    version = ledger.build_version("B1", trainer, as_of=date(2026, 11, 26))
    assert version.mature_count == 1 and version.pending_count == 2


def test_rerun_and_supplement_keep_one_logical_batch_version(tmp_path):
    ledger = ScorecardLedger(tmp_path / "cards.json")
    rows = [conclusion("甲", 1), conclusion("乙", 0)]
    ledger.register("B1", rows)
    first = ledger.build_version("B1", lambda xs: {"model_id": f"m-{len(xs)}"}, as_of="2026-11-26")
    second = ledger.build_version("B1", lambda xs: {"model_id": f"m-{len(xs)}"}, as_of="2026-11-26")
    assert first.version_id == second.version_id == "TRAIN-B1"
    assert len(ledger.training_versions) == 1
    assert second.build_count == 2 and second.revision == 1

    ledger.register("B1", [conclusion("丁", 1, cid="id-丁")])
    third = ledger.build_version("B1", lambda xs: {"model_id": f"m-{len(xs)}"}, as_of="2026-11-26")
    assert third.version_id == first.version_id
    assert len(ledger.training_versions) == 1
    assert third.revision == 2 and third.mature_count == 3


def test_same_conclusion_identity_is_idempotent_but_conflict_is_rejected():
    ledger = ScorecardLedger()
    row = conclusion("甲", 1, cid="same")
    ledger.register("B", [row])
    ledger.register("B", [row])
    assert len(ledger.conclusions("B")) == 1
    with pytest.raises(ValueError, match="冲突"):
        ledger.register("B", [{**row, "label": 0}])


def test_control_replay_blocks_only_increase_in_misses():
    controls = [
        {"debtor": "A", "label": 1},
        {"debtor": "B", "label": 1},
        {"debtor": "C", "label": 0},
    ]
    equal = evaluate_card_on_controls(
        controls,
        {"A": 0.1, "B": 0.9, "C": 0.9},
        {"A": 0.2, "B": 0.9, "C": 0.1},
        current_card_id="old",
        candidate_card_id="equal",
    )
    assert equal.current.missed == ("A",)
    assert equal.allowed is True

    worse = evaluate_card_on_controls(
        controls,
        {"A": 0.9, "B": 0.9, "C": 0.1},
        {"A": 0.1, "B": 0.9, "C": 0.1},
        current_card_id="old",
        candidate_card_id="worse",
    )
    assert worse.candidate.missed == ("A",)
    assert worse.missed_delta == 1 and worse.allowed is False


def test_failed_commit_and_gate_rejection_keep_previous_card(tmp_path):
    path = tmp_path / "ledger.json"
    ledger = ScorecardLedger(path)
    ledger.cards["old"] = {"card_id": "old"}
    ledger.active_card_id = "old"
    ledger.register("B", [conclusion("甲", 1), conclusion("乙", 0)])
    ledger.build_version("B", lambda xs: {"card_id": "new", "model_id": "new"}, as_of="2026-11-26")

    controls = [{"debtor": "A", "label": 1}]
    bad_gate = evaluate_card_on_controls(
        controls, {"A": 0.9}, {"A": 0.1}, candidate_card_id="new"
    )
    rejected = ledger.reject_switch("B", bad_gate)
    assert rejected["switched"] is False
    assert ledger.active_card_id == "old"

    good_gate = evaluate_card_on_controls(
        controls, {"A": 0.1}, {"A": 0.9}, candidate_card_id="new"
    )

    def broken_commit(_card):
        raise RuntimeError("atomic switch failed")

    with pytest.raises(ScorecardSwitchError):
        ledger.promote("B", comparison=good_gate, commit=broken_commit)
    assert ledger.active_card_id == "old"
    assert "new" not in ledger.cards or ledger.cards.get("new", {}).get("card_id") != "new"

    ledger.promote("B", comparison=good_gate)
    assert ledger.active_card_id == "new"
    assert ledger.previous_card_id == "old"
    assert set(ledger.cards) == {"old", "new"}
    assert len(ledger.switch_history) == 1

    reloaded = ScorecardLedger.load(path)
    assert reloaded.active_card_id == "new"
    assert reloaded.previous_card_id == "old"
    assert "old" in reloaded.cards and "new" in reloaded.cards


def test_promote_rerun_is_idempotent():
    ledger = ScorecardLedger()
    ledger.cards["old"] = {"card_id": "old"}
    ledger.active_card_id = "old"
    comparison = evaluate_card_on_controls(
        [{"debtor": "A", "label": 1}], {"A": 0.1}, {"A": 0.9}, candidate_card_id="new"
    )
    ledger.promote({"card_id": "new"}, comparison=comparison)
    again = ledger.promote({"card_id": "new"}, comparison=comparison)
    assert again["already_active"] is True
    assert ledger.previous_card_id == "old"
    assert len(ledger.switch_history) == 1
