#!/usr/bin/env python3
"""Run one governed scorecard refresh for the bundled September sample.

The script is intentionally explicit about its inputs:
* current book/review seats come from the existing pipeline;
* matured snapshot rows after the review date are review conclusions/control labels;
* only the prior review seats enter the incremental training batch;
* unreviewed debtors are replay-only controls.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from invoicecover.aging_features.service import aggregate, training_periods  # noqa: E402
from invoicecover.credit_terms.contracts import ReviewPolicy  # noqa: E402
from invoicecover.credit_terms.service import dispatch, rows as review_rows  # noqa: E402
from invoicecover.invoice_cutoff.service import from_csv as invoices_from_csv  # noqa: E402
from invoicecover.repayment_clock.service import from_csv as receipts_from_csv  # noqa: E402
from invoicecover.scorecard_governance import (  # noqa: E402
    ScorecardLedger,
    evaluate_card_on_controls,
    run_governed_switch,
)
from invoicecover.term_scorecard.service import evaluate_book, fit  # noqa: E402


def parse_date(value: str) -> date:
    return date.fromisoformat(value)


def load_latest_labels(data_dir: Path, review_date: date, as_of: date) -> dict[str, dict[str, object]]:
    """Load matured labels published after the review date."""
    path = data_dir / "matured_month_end_snapshots.csv"
    latest: dict[str, dict[str, object]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for raw in csv.DictReader(handle):
            observed = date.fromisoformat(raw["observed"])
            outcome = date.fromisoformat(raw["outcome"])
            if outcome <= review_date or outcome > as_of:
                continue
            candidate = {
                "debtor": raw["debtor"],
                "label": int(raw["late"]),
                "observed_at": observed.isoformat(),
                "mature_on": outcome.isoformat(),
            }
            old = latest.get(raw["debtor"])
            if old is None or str(old["observed_at"]) < candidate["observed_at"]:
                latest[raw["debtor"]] = candidate
    return latest


def card_record(model_fit) -> dict:
    card = model_fit.card
    record = asdict(card)
    record["card_id"] = card.model_id
    record["model_id"] = card.model_id
    record["cutoff"] = card.cutoff.isoformat()
    record["metrics"] = model_fit.metrics
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "sample_data")
    parser.add_argument("--ledger", type=Path, default=ROOT / "scorecard_ledger.json")
    parser.add_argument("--report", type=Path, default=ROOT / "scorecard_review_result.json")
    parser.add_argument("--review-date", type=parse_date, default=date(2026, 9, 1))
    parser.add_argument("--as-of", type=parse_date, default=date(2026, 11, 26))
    parser.add_argument("--split", type=parse_date, default=date(2026, 6, 1))
    parser.add_argument("--seats", type=int, default=6)
    parser.add_argument("--batch-id", default="REVIEW-2026-09-01")
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    ledger = ScorecardLedger.load(args.ledger) if args.ledger.exists() else ScorecardLedger(args.ledger)

    baseline_history = training_periods(
        data_dir / "matured_month_end_snapshots.csv", args.review_date
    )
    baseline_fit = fit(baseline_history, args.split)

    # Reconstruct the current book and current-card review sheet.
    invoices = invoices_from_csv(
        data_dir / "september_open_invoices.csv", cutoff=args.review_date
    )
    cash = receipts_from_csv(data_dir / "clearing_house_receipts.csv", invoices)
    accounts = aggregate(invoices.bills, cash.statements, args.review_date)
    policy = ReviewPolicy(args.seats)
    baseline_scores = evaluate_book(baseline_fit, accounts)
    baseline_sheet = dispatch(accounts, baseline_scores, policy)
    baseline_rows = review_rows(baseline_sheet)
    reviewed = {row["debtor"] for row in baseline_rows if row["seat"] is not None}

    labels = load_latest_labels(data_dir, args.review_date, args.as_of)
    conclusions = []
    for debtor in sorted(reviewed):
        label = labels.get(debtor)
        if label is None:
            # The OOD account in the bundled sample has no matured label.
            conclusions.append(
                {
                    "batch_id": args.batch_id,
                    "debtor": debtor,
                    "label": None,
                    "observed_at": args.review_date.isoformat(),
                    "mature_on": None,
                    "source": "account_period_review",
                }
            )
        else:
            conclusions.append(
                {
                    "batch_id": args.batch_id,
                    "source": "account_period_review",
                    **label,
                }
            )

    registration = ledger.register(
        args.batch_id, conclusions, as_of=args.as_of
    )

    full_history = training_periods(
        data_dir / "matured_month_end_snapshots.csv", args.as_of
    )
    vintage_index = {
        (row.debtor, row.as_of): row
        for row in full_history
    }

    def trainer(mature_conclusions):
        incremental = []
        for item in mature_conclusions:
            key = (item.debtor, item.observed_at)
            if key not in vintage_index:
                raise ValueError(f"缺少 {item.debtor} / {item.observed_at} 的特征快照")
            incremental.append(vintage_index[key])
        candidate_fit = fit(baseline_history + tuple(incremental), args.split)
        return card_record(candidate_fit)

    version = ledger.build_version(
        args.batch_id, trainer, as_of=args.as_of
    )
    candidate_fit = fit(
        baseline_history
        + tuple(
            vintage_index[(item.debtor, item.observed_at)]
            for item in ledger.mature_conclusions(args.batch_id, args.as_of)
        ),
        args.split,
    )

    candidate_scores = evaluate_book(candidate_fit, accounts)
    old_probability = {score.debtor: score.probability for score in baseline_scores}
    new_probability = {score.debtor: score.probability for score in candidate_scores}

    controls = [
        {
            "debtor": account.debtor,
            "label": labels[account.debtor]["label"],
        }
        for account in accounts
        if account.debtor not in reviewed and account.debtor in labels
    ]
    comparison = evaluate_card_on_controls(
        controls,
        old_probability,
        new_probability,
        current_card_id=baseline_fit.card.model_id,
        candidate_card_id=candidate_fit.card.model_id,
        threshold=policy.high_risk,
    )

    # Same-feature-period backtest: the August snapshots have matched labels
    # and features, while ``comparison`` replays the current September book.
    control_names = {row["debtor"] for row in controls}
    same_period_rows = tuple(
        row
        for row in full_history
        if row.as_of == date(2026, 8, 1) and row.debtor in control_names
    )
    old_same_period = {
        score.debtor: score.probability
        for score in evaluate_book(baseline_fit, same_period_rows)
    }
    new_same_period = {
        score.debtor: score.probability
        for score in evaluate_book(candidate_fit, same_period_rows)
    }
    same_period_comparison = evaluate_card_on_controls(
        same_period_rows,
        old_same_period,
        new_same_period,
        current_card_id=baseline_fit.card.model_id,
        candidate_card_id=candidate_fit.card.model_id,
        threshold=policy.high_risk,
    )

    # Ensure the factory card is in the ledger as the rollback baseline.
    if ledger.active_card_id is None:
        baseline_record = card_record(baseline_fit)
        ledger.cards[baseline_fit.card.model_id] = baseline_record
        ledger.active_card_id = baseline_fit.card.model_id
        ledger.save()

    decision = run_governed_switch(
        ledger,
        args.batch_id,
        comparison=comparison,
    )

    active_fit = candidate_fit if comparison.allowed else baseline_fit
    active_scores = candidate_scores if comparison.allowed else baseline_scores
    active_sheet = dispatch(accounts, active_scores, policy)
    final_rows = review_rows(active_sheet)

    report = {
        "as_of": args.as_of.isoformat(),
        "review_date": args.review_date.isoformat(),
        "batch_id": args.batch_id,
        "seats": args.seats,
        "registration": registration,
        "training_version": {
            "version_id": version.version_id,
            "revision": version.revision,
            "build_count": version.build_count,
            "input_hash": version.input_hash,
            "mature_count": version.mature_count,
            "pending_count": version.pending_count,
        },
        "switch": decision,
        "same_period_replay": same_period_comparison.to_dict(),
        "active_model_id": active_fit.card.model_id,
        "previous_model_id": baseline_fit.card.model_id,
        "review_rows": final_rows,
        "control_accounts": controls,
        "ledger_path": str(args.ledger),
    }
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
