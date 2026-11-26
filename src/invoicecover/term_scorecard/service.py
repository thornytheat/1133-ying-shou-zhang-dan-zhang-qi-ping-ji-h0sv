from datetime import date

from ..aging_features.contracts import FeatureScaler
from .contracts import CalibrationBand, ModelFit, Scorecard
from .engine import calibrate, diagnostics, fit, isotonic, logistic, predict


def card_from_record(record: dict) -> Scorecard:
    """Rebuild a frozen scorecard from its JSON ledger record."""
    scaler_record = record["scaler"]
    scaler = FeatureScaler(
        center=tuple(float(x) for x in scaler_record["center"]),
        scales=tuple(float(x) for x in scaler_record["scales"]),
        minimum=tuple(float(x) for x in scaler_record["minimum"]),
        maximum=tuple(float(x) for x in scaler_record["maximum"]),
    )
    bands = tuple(
        CalibrationBand(
            maximum_score=float(band["maximum_score"]),
            loans=int(band["loans"]),
            events=int(band["events"]),
        )
        for band in record.get("bands", ())
    )
    cutoff = record["cutoff"]
    return Scorecard(
        scaler=scaler,
        weights=tuple(float(x) for x in record["weights"]),
        intercept=float(record["intercept"]),
        bands=bands,
        cutoff=date.fromisoformat(cutoff) if isinstance(cutoff, str) else cutoff,
        train_count=int(record["train_count"]),
        calibration_count=int(record["calibration_count"]),
        model_id=str(record.get("model_id") or record["card_id"]),
    )


def model_fit_from_record(record: dict) -> ModelFit:
    card = card_from_record(record)
    calibration_rows = tuple(
        {
            "上限": round(band.maximum_score, 5),
            "样本": band.loans,
            "逾期": band.events,
            "校准概率": round(band.probability, 5),
        }
        for band in card.bands
    )
    return ModelFit(card, dict(record.get("metrics", {})), calibration_rows)


def evaluate_book(model: ModelFit, debtors):
    return tuple(predict(model.card, debt) for debt in debtors)


def evidence(model: ModelFit):
    return {
        "title": "账期建模",
        "summary": f"训练 {model.card.train_count} 户月，校准 {model.card.calibration_count} 户月",
        "records": list(model.calibration_rows),
        "metrics": model.metrics,
        "model_id": model.card.model_id,
    }


def model_coverage(scores):
    counts = {"可评分": 0, "训练分布外": 0}
    for row in scores:
        counts[row.state] += 1
    return counts
