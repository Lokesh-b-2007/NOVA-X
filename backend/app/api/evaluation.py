"""
GET /api/evaluation — model evaluation metrics for the Model Evaluation page.

See services/evaluation.py for the honesty policy: metric blocks are None
(never fabricated) when their required labeled eval data isn't present.
"""
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.schemas.schemas import EvaluationResponse
from app.services import evaluation

router = APIRouter(prefix="/api", tags=["evaluation"])


@router.get("/evaluation", response_model=EvaluationResponse)
def get_evaluation(db: Session = Depends(get_db)):
    known = evaluation.evaluate_known_detector()
    ood = evaluation.evaluate_ood(db)
    unknown = evaluation.evaluate_unknown_detection(db)
    ovd = evaluation.evaluate_ovd(db)
    ood_counts = evaluation.get_ood_sample_counts(db)
    ovd_sample_count = evaluation.get_ovd_sample_count(db)
    ovd_sample_source = evaluation.get_ovd_sample_source(db)

    missing = []
    if known is None:
        missing.append(
            "known-detector metrics (from data/known_dataset/eval/data.yaml; "
            "run `python scripts/run_known_eval.py` from backend/)"
        )
    if ood is None:
        missing.append(
            "OOD metrics (run `python scripts/run_ood_eval.py` from backend/ against "
            "the downloaded, ground-truth-labeled COCO128 boxes)"
        )
    if unknown is None:
        missing.append("unknown-detection metrics (computed by the same run_ood_eval.py script)")
    if ovd is None:
        missing.append(
            "open-vocabulary metrics (requires human-confirmed examples or custom folder-labeled photos with saved OWLv2 predictions)"
        )

    caveats = [
        "COCO128 reuses its training sample and the base detector was trained on COCO; these are sanity-check metrics, not held-out benchmark results.",
        "Known vs. unknown ground truth uses NOVA-X's configured and registered class allow-list; all other annotated COCO categories are treated as unknown."
    ]
    if ood is not None and ood_counts and ood_counts.get("embedding_extraction_available") is False:
        caveats.append(
            "The detector could not produce embeddings in this run, so hybrid novelty scoring fell back to confidence-only scoring."
        )
    if ovd is not None and ovd_sample_source == "human-confirmed crops":
        caveats.append(
            "Open-vocabulary accuracy uses human-confirmed repository examples. Reviewers may have seen OWLv2 suggestions, so this is not an independent held-out test."
        )
    elif ovd is not None:
        caveats.append(
            "Open-vocabulary accuracy uses custom folder-labeled photos and is not a held-out benchmark."
        )

    notes = (
        "Metrics were computed from real annotated and human-confirmed data."
        if not missing
        else "Some metrics are unavailable until their evaluation data is prepared: " + "; ".join(missing) + "."
    )

    return EvaluationResponse(
        known_metrics=known,
        ood_metrics=ood,
        unknown_metrics=unknown,
        ovd_metrics=ovd,
        ood_sample_count=ood_counts["ood"] if ood_counts else None,
        ood_ground_truth_unknown_count=ood_counts["unknown"] if ood_counts else None,
        ovd_sample_count=ovd_sample_count or None,
        ovd_sample_source=ovd_sample_source,
        evaluation_caveats=caveats,
        evaluated_at=datetime.now(timezone.utc),
        notes=notes,
    )
