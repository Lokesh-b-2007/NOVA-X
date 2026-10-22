"""
GET /api/analytics — chart-ready aggregates for the Analytics Dashboard page.

Every field is computed from real rows in the database. If there is no
data yet (fresh install), arrays/counts are simply empty/zero rather than
fabricated — the frontend renders an empty-state instead of fake charts.
"""
from collections import Counter, defaultdict

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.database import models as db_models
from app.schemas.schemas import AnalyticsResponse

router = APIRouter(prefix="/api", tags=["analytics"])


@router.get("/analytics", response_model=AnalyticsResponse)
def get_analytics(db: Session = Depends(get_db)):
    detections = db.query(db_models.Detection).all()
    novel_objects = db.query(db_models.NovelObject).all()
    clusters = db.query(db_models.NovelCluster).all()

    # 1. Known vs Unknown
    known_vs_unknown = {"known": 0, "unknown": 0, "suspicious": 0}
    for d in detections:
        key = d.novelty_status.value.lower() if d.novelty_status else "known"
        if key in known_vs_unknown:
            known_vs_unknown[key] += 1

    # 2. Unknown objects over time (by day)
    by_day = defaultdict(int)
    for obj in novel_objects:
        day = obj.detected_at.strftime("%Y-%m-%d") if obj.detected_at else "unknown"
        by_day[day] += 1
    unknown_over_time = [
        {"date": day, "count": count} for day, count in sorted(by_day.items())
    ]

    # 3. Novelty score distribution (10 buckets of width 0.1)
    buckets = [0] * 10
    for d in detections:
        score = d.novelty_score or 0.0
        idx = min(9, int(score * 10))
        buckets[idx] += 1
    novelty_score_distribution = [
        {"bucket": f"{i / 10:.1f}-{(i + 1) / 10:.1f}", "count": buckets[i]} for i in range(10)
    ]

    # 4. Most frequent unknown classes (by OVD label)
    label_counter = Counter(obj.ovd_label for obj in novel_objects if obj.ovd_label)
    most_frequent_unknown_classes = [
        {"label": label, "count": count} for label, count in label_counter.most_common(10)
    ]

    # 5. Cluster distribution
    cluster_distribution = [
        {
            "cluster_id": c.id,
            "suggested_label": c.suggested_label or "unlabeled",
            "object_count": c.object_count,
        }
        for c in clusters
    ]

    return AnalyticsResponse(
        known_vs_unknown=known_vs_unknown,
        unknown_over_time=unknown_over_time,
        novelty_score_distribution=novelty_score_distribution,
        most_frequent_unknown_classes=most_frequent_unknown_classes,
        cluster_distribution=cluster_distribution,
    )
