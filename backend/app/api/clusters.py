"""
Novel Object Clustering Endpoints
------------------------------------
GET  /api/clusters            - list current clusters
POST /api/clusters/recompute  - re-run DBSCAN/HDBSCAN over all novel objects
"""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.database import models as db_models
from app.schemas.schemas import ClusterOut
from app.services import clustering

router = APIRouter(prefix="/api/clusters", tags=["clusters"])


def _to_out(c: db_models.NovelCluster) -> ClusterOut:
    return ClusterOut(
        cluster_id=c.id,
        object_count=c.object_count,
        suggested_label=c.suggested_label,
        average_confidence=c.average_confidence,
    )


@router.get("", response_model=list[ClusterOut])
def list_clusters(db: Session = Depends(get_db)):
    clusters = db.query(db_models.NovelCluster).order_by(db_models.NovelCluster.object_count.desc()).all()
    return [_to_out(c) for c in clusters]


@router.post("/recompute", response_model=list[ClusterOut])
def recompute_clusters(db: Session = Depends(get_db)):
    clusters = clustering.recompute_clusters(db)
    return [_to_out(c) for c in clusters]
