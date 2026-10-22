"""
Unknown Object Clustering
----------------------------
Groups visually similar NovelObjects using their stored feature embeddings.
Uses DBSCAN by default (density-based, no need to pre-specify cluster
count — appropriate since we don't know in advance how many distinct novel
categories exist). HDBSCAN can be swapped in via NOVA_CLUSTERING_METHOD.
"""
import logging

import numpy as np
from sklearn.cluster import DBSCAN
from sqlalchemy.orm import Session

from app.config import settings
from app.database import models as db_models

logger = logging.getLogger("nova_x.clustering")


def recompute_clusters(db: Session) -> list[db_models.NovelCluster]:
    """
    Re-runs clustering over every NovelObject that has an embedding and is
    not REJECTED, replacing all existing cluster assignments. Safe to call
    repeatedly (e.g. from a button in the UI) — it's idempotent given the
    same input embeddings.
    """
    objects = (
        db.query(db_models.NovelObject)
        .filter(db_models.NovelObject.embedding.isnot(None))
        .filter(db_models.NovelObject.verification_status != db_models.VerificationStatus.REJECTED)
        .all()
    )

    if len(objects) < settings.DBSCAN_MIN_SAMPLES:
        logger.info(
            "Not enough embedded unknown objects to cluster (%d < %d minimum).",
            len(objects), settings.DBSCAN_MIN_SAMPLES,
        )
        return db.query(db_models.NovelCluster).all()

    embeddings = np.array([obj.embedding for obj in objects], dtype=np.float32)

    if settings.CLUSTERING_METHOD == "hdbscan":
        try:
            import hdbscan  # type: ignore[reportMissingImports]
            clusterer = hdbscan.HDBSCAN(min_cluster_size=settings.DBSCAN_MIN_SAMPLES, metric="euclidean")
            labels = clusterer.fit_predict(embeddings)
        except ImportError:
            logger.warning("hdbscan not installed, falling back to DBSCAN.")
            labels = _run_dbscan(embeddings)
    else:
        labels = _run_dbscan(embeddings)

    # Clear old cluster assignments and cluster rows before writing new ones.
    for obj in objects:
        obj.cluster_id = None
    db.query(db_models.NovelCluster).delete()
    db.flush()

    clusters_by_label: dict[int, db_models.NovelCluster] = {}
    for obj, label in zip(objects, labels):
        if label == -1:
            continue  # DBSCAN noise point — stays unclustered, not an error.
        if label not in clusters_by_label:
            cluster = db_models.NovelCluster(object_count=0, average_confidence=0.0)
            db.add(cluster)
            db.flush()  # assign an id
            clusters_by_label[label] = cluster
        obj.cluster_id = clusters_by_label[label].id

    # Finalize per-cluster stats: suggested label = most common OVD label
    # among members, average_confidence = mean OVD confidence among members.
    for cluster in clusters_by_label.values():
        members = [o for o in objects if o.cluster_id == cluster.id]
        cluster.object_count = len(members)
        labels_seen = [m.ovd_label for m in members if m.ovd_label]
        if labels_seen:
            cluster.suggested_label = max(set(labels_seen), key=labels_seen.count)
        confidences = [m.ovd_confidence for m in members if m.ovd_confidence is not None]
        cluster.average_confidence = round(float(np.mean(confidences)), 4) if confidences else 0.0

    db.commit()
    return list(clusters_by_label.values())


def _run_dbscan(embeddings: np.ndarray) -> np.ndarray:
    clusterer = DBSCAN(
        eps=settings.DBSCAN_EPS,
        min_samples=settings.DBSCAN_MIN_SAMPLES,
        metric="cosine",
    )
    return clusterer.fit_predict(embeddings)
