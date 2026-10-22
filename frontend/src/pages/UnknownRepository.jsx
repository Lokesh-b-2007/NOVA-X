import { useEffect, useState } from "react";
import { Trash2 } from "lucide-react";
import { listUnknownObjects, deleteUnknownObject, clearUnknownObjects } from "../services/api";
import PageHeader from "../components/PageHeader";
import StatusBadge from "../components/StatusBadge";
import { ErrorNotice, EmptyState } from "../components/Feedback";

const FILTERS = ["ALL", "PENDING", "CONFIRMED", "REJECTED"];

export default function UnknownRepository() {
  const [objects, setObjects] = useState([]);
  const [filter, setFilter] = useState("ALL");
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [clearing, setClearing] = useState(false);
  const [notice, setNotice] = useState(null);

  const load = () => {
    setLoading(true);
    listUnknownObjects(filter === "ALL" ? undefined : filter)
      .then((res) => setObjects(res.data))
      .catch((err) =>
        setError(err.response?.data?.detail || "Could not load the unknown object repository.")
      )
      .finally(() => setLoading(false));
  };

  useEffect(load, [filter]);

  const handleDelete = async (id) => {
    try {
      await deleteUnknownObject(id);
      setObjects((prev) => prev.filter((o) => o.id !== id));
    } catch (err) {
      setError(err.response?.data?.detail || "Could not delete this object.");
    }
  };

  const handleClearAll = async () => {
    const confirmed = window.confirm(
      "Clear the entire repository? This permanently deletes all objects, their verified training examples, saved crops, and clusters. Detection history will be kept."
    );
    if (!confirmed) return;

    setClearing(true);
    setError(null);
    setNotice(null);
    try {
      const res = await clearUnknownObjects();
      setObjects([]);
      setNotice(`Cleared ${res.data.deleted} object${res.data.deleted === 1 ? "" : "s"}.`);
    } catch (err) {
      setError(err.response?.data?.detail || "Could not clear the repository.");
    } finally {
      setClearing(false);
    }
  };

  return (
    <div>
      <PageHeader
        title="Unknown Object Repository"
        subtitle="Every object the OOD module flagged as outside the known-class distribution."
      />

      <ErrorNotice message={error} />
      {notice && <p className="text-sm text-known mb-4">{notice}</p>}

      <div className="flex flex-wrap items-center gap-2 mb-5">
        {FILTERS.map((f) => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={`px-3 py-1.5 rounded-sm text-[12px] font-medium transition-colors focus-ring ${
              filter === f
                ? "bg-base-700 text-text-primary"
                : "bg-base-800 text-text-muted hover:text-text-primary"
            }`}
          >
            {f}
          </button>
        ))}
        <button
          onClick={handleClearAll}
          disabled={loading || clearing}
          className="ml-auto flex items-center gap-2 px-3 py-1.5 rounded-sm text-[12px] font-medium bg-danger/10 text-danger hover:bg-danger/20 transition-colors focus-ring disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <Trash2 size={14} />
          {clearing ? "Clearing…" : "Clear all"}
        </button>
      </div>

      {loading ? (
        <p className="text-sm text-text-muted">Loading…</p>
      ) : objects.length === 0 ? (
        <EmptyState
          title="No unknown objects yet"
          description="Run some detections on the Detect page — anything flagged UNKNOWN will show up here."
        />
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {objects.map((obj) => (
            <div key={obj.id} className="bg-base-800 border border-line rounded p-4">
              <div className="flex items-center justify-between mb-2">
                <StatusBadge status={obj.verification_status} />
                <button
                  onClick={() => handleDelete(obj.id)}
                  className="text-text-faint hover:text-danger transition-colors focus-ring"
                  title="Delete"
                >
                  <Trash2 size={14} />
                </button>
              </div>
              <p className="font-display text-sm font-medium mb-1">
                {obj.confirmed_label || obj.ovd_label || "Unlabeled"}
              </p>
              <p className="text-[12px] text-text-muted mb-3">
                Novelty {Math.round(obj.novelty_score * 100)}%
                {obj.ovd_confidence != null &&
                  ` · OVD confidence ${Math.round(obj.ovd_confidence * 100)}%`}
              </p>
              <p className="text-[11px] text-text-faint">
                Seen {obj.occurrence_count}×
                {obj.cluster_id ? " · clustered" : " · unclustered"}
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
