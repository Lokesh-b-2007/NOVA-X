import { useEffect, useState } from "react";
import { RefreshCw, Loader2 } from "lucide-react";
import { listClusters, recomputeClusters } from "../services/api";
import PageHeader from "../components/PageHeader";
import { ErrorNotice, EmptyState } from "../components/Feedback";

export default function Clusters() {
  const [clusters, setClusters] = useState([]);
  const [loading, setLoading] = useState(true);
  const [recomputing, setRecomputing] = useState(false);
  const [error, setError] = useState(null);

  const load = () => {
    setLoading(true);
    listClusters()
      .then((res) => setClusters(res.data))
      .catch((err) => setError(err.response?.data?.detail || "Could not load clusters."))
      .finally(() => setLoading(false));
  };

  useEffect(load, []);

  const handleRecompute = async () => {
    setRecomputing(true);
    setError(null);
    try {
      const res = await recomputeClusters();
      setClusters(res.data);
    } catch (err) {
      setError(err.response?.data?.detail || "Recompute failed.");
    } finally {
      setRecomputing(false);
    }
  };

  return (
    <div>
      <PageHeader
        title="Novel Clusters"
        subtitle="Visually similar unknown objects grouped by embedding distance (DBSCAN)."
        action={
          <button
            onClick={handleRecompute}
            disabled={recomputing}
            className="flex items-center gap-2 bg-base-800 border border-line text-sm px-3 py-1.5 rounded hover:border-known/40 transition-colors focus-ring disabled:opacity-50"
          >
            {recomputing ? (
              <Loader2 size={14} className="animate-spin" />
            ) : (
              <RefreshCw size={14} />
            )}
            Recompute
          </button>
        }
      />

      <ErrorNotice message={error} />

      {loading ? (
        <p className="text-sm text-text-muted">Loading…</p>
      ) : clusters.length === 0 ? (
        <EmptyState
          title="No clusters yet"
          description="Clusters form once there are enough unknown objects with embeddings. Try Recompute after collecting a few unknowns."
        />
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {clusters.map((c, i) => (
            <div key={c.cluster_id} className="bg-base-800 border border-line rounded p-4">
              <p className="text-[11px] text-text-faint mb-1">Cluster #{String(i + 1).padStart(2, "0")}</p>
              <p className="font-display text-base font-medium mb-3">
                {c.suggested_label || "Unlabeled group"}
              </p>
              <div className="flex justify-between text-sm">
                <span className="text-text-muted">Objects</span>
                <span className="tabular-nums text-text-primary">{c.object_count}</span>
              </div>
              <div className="flex justify-between text-sm mt-1">
                <span className="text-text-muted">Avg. OVD confidence</span>
                <span className="tabular-nums text-text-primary">
                  {Math.round(c.average_confidence * 100)}%
                </span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
