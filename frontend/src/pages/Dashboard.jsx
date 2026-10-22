import { useEffect, useState } from "react";
import { getDashboard } from "../services/api";
import PageHeader from "../components/PageHeader";
import StatCard from "../components/StatCard";
import { ErrorNotice } from "../components/Feedback";

export default function Dashboard() {
  const [stats, setStats] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    getDashboard()
      .then((res) => !cancelled && setStats(res.data))
      .catch((err) =>
        !cancelled &&
        setError(
          err.response?.data?.detail ||
            "Could not reach the NOVA-X backend. Is it running on port 8001?"
        )
      )
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div>
      <PageHeader
        title="Dashboard"
        subtitle="Live system state across every detection, novelty and learning event."
      />

      <ErrorNotice message={error} />

      {loading ? (
        <p className="text-sm text-text-muted">Loading…</p>
      ) : stats ? (
        <>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
            <StatCard
              label="Total detections analyzed"
              value={stats.total_images_analyzed}
              hint="One row per detected object, across all uploads and live frames"
            />
            <StatCard
              label="Known objects detected"
              value={stats.known_objects_detected}
              accent="text-known"
            />
            <StatCard
              label="Unknown objects detected"
              value={stats.unknown_objects_detected}
              accent="text-novel"
            />
            <StatCard
              label="Novel objects discovered"
              value={stats.novel_objects_discovered}
              hint="Unique crops filed to the Novel Object Repository"
            />
            <StatCard
              label="Confirmed new classes"
              value={stats.confirmed_new_classes}
              accent="text-known"
              hint="Classes added via continual learning"
            />
            <StatCard
              label="Active model version"
              value={stats.active_model_version}
            />
          </div>

          <div className="bg-base-800 border border-line rounded p-5">
            <h2 className="font-display text-sm font-medium mb-2">Where to go next</h2>
            <p className="text-sm text-text-muted leading-relaxed">
              Head to <span className="text-text-primary">Detect</span> to run an image through
              the pipeline, review anything flagged unknown in the{" "}
              <span className="text-text-primary">Unknown Repository</span>, and confirm labels
              under <span className="text-text-primary">Verification</span> once you have enough
              examples to teach NOVA-X a new class.
            </p>
          </div>
        </>
      ) : null}
    </div>
  );
}
