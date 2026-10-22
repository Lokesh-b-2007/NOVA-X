import { useEffect, useState } from "react";
import { listKnownClasses } from "../services/api";
import PageHeader from "../components/PageHeader";
import { ErrorNotice, EmptyState } from "../components/Feedback";

export default function KnownClasses() {
  const [classes, setClasses] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    listKnownClasses()
      .then((res) => setClasses(res.data))
      .catch((err) => setError(err.response?.data?.detail || "Could not load known classes."))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div>
      <PageHeader
        title="Known Classes"
        subtitle="Classes the active detector currently recognizes with confidence."
      />

      <ErrorNotice message={error} />

      {loading ? (
        <p className="text-sm text-text-muted">Loading…</p>
      ) : classes.length === 0 ? (
        <EmptyState title="No known classes configured" />
      ) : (
        <div className="bg-base-800 border border-line rounded divide-y divide-line">
          {classes.map((c) => (
            <div key={c.id} className="flex items-center justify-between px-4 py-3">
              <div>
                <p className="font-display text-sm font-medium capitalize">{c.name}</p>
                <p className="text-[11px] text-text-faint mt-0.5">
                  {c.source === "base" ? "Base pretrained class" : "Added via continual learning"}
                </p>
              </div>
              <p className="text-[12px] text-text-muted tabular-nums">
                {c.example_count} example{c.example_count === 1 ? "" : "s"}
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
