import { useEffect, useState } from "react";
import { GraduationCap, Loader2 } from "lucide-react";
import {
  getPendingDatasetSize,
  trainNewClass,
  listTrainingJobs,
  listModelVersions,
} from "../services/api";
import PageHeader from "../components/PageHeader";
import { ErrorNotice, EmptyState } from "../components/Feedback";

export default function Learning() {
  const [targetClass, setTargetClass] = useState("");
  const [pending, setPending] = useState(null);
  const [jobs, setJobs] = useState([]);
  const [versions, setVersions] = useState([]);
  const [training, setTraining] = useState(false);
  const [error, setError] = useState(null);
  const [checking, setChecking] = useState(false);

  const loadJobsAndVersions = () => {
    listTrainingJobs().then((res) => setJobs(res.data)).catch(() => {});
    listModelVersions().then((res) => setVersions(res.data)).catch(() => {});
  };

  useEffect(loadJobsAndVersions, []);

  const checkReadiness = async () => {
    if (!targetClass.trim()) return;
    setChecking(true);
    setError(null);
    try {
      const res = await getPendingDatasetSize(targetClass.trim());
      setPending(res.data);
    } catch (err) {
      setError(err.response?.data?.detail || "Could not check dataset size.");
    } finally {
      setChecking(false);
    }
  };

  const startTraining = async () => {
    setTraining(true);
    setError(null);
    try {
      await trainNewClass(targetClass.trim());
      setPending(null);
      setTargetClass("");
      loadJobsAndVersions();
    } catch (err) {
      setError(err.response?.data?.detail || "Training could not start.");
    } finally {
      setTraining(false);
    }
  };

  return (
    <div>
      <PageHeader
        title="Continual Learning"
        subtitle="Confirmed unknown objects accumulate here as training data. Retraining is always a manual, explicit action."
      />

      <ErrorNotice message={error} />

      <div className="bg-base-800 border border-line rounded p-5 mb-6">
        <h2 className="font-display text-sm font-medium mb-3">Create a new class</h2>
        <div className="flex gap-2 max-w-md">
          <input
            value={targetClass}
            onChange={(e) => setTargetClass(e.target.value)}
            placeholder="e.g. chair"
            className="flex-1 bg-base-900 border border-line rounded px-3 py-2 text-sm focus-ring"
          />
          <button
            onClick={checkReadiness}
            disabled={checking || !targetClass.trim()}
            className="bg-base-700 text-text-primary text-sm px-3 py-2 rounded hover:bg-base-700/80 disabled:opacity-40 focus-ring"
          >
            Check
          </button>
        </div>

        {pending && (
          <div className="mt-4 flex items-center justify-between bg-base-900 border border-line rounded p-3">
            <div>
              <p className="text-sm">
                <span className="tabular-nums text-text-primary">{pending.pending_examples}</span>{" "}
                <span className="text-text-muted">
                  / {pending.minimum_required} verified examples ready
                </span>
              </p>
              <p className="text-[11px] text-text-faint mt-0.5">
                {pending.ready
                  ? "Minimum dataset size met — safe to start training."
                  : "Not enough verified examples yet. Confirm more objects under Verification."}
              </p>
            </div>
            <button
              onClick={startTraining}
              disabled={!pending.ready || training}
              className="flex items-center gap-2 bg-known text-base-950 text-sm font-medium px-3 py-2 rounded disabled:opacity-30 hover:bg-known/90 focus-ring"
            >
              {training ? <Loader2 size={14} className="animate-spin" /> : <GraduationCap size={14} />}
              Start training
            </button>
          </div>
        )}
      </div>

      <h2 className="font-display text-sm font-medium mb-3 text-text-muted">Training jobs</h2>
      {jobs.length === 0 ? (
        <EmptyState title="No training jobs yet" />
      ) : (
        <div className="bg-base-800 border border-line rounded divide-y divide-line mb-6">
          {jobs.map((j) => (
            <div key={j.id} className="flex items-center justify-between px-4 py-3">
              <div>
                <p className="text-sm font-medium capitalize">{j.target_class}</p>
                <p className="text-[11px] text-text-faint mt-0.5">
                  {j.dataset_size} images · {j.status}
                  {j.error ? ` · ${j.error}` : ""}
                </p>
              </div>
              <p className="text-[12px] text-text-muted">{j.resulting_model_version || "—"}</p>
            </div>
          ))}
        </div>
      )}

      <h2 className="font-display text-sm font-medium mb-3 text-text-muted">Model versions</h2>
      {versions.length === 0 ? (
        <EmptyState title="Only the base pretrained model exists so far" />
      ) : (
        <div className="bg-base-800 border border-line rounded divide-y divide-line">
          {versions.map((v) => (
            <div key={v.id} className="flex items-center justify-between px-4 py-3">
              <div>
                <p className="text-sm font-medium">
                  {v.version_id} {v.is_active && <span className="text-known text-[11px]">· active</span>}
                </p>
                <p className="text-[11px] text-text-faint mt-0.5">{v.classes.join(", ")}</p>
              </div>
              <p className="text-[12px] text-text-muted tabular-nums">{v.dataset_size} images</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
