import { useEffect, useState } from "react";
import { RefreshCw } from "lucide-react";
import { getEvaluation } from "../services/api";
import PageHeader from "../components/PageHeader";
import { ErrorNotice } from "../components/Feedback";

function MetricGroup({ title, metrics, description, sampleCount, sampleLabel, caveat }) {
  return (
    <div className="bg-base-800 border border-line rounded p-4">
      <h3 className="font-display text-sm font-medium mb-1">{title}</h3>
      <p className="text-[11px] text-text-faint mb-4">{description}</p>
      {sampleCount != null && (
        <p className="text-[11px] text-text-muted mb-3">
          Evaluated on {sampleCount} {sampleLabel || "samples"}
        </p>
      )}
      {metrics ? (
        <div className="grid grid-cols-2 gap-3">
          {Object.entries(metrics).map(([key, value]) => (
            <div key={key}>
              <p className="text-[11px] text-text-muted capitalize">{key.replace(/_/g, " ")}</p>
              <p className="font-display text-xl tabular-nums text-text-primary">
                {typeof value === "number" ? value.toFixed(3) : value}
              </p>
            </div>
          ))}
        </div>
      ) : (
        <p className="text-[12px] text-text-faint italic">
          Not computed yet — see the evaluation setup instructions above.
        </p>
      )}
      {metrics && caveat && <p className="text-[11px] text-text-faint mt-3">{caveat}</p>}
    </div>
  );
}

export default function Evaluation() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  const loadEvaluation = () => {
    setLoading(true);
    setError(null);
    getEvaluation()
      .then((res) => setData(res.data))
      .catch((err) => setError(err.response?.data?.detail || "Could not load evaluation."))
      .finally(() => setLoading(false));
  };

  useEffect(loadEvaluation, []);

  return (
    <div>
      <PageHeader
        title="Model Evaluation"
        subtitle="Real metrics computed from labeled evaluation data — never estimated or fabricated."
      />

      <ErrorNotice message={error} />
      <button
        onClick={loadEvaluation}
        disabled={loading}
        className="flex items-center gap-2 bg-base-800 border border-line text-text-muted text-xs px-3 py-2 rounded hover:text-text-primary disabled:opacity-50 focus-ring mb-4"
      >
        <RefreshCw size={13} className={loading ? "animate-spin" : ""} />
        {loading ? "Refreshing…" : "Refresh metrics"}
      </button>

      {loading ? (
        <p className="text-sm text-text-muted">Loading…</p>
      ) : data ? (
        <>
          <div className="bg-known-dim border border-known/30 text-known rounded p-3 text-[12px] mb-5">
            {data.notes}
          </div>
          {data.notes.startsWith("Some metrics are unavailable") && (
            <div className="bg-base-800 border border-line rounded p-4 mb-5 text-[12px] text-text-muted space-y-2">
              <p className="font-medium text-text-primary">Enable evaluation metrics</p>
              <p>
                Known detection: from the <code>backend</code> folder run{" "}
                <code>python scripts/run_known_eval.py</code>, then refresh. This uses the
                provided COCO128 sample; it is a sanity check, not a held-out benchmark.
              </p>
              <p>
                OOD and unknown-object metrics: from <code>backend</code>, run{" "}
                <code>python scripts/run_ood_eval.py</code> against the supplied, annotated
                COCO128 dataset. For custom test photos instead, use the labeled folders and{" "}
                <code>python scripts/build_eval_labels.py</code>.
              </p>
              <p>Open-vocabulary metrics are calculated from objects confirmed in Human Verification.</p>
            </div>
          )}
          {data.evaluation_caveats?.map((caveat) => (
            <p key={caveat} className="text-[11px] text-text-faint mb-3">{caveat}</p>
          ))}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <MetricGroup
              title="Known Detection"
              description="Precision, recall and mAP over the known-class validation set."
              metrics={data.known_metrics}
            />
            <MetricGroup
              title="OOD / Novelty Detection"
              description="AUROC, AUPR and FPR@95%TPR on detected boxes matched to COCO128 ground truth (IoU ≥ 0.5)."
              metrics={data.ood_metrics}
              sampleCount={data.ood_sample_count}
              sampleLabel="ground-truth-matched detections"
              caveat={data.ood_metrics ? "Unmatched detector predictions count as negatives; missed objects are excluded from ranking." : null}
            />
            <MetricGroup
              title="Unknown Object Detection"
              description="Recall and precision for COCO128 objects outside NOVA-X's configured known-class list; missed objects count as false negatives."
              metrics={data.unknown_metrics}
              sampleCount={data.ood_ground_truth_unknown_count}
              sampleLabel="ground-truth unknown objects"
            />
            <MetricGroup
              title="Open-Vocabulary Recognition"
              description="Top-1 / top-5 OWLv2 accuracy against human-confirmed or folder-labeled ground truth."
              metrics={data.ovd_metrics}
              sampleCount={data.ovd_sample_count}
              sampleLabel={data.ovd_sample_source || "crops"}
              caveat={data.ovd_metrics ? "This repository sample is not a held-out test set and may be biased by the original suggestions shown to reviewers." : null}
            />
          </div>
        </>
      ) : null}
    </div>
  );
}
