import StatusBadge from "./StatusBadge";

function NoveltyBar({ score }) {
  const pct = Math.round(score * 100);
  const color = score >= 0.7 ? "bg-novel" : score >= 0.4 ? "bg-suspicious" : "bg-known";
  return (
    <div>
      <div className="flex justify-between text-[11px] text-text-muted mb-1">
        <span>Novelty score</span>
        <span className="tabular-nums text-text-primary">{pct}%</span>
      </div>
      <div className="h-1.5 rounded-full bg-base-700 overflow-hidden">
        <div className={`h-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

export default function DetectionCard({ detection }) {
  const { class_name, confidence, novelty, ovd, ovd_top_predictions, model_version } = detection;
  const isKnown = novelty.status === "KNOWN";

  return (
    <div className="bg-base-800 border border-line rounded p-4 space-y-3">
      <div className="flex items-center justify-between">
        <span className="font-display font-medium text-sm">
          {isKnown ? class_name : "Unidentified object"}
        </span>
        <StatusBadge status={novelty.status} />
      </div>

      {isKnown ? (
        <p className="text-sm text-text-muted">
          Detector confidence:{" "}
          <span className="text-text-primary tabular-nums">
            {Math.round((confidence ?? 0) * 100)}%
          </span>
        </p>
      ) : (
        <>
          <NoveltyBar score={novelty.novelty_score} />
          {ovd?.label ? (
            <div className="pt-1 border-t border-line">
              <p className="text-[11px] text-text-muted mb-1.5">Open-vocabulary prediction</p>
              <div className="space-y-1">
                {(ovd_top_predictions?.length ? ovd_top_predictions : [ovd]).map((p, i) => (
                  <div key={i} className="flex justify-between text-sm">
                    <span className={i === 0 ? "text-text-primary" : "text-text-muted"}>
                      {p.label}
                    </span>
                    <span className="tabular-nums text-text-muted">
                      {Math.round(p.confidence * 100)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <p className="text-[12px] text-text-faint italic pt-1 border-t border-line">
              No open-vocabulary prediction available (OWLv2 not loaded).
            </p>
          )}
        </>
      )}

      <p className="text-[11px] text-text-faint pt-1">
        Model: {model_version || "—"} · Reason: {novelty.reason}
      </p>
    </div>
  );
}
