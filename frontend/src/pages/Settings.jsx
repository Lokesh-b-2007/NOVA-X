import PageHeader from "../components/PageHeader";

const SETTING_GROUPS = [
  {
    title: "Known-Class Detector",
    rows: [
      ["Weights", "NOVA_DETECTOR_WEIGHTS", "yolov8n.pt"],
      ["Device", "NOVA_DETECTOR_DEVICE", "auto (CUDA if available, else CPU)"],
      ["Confidence threshold", "NOVA_DETECTOR_CONF", "0.35"],
      ["Known classes", "NOVA_KNOWN_CLASSES", "person, car, bicycle, dog"],
    ],
  },
  {
    title: "OOD / Novelty Detection",
    rows: [
      ["Unknown threshold", "NOVA_NOVELTY_UNKNOWN_THRESHOLD", "0.70"],
      ["Suspicious threshold", "NOVA_NOVELTY_SUSPICIOUS_THRESHOLD", "0.40"],
      ["Scoring method", "NOVA_OOD_METHOD", "hybrid"],
    ],
  },
  {
    title: "Open-Vocabulary Detection (OWLv2)",
    rows: [
      ["Model", "NOVA_OVD_MODEL_ID", "google/owlv2-base-patch16-ensemble"],
      ["Score threshold", "NOVA_OVD_SCORE_THRESHOLD", "0.10"],
    ],
  },
  {
    title: "Clustering",
    rows: [
      ["Method", "NOVA_CLUSTERING_METHOD", "dbscan"],
      ["DBSCAN eps", "NOVA_DBSCAN_EPS", "0.35"],
      ["DBSCAN min samples", "NOVA_DBSCAN_MIN_SAMPLES", "3"],
    ],
  },
  {
    title: "Continual Learning",
    rows: [["Minimum dataset size", "NOVA_MIN_DATASET_SIZE", "30"]],
  },
];

export default function Settings() {
  return (
    <div>
      <PageHeader
        title="Settings"
        subtitle="NOVA-X is configured entirely through environment variables — nothing here is stored in the browser or database."
      />

      <div className="bg-base-800 border border-line rounded p-4 mb-6 text-[13px] text-text-muted leading-relaxed">
        To change any of these values, edit{" "}
        <code className="bg-base-900 px-1.5 py-0.5 rounded text-known text-[12px]">backend/.env</code>{" "}
        (copy it from <code className="bg-base-900 px-1.5 py-0.5 rounded text-[12px]">.env.example</code>{" "}
        first) and restart the backend.
      </div>

      <div className="space-y-5">
        {SETTING_GROUPS.map((group) => (
          <div key={group.title} className="bg-base-800 border border-line rounded overflow-hidden">
            <div className="px-4 py-2.5 border-b border-line">
              <h2 className="font-display text-sm font-medium">{group.title}</h2>
            </div>
            <div className="divide-y divide-line">
              {group.rows.map(([label, envVar, defaultVal]) => (
                <div key={envVar} className="flex items-center justify-between px-4 py-3">
                  <div>
                    <p className="text-sm text-text-primary">{label}</p>
                    <p className="text-[11px] text-text-faint font-mono mt-0.5">{envVar}</p>
                  </div>
                  <p className="text-[12px] text-text-muted text-right">{defaultVal}</p>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
