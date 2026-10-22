import { useEffect, useState } from "react";
import { Check, X, Pencil, Image as ImageIcon } from "lucide-react";
import {
  listUnknownObjects,
  getUnknownObjectImageUrl,
  suggestUnknownObject,
  confirmUnknownObject,
  rejectUnknownObject,
} from "../services/api";
import PageHeader from "../components/PageHeader";
import { ErrorNotice, EmptyState } from "../components/Feedback";

export default function Verification() {
  const [objects, setObjects] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [notice, setNotice] = useState(null);
  const [editingId, setEditingId] = useState(null);
  const [editValue, setEditValue] = useState("");
  const [imageErrors, setImageErrors] = useState({});
  const [suggestingIds, setSuggestingIds] = useState({});

  const formatConfidence = (confidence) =>
    confidence < 0.01 ? "<1%" : `${Math.round(confidence * 100)}%`;

  const load = () => {
    setLoading(true);
    listUnknownObjects("PENDING")
      .then((res) => setObjects(res.data))
      .catch((err) => setError(err.response?.data?.detail || "Could not load pending objects."))
      .finally(() => setLoading(false));
  };

  useEffect(load, []);

  const removeFromList = (id) => setObjects((prev) => prev.filter((o) => o.id !== id));

  const handleConfirm = async (obj, label) => {
    try {
      await confirmUnknownObject(obj.id, label);
      setNotice(`Confirmed as "${label}" and saved this crop as a verified class example.`);
      removeFromList(obj.id);
    } catch (err) {
      setError(err.response?.data?.detail || "Could not confirm label.");
    }
  };

  const handleReject = async (obj) => {
    try {
      await rejectUnknownObject(obj.id);
      removeFromList(obj.id);
    } catch (err) {
      setError(err.response?.data?.detail || "Could not reject object.");
    }
  };

  const handleSuggest = async (obj) => {
    setError(null);
    setSuggestingIds((previous) => ({ ...previous, [obj.id]: true }));
    try {
      const res = await suggestUnknownObject(obj.id);
      setObjects((previous) =>
        previous.map((item) => (item.id === obj.id ? res.data : item))
      );
      if (!res.data.ovd_label) {
        setNotice("The model could not find a confident match. You can still enter the label yourself.");
      }
    } catch (err) {
      setError(err.response?.data?.detail || "Could not generate an open-vocabulary suggestion.");
    } finally {
      setSuggestingIds((previous) => ({ ...previous, [obj.id]: false }));
    }
  };

  return (
    <div>
      <PageHeader
        title="Human Verification"
        subtitle="Review each detected object's photo, then confirm, correct, or keep it unknown."
      />

      <ErrorNotice message={error} />
      {notice && <p className="text-sm text-known mb-4">{notice}</p>}

      {loading ? (
        <p className="text-sm text-text-muted">Loading…</p>
      ) : objects.length === 0 ? (
        <EmptyState
          title="Nothing pending review"
          description="All caught up — unknown objects awaiting verification will appear here."
        />
      ) : (
        <div className="space-y-3">
          {objects.map((obj) => (
            <div
              key={obj.id}
              className="bg-base-800 border border-line rounded p-4 grid grid-cols-1 md:grid-cols-[minmax(200px,280px)_minmax(0,1fr)] gap-4"
            >
              <div>
                <div className="min-h-48 md:min-h-52 bg-base-950 border border-line rounded overflow-hidden flex items-center justify-center">
                  {imageErrors[obj.id] ? (
                    <div className="flex flex-col items-center gap-2 p-4 text-text-faint">
                      <ImageIcon size={22} />
                      <span className="text-xs">Object photo unavailable</span>
                    </div>
                  ) : (
                    <img
                      src={getUnknownObjectImageUrl(obj.id)}
                      alt={`Detected unknown object outlined in red${obj.ovd_label ? `: ${obj.ovd_label}` : ""}`}
                      className="w-full h-48 md:h-52 object-contain"
                      onError={() =>
                        setImageErrors((previous) => ({ ...previous, [obj.id]: true }))
                      }
                    />
                  )}
                </div>
                {!imageErrors[obj.id] && (
                  <p className="text-[11px] text-danger mt-1.5">Detected object outlined in red</p>
                )}
              </div>

              <div className="min-w-0 flex flex-col justify-between gap-4">
                <div>
                  <p className="text-[11px] text-text-faint mb-1">
                    Novelty {Math.round(obj.novelty_score * 100)}%
                  </p>
                  <p className="font-display text-sm font-medium">
                    AI suggestion:{" "}
                    <span className="text-novel">{obj.ovd_label || "no suggestion"}</span>
                    {obj.ovd_confidence != null && (
                      <span className="text-text-muted font-body">
                        {" "}({formatConfidence(obj.ovd_confidence)})
                      </span>
                    )}
                  </p>
                  {obj.ovd_top_predictions?.length > 1 && (
                    <p className="text-xs text-text-muted mt-2">
                      Other guesses:{" "}
                      {obj.ovd_top_predictions
                        .slice(1, 4)
                        .map((prediction) => `${prediction.label} (${formatConfidence(prediction.confidence)})`)
                        .join(", ")}
                    </p>
                  )}
                  {!obj.ovd_label && (
                    <p className="text-xs text-text-muted mt-2">
                      No suggestion was saved for this object yet.
                    </p>
                  )}
                  {editingId === obj.id && (
                    <div className="flex flex-wrap items-center gap-2 mt-3">
                      <input
                        autoFocus
                        value={editValue}
                        onChange={(e) => setEditValue(e.target.value)}
                        placeholder="Corrected label"
                        className="bg-base-900 border border-line rounded px-2.5 py-1.5 text-sm text-text-primary focus-ring flex-1 max-w-[220px]"
                      />
                      <button
                        onClick={() => editValue.trim() && handleConfirm(obj, editValue.trim())}
                        className="text-known text-[12px] font-medium hover:underline"
                      >
                        Save
                      </button>
                      <button
                        onClick={() => setEditingId(null)}
                        className="text-text-faint text-[12px] hover:underline"
                      >
                        Cancel
                      </button>
                    </div>
                  )}
                </div>

                {editingId !== obj.id && (
                  <div className="flex flex-wrap items-center gap-2">
                    {!obj.ovd_label && (
                      <button
                        onClick={() => handleSuggest(obj)}
                        disabled={suggestingIds[obj.id]}
                        className="bg-novel/10 text-novel text-[12px] font-medium px-3 py-1.5 rounded hover:bg-novel/20 transition-colors disabled:opacity-50 focus-ring"
                      >
                        {suggestingIds[obj.id] ? "Finding suggestion…" : "Generate suggestion"}
                      </button>
                    )}
                    <button
                      onClick={() => obj.ovd_label && handleConfirm(obj, obj.ovd_label)}
                      disabled={!obj.ovd_label}
                      className="flex items-center gap-1.5 bg-known/10 text-known text-[12px] font-medium px-3 py-1.5 rounded hover:bg-known/20 transition-colors disabled:opacity-30 focus-ring"
                    >
                      <Check size={13} /> Confirm
                    </button>
                    <button
                      onClick={() => {
                        setEditingId(obj.id);
                        setEditValue(obj.ovd_label || "");
                      }}
                      className="flex items-center gap-1.5 bg-base-700 text-text-muted text-[12px] font-medium px-3 py-1.5 rounded hover:text-text-primary transition-colors focus-ring"
                    >
                      <Pencil size={13} /> Edit
                    </button>
                    <button
                      onClick={() => handleReject(obj)}
                      className="flex items-center gap-1.5 bg-danger/10 text-danger text-[12px] font-medium px-3 py-1.5 rounded hover:bg-danger/20 transition-colors focus-ring"
                    >
                      <X size={13} /> Keep unknown
                    </button>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
