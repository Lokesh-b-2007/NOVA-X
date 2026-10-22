import { useCallback, useRef, useState } from "react";
import { UploadCloud, Loader2 } from "lucide-react";
import { detectImage } from "../services/api";
import PageHeader from "../components/PageHeader";
import DetectionCard from "../components/DetectionCard";
import { ErrorNotice, EmptyState } from "../components/Feedback";

export default function Detect() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const inputRef = useRef(null);

  const handleFile = useCallback((f) => {
    if (!f) return;
    setFile(f);
    setPreviewUrl(URL.createObjectURL(f));
    setResult(null);
    setError(null);
  }, []);

  const onDrop = useCallback(
    (e) => {
      e.preventDefault();
      const f = e.dataTransfer.files?.[0];
      if (f) handleFile(f);
    },
    [handleFile]
  );

  const runDetection = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const res = await detectImage(file);
      setResult(res.data);
    } catch (err) {
      setError(
        err.response?.data?.detail ||
          "Detection failed. Make sure the backend is running and the YOLO model could load " +
            "(check the backend console for a DetectorLoadError)."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <PageHeader
        title="Detect"
        subtitle="Upload an image to run the full known-object + OOD + open-vocabulary pipeline."
      />

      <ErrorNotice message={error} />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Upload / preview panel */}
        <div>
          <div
            onDrop={onDrop}
            onDragOver={(e) => e.preventDefault()}
            onClick={() => inputRef.current?.click()}
            className={`bracket-frame text-known cursor-pointer border border-dashed border-line rounded p-6 flex flex-col items-center justify-center text-center transition-colors hover:border-known/40 ${
              previewUrl ? "min-h-[200px]" : "min-h-[280px]"
            }`}
          >
            {previewUrl ? (
              <img
                src={previewUrl}
                alt="Selected upload"
                className="max-h-[420px] rounded object-contain"
              />
            ) : (
              <>
                <UploadCloud size={28} className="text-text-faint mb-3" strokeWidth={1.5} />
                <p className="text-sm text-text-primary font-medium">
                  Drop an image here, or click to browse
                </p>
                <p className="text-[12px] text-text-faint mt-1">JPG or PNG</p>
              </>
            )}
            <input
              ref={inputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => handleFile(e.target.files?.[0])}
            />
          </div>

          <button
            onClick={runDetection}
            disabled={!file || loading}
            className="mt-4 w-full flex items-center justify-center gap-2 bg-known text-base-950 font-medium text-sm rounded py-2.5 disabled:opacity-40 disabled:cursor-not-allowed hover:bg-known/90 transition-colors focus-ring"
          >
            {loading && <Loader2 size={16} className="animate-spin" />}
            {loading ? "Running pipeline…" : "Run detection"}
          </button>

          {result && (
            <div className="mt-4 grid grid-cols-3 gap-3 text-center">
              <div className="bg-base-800 border border-line rounded p-3">
                <p className="text-[11px] text-text-muted">Known</p>
                <p className="font-display text-lg text-known tabular-nums">{result.known_count}</p>
              </div>
              <div className="bg-base-800 border border-line rounded p-3">
                <p className="text-[11px] text-text-muted">Unknown</p>
                <p className="font-display text-lg text-novel tabular-nums">{result.unknown_count}</p>
              </div>
              <div className="bg-base-800 border border-line rounded p-3">
                <p className="text-[11px] text-text-muted">Time</p>
                <p className="font-display text-lg tabular-nums">
                  {Math.round(result.processing_time_ms)}ms
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Results panel */}
        <div>
          <h2 className="font-display text-sm font-medium mb-3 text-text-muted">
            Detections {result ? `(${result.detections.length})` : ""}
          </h2>
          {!result ? (
            <EmptyState
              title="No detections yet"
              description="Upload an image and run detection to see results here."
            />
          ) : result.detections.length === 0 ? (
            <EmptyState
              title="Nothing detected"
              description="The detector found no objects above the confidence threshold."
            />
          ) : (
            <div className="space-y-3 max-h-[560px] overflow-y-auto pr-1">
              {result.detections.map((d) => (
                <DetectionCard key={d.id} detection={d} />
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
