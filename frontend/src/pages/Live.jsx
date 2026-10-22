import { useEffect, useRef, useState, useCallback } from "react";
import { Video, VideoOff } from "lucide-react";
import { detectFrame } from "../services/api";
import PageHeader from "../components/PageHeader";
import DetectionCard from "../components/DetectionCard";
import { ErrorNotice, EmptyState } from "../components/Feedback";

const FRAME_INTERVAL_MS = 700; // ~1.4 FPS request rate; tune per hardware (target 5-15 FPS end-to-end)
const CAPTURE_WIDTH = 640;

export default function Live() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const streamIdRef = useRef(null);
  const intervalRef = useRef(null);
  const busyRef = useRef(false);

  const [active, setActive] = useState(false);
  const [detections, setDetections] = useState([]);
  const [error, setError] = useState(null);
  const [fps, setFps] = useState(0);
  const [starting, setStarting] = useState(false);
  const [processing, setProcessing] = useState(false);

  const stop = useCallback(() => {
    clearInterval(intervalRef.current);
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    streamIdRef.current = null;
    setActive(false);
  }, []);

  const start = async () => {
    setError(null);
    setStarting(true);
    let stream;
    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error("Camera access is unavailable. Open NOVA-X on localhost or HTTPS.");
      }
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: CAPTURE_WIDTH } },
        audio: false,
      });
      streamRef.current = stream;
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      streamIdRef.current =
        globalThis.crypto?.randomUUID?.() ||
        `live-${Date.now()}-${Math.random().toString(36).slice(2)}`;
      setDetections([]);
      setActive(true);
    } catch (err) {
      stream?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
      if (videoRef.current) videoRef.current.srcObject = null;
      setError(err.message || "Could not access the webcam. Check camera permission and close other apps using it.");
    } finally {
      setStarting(false);
    }
  };

  useEffect(() => {
    if (!active) return;
    let lastTick = performance.now();

    intervalRef.current = setInterval(async () => {
      if (busyRef.current || !videoRef.current || !canvasRef.current) return;
      busyRef.current = true;

      const video = videoRef.current;
      const canvas = canvasRef.current;
      const streamId = streamIdRef.current;
      if (!streamId) {
        busyRef.current = false;
        return;
      }
      if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA || !video.videoWidth || !video.videoHeight) {
        busyRef.current = false;
        return;
      }
      const scale = Math.min(CAPTURE_WIDTH / video.videoWidth, 1);
      canvas.width = Math.round(video.videoWidth * scale);
      canvas.height = Math.round(video.videoHeight * scale);
      const ctx = canvas.getContext("2d");
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      canvas.toBlob(
        async (blob) => {
          if (!blob) {
            busyRef.current = false;
            return;
          }
          setProcessing(true);
          try {
            const res = await detectFrame(blob, streamId);
            if (streamIdRef.current !== streamId) return;
            setDetections(res.data.detections);
            const now = performance.now();
            setFps(1000 / (now - lastTick));
            lastTick = now;
          } catch (err) {
            if (streamIdRef.current !== streamId) return;
            setError(
              err.response?.data?.detail || "Frame detection failed — stopping stream."
            );
            stop();
          } finally {
            setProcessing(false);
            busyRef.current = false;
          }
        },
        "image/jpeg",
        0.7
      );
    }, FRAME_INTERVAL_MS);

    return () => clearInterval(intervalRef.current);
  }, [active, stop]);

  useEffect(() => () => stop(), [stop]);

  return (
    <div>
      <PageHeader
        title="Live Vision"
        subtitle="Objects are locked to one detection per webcam session. Newly entering known or unknown objects are detected and added."
      />

      <ErrorNotice message={error} />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div>
          <div className="bracket-frame text-known bg-base-800 border border-line rounded overflow-hidden aspect-video flex items-center justify-center">
            <video
              ref={videoRef}
              muted
              playsInline
              className={active ? "w-full h-full object-cover" : "hidden"}
            />
            {!active && <p className="text-[13px] text-text-faint">Webcam is off</p>}
          </div>
          <canvas ref={canvasRef} className="hidden" />

          <div className="flex items-center justify-between mt-4">
            <button
              onClick={active ? stop : start}
              disabled={starting}
              className={`flex items-center gap-2 text-sm font-medium px-4 py-2 rounded transition-colors focus-ring ${
                active
                  ? "bg-danger/10 text-danger hover:bg-danger/20"
                  : "bg-known text-base-950 hover:bg-known/90"
              }`}
            >
              {active ? <VideoOff size={15} /> : <Video size={15} />}
              {active ? "Stop stream" : starting ? "Starting webcam…" : "Start webcam"}
            </button>
            {active && (
              <span className="text-[12px] text-text-faint tabular-nums">
                {fps > 0 ? `~${fps.toFixed(1)} FPS` : processing ? "Analyzing frame…" : "Waiting for frame…"}
              </span>
            )}
          </div>
        </div>

        <div>
          <h2 className="font-display text-sm font-medium mb-3 text-text-muted">
            Locked objects {detections.length ? `(${detections.length})` : ""}
          </h2>
          {detections.length === 0 ? (
            <EmptyState
              title={processing ? "Analyzing current frame…" : "No detections yet"}
              description={
                processing
                  ? "The CPU detector can take a few seconds to process each frame."
                  : "Start the webcam to begin streaming frames through the pipeline."
              }
            />
          ) : (
            <div className="space-y-3 max-h-[420px] overflow-y-auto pr-1">
              {detections.map((d) => (
                <DetectionCard key={d.id} detection={d} />
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
