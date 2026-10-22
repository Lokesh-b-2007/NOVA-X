import axios from "axios";

// In dev, Vite proxies /api -> http://localhost:8000 (see vite.config.js),
// so no base URL is needed. In production, set VITE_API_BASE_URL.
const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || "",
  timeout: 30000,
});

// --- Detection ---
export const detectImage = (file) => {
  const form = new FormData();
  form.append("file", file);
  return api.post("/api/detect/image", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
};

export const detectFrame = (blob, streamId) => {
  const form = new FormData();
  form.append("file", blob, "frame.jpg");
  if (streamId) form.append("stream_id", streamId);
  return api.post("/api/detect/frame", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
};

// --- Dashboard / analytics / evaluation ---
export const getDashboard = () => api.get("/api/dashboard");
export const getAnalytics = () => api.get("/api/analytics");
export const getEvaluation = () => api.get("/api/evaluation");

// --- Unknown object repository ---
export const listUnknownObjects = (status) =>
  api.get("/api/unknown-objects", { params: status ? { status } : {} });
export const getUnknownObject = (id) => api.get(`/api/unknown-objects/${id}`);
export const suggestUnknownObject = (id) => api.post(`/api/unknown-objects/${id}/suggest`);
export const getUnknownObjectImageUrl = (id) => {
  const path = `/api/unknown-objects/${encodeURIComponent(id)}/image`;
  return api.defaults.baseURL ? new URL(path, api.defaults.baseURL).toString() : path;
};
export const deleteUnknownObject = (id) => api.delete(`/api/unknown-objects/${id}`);
export const clearUnknownObjects = () => api.delete("/api/unknown-objects");
export const confirmUnknownObject = (id, label) =>
  api.post(`/api/unknown-objects/${id}/confirm`, { label });
export const rejectUnknownObject = (id) => api.post(`/api/unknown-objects/${id}/reject`);

// --- Clusters ---
export const listClusters = () => api.get("/api/clusters");
export const recomputeClusters = () => api.post("/api/clusters/recompute");

// --- Known classes / models / continual learning ---
export const listKnownClasses = () => api.get("/api/known-classes");
export const listModelVersions = () => api.get("/api/models");
export const trainNewClass = (targetClass) => api.post("/api/models/train", { target_class: targetClass });
export const listTrainingJobs = () => api.get("/api/training-jobs");
export const getPendingDatasetSize = (targetClass) =>
  api.get("/api/training-jobs/pending-size", { params: { target_class: targetClass } });

export default api;
