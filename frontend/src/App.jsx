import { lazy, Suspense } from "react";
import { Routes, Route } from "react-router-dom";
import Sidebar from "./components/Sidebar";

const Dashboard = lazy(() => import("./pages/Dashboard"));
const Detect = lazy(() => import("./pages/Detect"));
const Live = lazy(() => import("./pages/Live"));
const UnknownRepository = lazy(() => import("./pages/UnknownRepository"));
const Clusters = lazy(() => import("./pages/Clusters"));
const Verification = lazy(() => import("./pages/Verification"));
const KnownClasses = lazy(() => import("./pages/KnownClasses"));
const Learning = lazy(() => import("./pages/Learning"));
const Analytics = lazy(() => import("./pages/Analytics"));
const Evaluation = lazy(() => import("./pages/Evaluation"));
const Settings = lazy(() => import("./pages/Settings"));

export default function App() {
  return (
    <div className="flex h-screen w-full overflow-hidden">
      <Sidebar />
      <main className="flex-1 overflow-y-auto">
        <div className="max-w-6xl mx-auto px-8 py-8">
          <Suspense fallback={<p className="text-sm text-text-muted">Loading page…</p>}>
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/detect" element={<Detect />} />
              <Route path="/live" element={<Live />} />
              <Route path="/unknown-repository" element={<UnknownRepository />} />
              <Route path="/clusters" element={<Clusters />} />
              <Route path="/verification" element={<Verification />} />
              <Route path="/known-classes" element={<KnownClasses />} />
              <Route path="/learning" element={<Learning />} />
              <Route path="/analytics" element={<Analytics />} />
              <Route path="/evaluation" element={<Evaluation />} />
              <Route path="/settings" element={<Settings />} />
            </Routes>
          </Suspense>
        </div>
      </main>
    </div>
  );
}
