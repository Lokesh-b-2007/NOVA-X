import { NavLink } from "react-router-dom";
import {
  LayoutDashboard,
  ScanEye,
  Video,
  Boxes,
  Shapes,
  ShieldCheck,
  Tags,
  GitBranch,
  BarChart3,
  Gauge,
  Settings as SettingsIcon,
} from "lucide-react";

const NAV_ITEMS = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard, end: true },
  { to: "/detect", label: "Detect", icon: ScanEye },
  { to: "/live", label: "Live Vision", icon: Video },
  { to: "/unknown-repository", label: "Unknown Repository", icon: Boxes },
  { to: "/clusters", label: "Novel Clusters", icon: Shapes },
  { to: "/verification", label: "Verification", icon: ShieldCheck },
  { to: "/known-classes", label: "Known Classes", icon: Tags },
  { to: "/learning", label: "Learning", icon: GitBranch },
  { to: "/analytics", label: "Analytics", icon: BarChart3 },
  { to: "/evaluation", label: "Evaluation", icon: Gauge },
  { to: "/settings", label: "Settings", icon: SettingsIcon },
];

export default function Sidebar() {
  return (
    <aside className="w-60 shrink-0 h-full bg-base-900 border-r border-line flex flex-col">
      <div className="px-5 pt-6 pb-5 border-b border-line">
        <div className="flex items-baseline gap-1.5">
          <span className="font-display font-semibold text-lg tracking-tight text-text-primary">
            NOVA
          </span>
          <span className="font-display font-semibold text-lg tracking-tight text-known">
            -X
          </span>
        </div>
        <p className="text-[11px] text-text-faint mt-1 leading-snug">
          Novelty-Aware Open-World Visual Intelligence
        </p>
      </div>

      <nav className="flex-1 overflow-y-auto py-3 px-2.5">
        {NAV_ITEMS.map(({ to, label, icon: Icon, end }) => (
          <NavLink
            key={to}
            to={to}
            end={end}
            className={({ isActive }) =>
              `flex items-center gap-2.5 px-3 py-2 rounded text-sm mb-0.5 transition-colors focus-ring ${
                isActive
                  ? "bg-base-700 text-text-primary"
                  : "text-text-muted hover:text-text-primary hover:bg-base-800"
              }`
            }
          >
            {({ isActive }) => (
              <>
                <Icon
                  size={16}
                  strokeWidth={1.75}
                  className={isActive ? "text-known" : "text-text-faint"}
                />
                {label}
              </>
            )}
          </NavLink>
        ))}
      </nav>

      <div className="px-4 py-3 border-t border-line text-[11px] text-text-faint">
        Research build · local inference
      </div>
    </aside>
  );
}
