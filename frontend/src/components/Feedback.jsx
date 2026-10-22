import { AlertTriangle } from "lucide-react";

export function ErrorNotice({ message }) {
  if (!message) return null;
  return (
    <div className="flex items-start gap-2.5 bg-danger-dim border border-danger/30 text-danger rounded p-3 text-sm mb-4">
      <AlertTriangle size={16} className="mt-0.5 shrink-0" />
      <span>{message}</span>
    </div>
  );
}

export function EmptyState({ title, description }) {
  return (
    <div className="border border-dashed border-line rounded p-10 text-center">
      <p className="font-display text-sm text-text-primary mb-1">{title}</p>
      {description && <p className="text-[13px] text-text-faint">{description}</p>}
    </div>
  );
}
