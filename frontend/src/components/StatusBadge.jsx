const STYLES = {
  KNOWN: "bg-known-dim text-known border-known/30",
  UNKNOWN: "bg-novel-dim text-novel border-novel/30",
  SUSPICIOUS: "bg-suspicious-dim text-suspicious border-suspicious/30",
  CONFIRMED: "bg-known-dim text-known border-known/30",
  REJECTED: "bg-danger-dim text-danger border-danger/30",
  PENDING: "bg-base-700 text-text-muted border-line",
};

export default function StatusBadge({ status, className = "" }) {
  const style = STYLES[status] || STYLES.PENDING;
  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-sm border text-[11px] font-medium tracking-wide ${style} ${className}`}
    >
      <span className="w-1.5 h-1.5 rounded-full bg-current" />
      {status}
    </span>
  );
}
