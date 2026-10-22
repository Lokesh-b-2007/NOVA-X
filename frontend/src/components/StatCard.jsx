export default function StatCard({ label, value, accent = "text-text-primary", hint }) {
  return (
    <div className="bg-base-800 border border-line rounded p-4">
      <p className="text-[12px] text-text-muted mb-2">{label}</p>
      <p className={`font-display text-3xl font-semibold tabular-nums ${accent}`}>{value}</p>
      {hint && <p className="text-[11px] text-text-faint mt-1.5">{hint}</p>}
    </div>
  );
}
