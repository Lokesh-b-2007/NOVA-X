import { useEffect, useState } from "react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  PieChart,
  Pie,
  Cell,
} from "recharts";
import { getAnalytics } from "../services/api";
import PageHeader from "../components/PageHeader";
import { ErrorNotice, EmptyState } from "../components/Feedback";

const COLORS = ["#33D6C0", "#F5A623", "#E8825C", "#5B6577", "#8B96A6"];

const CHART_TOOLTIP_STYLE = {
  background: "#161C26",
  border: "1px solid #232C39",
  borderRadius: 6,
  fontSize: 12,
  color: "#E7ECF2",
};

function ChartPanel({ title, children, empty }) {
  return (
    <div className="bg-base-800 border border-line rounded p-4">
      <h3 className="font-display text-sm font-medium text-text-muted mb-4">{title}</h3>
      {empty ? (
        <div className="h-[220px] flex items-center justify-center text-[12px] text-text-faint">
          No data yet
        </div>
      ) : (
        <div className="h-[220px]">{children}</div>
      )}
    </div>
  );
}

export default function Analytics() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getAnalytics()
      .then((res) => setData(res.data))
      .catch((err) => setError(err.response?.data?.detail || "Could not load analytics."))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <p className="text-sm text-text-muted">Loading…</p>;

  const knownVsUnknown = data
    ? [
        { name: "Known", value: data.known_vs_unknown.known },
        { name: "Unknown", value: data.known_vs_unknown.unknown },
        { name: "Suspicious", value: data.known_vs_unknown.suspicious },
      ].filter((d) => d.value > 0)
    : [];

  return (
    <div>
      <PageHeader
        title="Analytics"
        subtitle="Aggregate charts computed live from every stored detection and novel object."
      />

      <ErrorNotice message={error} />

      {!data ? (
        <EmptyState title="No analytics available" />
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <ChartPanel title="Known vs Unknown Objects" empty={knownVsUnknown.length === 0}>
            <ResponsiveContainer>
              <PieChart>
                <Pie data={knownVsUnknown} dataKey="value" nameKey="name" innerRadius={50} outerRadius={80}>
                  {knownVsUnknown.map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip contentStyle={CHART_TOOLTIP_STYLE} />
              </PieChart>
            </ResponsiveContainer>
          </ChartPanel>

          <ChartPanel
            title="Unknown Objects Over Time"
            empty={data.unknown_over_time.length === 0}
          >
            <ResponsiveContainer>
              <LineChart data={data.unknown_over_time}>
                <CartesianGrid stroke="#232C39" strokeDasharray="3 3" />
                <XAxis dataKey="date" stroke="#5B6577" fontSize={11} />
                <YAxis stroke="#5B6577" fontSize={11} allowDecimals={false} />
                <Tooltip contentStyle={CHART_TOOLTIP_STYLE} />
                <Line type="monotone" dataKey="count" stroke="#F5A623" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </ChartPanel>

          <ChartPanel
            title="Novelty Score Distribution"
            empty={data.novelty_score_distribution.every((b) => b.count === 0)}
          >
            <ResponsiveContainer>
              <BarChart data={data.novelty_score_distribution}>
                <CartesianGrid stroke="#232C39" strokeDasharray="3 3" />
                <XAxis dataKey="bucket" stroke="#5B6577" fontSize={10} />
                <YAxis stroke="#5B6577" fontSize={11} allowDecimals={false} />
                <Tooltip contentStyle={CHART_TOOLTIP_STYLE} />
                <Bar dataKey="count" fill="#33D6C0" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </ChartPanel>

          <ChartPanel
            title="Most Frequent Unknown Classes"
            empty={data.most_frequent_unknown_classes.length === 0}
          >
            <ResponsiveContainer>
              <BarChart data={data.most_frequent_unknown_classes} layout="vertical">
                <CartesianGrid stroke="#232C39" strokeDasharray="3 3" />
                <XAxis type="number" stroke="#5B6577" fontSize={11} allowDecimals={false} />
                <YAxis dataKey="label" type="category" stroke="#5B6577" fontSize={11} width={70} />
                <Tooltip contentStyle={CHART_TOOLTIP_STYLE} />
                <Bar dataKey="count" fill="#F5A623" radius={[0, 3, 3, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </ChartPanel>

          <ChartPanel
            title="Cluster Distribution"
            empty={data.cluster_distribution.length === 0}
          >
            <ResponsiveContainer>
              <BarChart data={data.cluster_distribution}>
                <CartesianGrid stroke="#232C39" strokeDasharray="3 3" />
                <XAxis dataKey="suggested_label" stroke="#5B6577" fontSize={10} />
                <YAxis stroke="#5B6577" fontSize={11} allowDecimals={false} />
                <Tooltip contentStyle={CHART_TOOLTIP_STYLE} />
                <Bar dataKey="object_count" fill="#E8825C" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </ChartPanel>
        </div>
      )}
    </div>
  );
}
