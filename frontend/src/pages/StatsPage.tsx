import { useEffect, useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { api } from '../api/client';
import { useAppStore } from '../stores/appStore';
import ChartImage from '../components/ChartImage';
import DataTable from '../components/DataTable';
import LoadingSpinner from '../components/LoadingSpinner';
import type { InventoryRow, AnalysisResult } from '../types/api';

const CHART_NAMES = [
  { file: 'chart_0_inventory_health.png', label: 'Inventory Health Dashboard' },
  { file: 'chart_01_correlation_matrix.png', label: 'Correlation Matrix' },
  { file: 'chart_02_distribution_analysis.png', label: 'Distribution Analysis' },
  { file: 'chart_03_vendor_performance.png', label: 'Vendor Performance' },
  { file: 'chart_04_category_vendor_heatmap.png', label: 'Category-Vendor Heatmap' },
  { file: 'chart_05_regression_analysis.png', label: 'Regression Analysis' },
  { file: 'chart_06_category_risk_profile.png', label: 'Category Risk Profile' },
  { file: 'chart_07_outlier_risk_analysis.png', label: 'Outlier Risk Analysis' },
  { file: 'chart_08_pairplot_regression.png', label: 'Pairplot & Regression' },
];

export default function StatsPage() {
  const [tab, setTab] = useState<'interactive' | 'png' | 'data'>('interactive');
  const [batchId, setBatchId] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [inventory, setInventory] = useState<InventoryRow[]>([]);
  const [loading, setLoading] = useState(true);
  const latestBatchId = useAppStore((s) => s.latestBatchId);

  useEffect(() => {
    loadData();
  }, [latestBatchId]);

  async function loadData() {
    setLoading(true);
    try {
      let bid = latestBatchId;
      if (!bid) {
        const kpis = await api.getLatestKPIs();
        bid = kpis.batch_id;
      }
      setBatchId(bid);
      const [analysisData, invData] = await Promise.all([
        api.getAnalysis(bid, 'stats'),
        api.getInventoryData(bid),
      ]);
      setAnalysis(analysisData);
      setInventory(invData);
    } catch {
      // No data available
    } finally {
      setLoading(false);
    }
  }

  if (loading) return <LoadingSpinner text="Loading statistics..." />;
  if (!batchId) return <div className="text-ci-gray text-center py-12">No data available. Run a pipeline first.</div>;

  // Build interactive chart data
  const numericCols = ['unit_cost', 'current_stock', 'daily_demand_est', 'safety_stock_target', 'lead_time_days', 'reorder_point', 'inventory_value'];

  // Distribution histogram data (unit_cost buckets)
  const costValues = inventory.map((r) => r.unit_cost).filter(Boolean);
  const minCost = Math.min(...costValues);
  const maxCost = Math.max(...costValues);
  const bucketSize = (maxCost - minCost) / 10 || 1;
  const histData: { range: string; count: number }[] = [];
  for (let i = 0; i < 10; i++) {
    const lo = minCost + i * bucketSize;
    const hi = lo + bucketSize;
    const count = costValues.filter((v) => v >= lo && (i === 9 ? v <= hi : v < hi)).length;
    histData.push({ range: `$${lo.toFixed(0)}-${hi.toFixed(0)}`, count });
  }

  // Correlation heatmap data
  const corrMatrix = computeCorrelation(inventory, numericCols);

  // Vendor box-plot approximation data
  const vendorGroups = inventory.reduce<Record<string, number[]>>((acc, r) => {
    (acc[r.vendor_name] = acc[r.vendor_name] || []).push(r.inventory_value);
    return acc;
  }, {});
  const vendorBoxData = Object.entries(vendorGroups).map(([vendor, values]) => {
    values.sort((a, b) => a - b);
    return {
      vendor: vendor.split(' ').pop() || vendor,
      min: values[0],
      q1: values[Math.floor(values.length * 0.25)],
      median: values[Math.floor(values.length * 0.5)],
      q3: values[Math.floor(values.length * 0.75)],
      max: values[values.length - 1],
      avg: Math.round(values.reduce((a, b) => a + b, 0) / values.length),
    };
  });

  const tabs = [
    { key: 'interactive', label: 'Interactive' },
    { key: 'png', label: 'PNG Charts' },
    { key: 'data', label: 'Raw Data' },
  ] as const;

  return (
    <div className="space-y-6">
      <h2 className="text-xl font-bold">Statistical Analysis</h2>

      {/* Tabs */}
      <div className="flex gap-1 bg-gray-100 dark:bg-gray-800 rounded-lg p-1 w-fit">
        {tabs.map((t) => (
          <button
            key={t.key}
            onClick={() => setTab(t.key)}
            className={`px-4 py-1.5 text-sm rounded-md transition ${
              tab === t.key ? 'bg-white dark:bg-ci-dark-card shadow font-medium' : 'text-ci-gray hover:text-ci-text'
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {tab === 'interactive' && (
        <div className="space-y-6">
          {/* Correlation Heatmap */}
          <div className="bg-white dark:bg-ci-dark-card rounded-lg border border-gray-200 dark:border-gray-700 p-4">
            <h3 className="text-sm font-medium mb-3">Correlation Heatmap</h3>
            <div className="overflow-auto">
              <table className="text-xs">
                <thead>
                  <tr>
                    <th></th>
                    {numericCols.map((c) => (
                      <th key={c} className="px-1 py-1 text-center" style={{ writingMode: 'vertical-lr', maxHeight: 100 }}>
                        {c.replace(/_/g, ' ')}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {numericCols.map((row, ri) => (
                    <tr key={row}>
                      <td className="pr-2 text-right whitespace-nowrap">{row.replace(/_/g, ' ')}</td>
                      {numericCols.map((col, ci) => {
                        const val = corrMatrix[ri][ci];
                        return (
                          <td
                            key={col}
                            className="w-10 h-8 text-center border border-gray-100 dark:border-gray-700"
                            style={{ backgroundColor: corrColor(val), color: Math.abs(val) > 0.5 ? '#fff' : '#333' }}
                            title={`${row} vs ${col}: ${val.toFixed(2)}`}
                          >
                            {val.toFixed(2)}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Distribution Histogram */}
          <div className="bg-white dark:bg-ci-dark-card rounded-lg border border-gray-200 dark:border-gray-700 p-4">
            <h3 className="text-sm font-medium mb-3">Unit Cost Distribution</h3>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={histData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="range" tick={{ fontSize: 10 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#2E86C1" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Vendor Box Plot Approximation */}
          <div className="bg-white dark:bg-ci-dark-card rounded-lg border border-gray-200 dark:border-gray-700 p-4">
            <h3 className="text-sm font-medium mb-3">Vendor Inventory Value (Avg)</h3>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={vendorBoxData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="vendor" tick={{ fontSize: 11 }} />
                <YAxis tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`} />
                <Tooltip formatter={(v: number) => `$${v.toLocaleString()}`} />
                <Bar dataKey="avg" fill="#8E44AD" radius={[4, 4, 0, 0]}>
                  {vendorBoxData.map((_, i) => (
                    <Cell key={i} fill={['#2E86C1', '#27AE60', '#E74C3C', '#F39C12', '#8E44AD', '#1ABC9C', '#95A5A6'][i % 7]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {tab === 'png' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {CHART_NAMES.map((chart) => (
            <div key={chart.file}>
              <p className="text-xs text-ci-gray mb-1">{chart.label}</p>
              <ChartImage src={api.getChartURL(batchId, chart.file)} alt={chart.label} />
            </div>
          ))}
        </div>
      )}

      {tab === 'data' && (
        <DataTable
          data={inventory as unknown as Record<string, unknown>[]}
          title="Inventory Snapshot"
          enableCSVDownload
          maxHeight="600px"
        />
      )}
    </div>
  );
}

// Helpers

function computeCorrelation(data: InventoryRow[], cols: string[]): number[][] {
  const n = data.length;
  if (n < 2) return cols.map(() => cols.map(() => 0));

  const vals = cols.map((c) => data.map((r) => (r as unknown as Record<string, unknown>)[c] as number || 0));
  const means = vals.map((v) => v.reduce((a, b) => a + b, 0) / n);
  const stds = vals.map((v, i) => {
    const m = means[i];
    return Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / n);
  });

  return cols.map((_, i) =>
    cols.map((_, j) => {
      if (stds[i] === 0 || stds[j] === 0) return i === j ? 1 : 0;
      const cov = vals[i].reduce((a, v, k) => a + (v - means[i]) * (vals[j][k] - means[j]), 0) / n;
      return cov / (stds[i] * stds[j]);
    })
  );
}

function corrColor(val: number): string {
  if (val > 0.5) return '#2E86C1';
  if (val > 0.2) return '#85C1E9';
  if (val > -0.2) return '#F0F2F6';
  if (val > -0.5) return '#F5B7B1';
  return '#E74C3C';
}
