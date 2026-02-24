import { useEffect, useState } from 'react';
import { api } from '../api/client';
import { useAppStore } from '../stores/appStore';
import ChartImage from '../components/ChartImage';
import DataTable from '../components/DataTable';
import LoadingSpinner from '../components/LoadingSpinner';
import type { AnalysisResult } from '../types/api';

const ML_CHARTS = [
  { file: 'chart_15_classification_comparison.png', label: 'Classification Comparison' },
  { file: 'chart_16_feature_importance.png', label: 'Feature Importance' },
  { file: 'chart_17_regression_prediction.png', label: 'Regression Prediction' },
  { file: 'chart_18_clustering_analysis.png', label: 'Clustering Analysis' },
  { file: 'chart_19_pca_tsne.png', label: 'PCA / t-SNE' },
  { file: 'chart_20_anomaly_detection.png', label: 'Anomaly Detection' },
  { file: 'chart_21_genetic_algorithm.png', label: 'Genetic Algorithm' },
  { file: 'chart_22_algorithm_overview.png', label: 'Algorithm Overview' },
];

const STATUS_COLORS: Record<string, string> = {
  trained: 'text-ci-success',
  converged: 'text-ci-success',
  completed: 'text-ci-success',
  failed: 'text-ci-danger',
  skipped: 'text-ci-gray',
};

export default function MLPage() {
  const [batchId, setBatchId] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
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
      const data = await api.getAnalysis(bid, 'ml');
      setAnalysis(data);
    } catch {
      // no data
    } finally {
      setLoading(false);
    }
  }

  if (loading) return <LoadingSpinner text="Loading ML results..." />;
  if (!batchId) return <div className="text-ci-gray text-center py-12">No data available.</div>;

  // Parse ML results for algorithm table
  const mlResults = analysis?.kpis as Record<string, unknown> || {};
  const algorithms: Record<string, unknown>[] = [];

  // Try to extract algorithm results from various possible structures
  const categories = ['classification', 'regression', 'clustering', 'anomaly_detection', 'feature_engineering', 'genetic_algorithm'];
  for (const cat of categories) {
    const catData = mlResults[cat] as Record<string, unknown> | undefined;
    if (catData && typeof catData === 'object') {
      for (const [name, details] of Object.entries(catData)) {
        if (typeof details === 'object' && details) {
          algorithms.push({
            category: cat.replace(/_/g, ' '),
            algorithm: name.replace(/_/g, ' '),
            ...details as Record<string, unknown>,
          });
        }
      }
    }
  }

  // If no parsed algorithms, build from top-level keys
  if (algorithms.length === 0) {
    for (const [key, val] of Object.entries(mlResults)) {
      if (typeof val === 'object' && val && !Array.isArray(val)) {
        const entry = val as Record<string, unknown>;
        algorithms.push({
          algorithm: key.replace(/_/g, ' '),
          status: entry.status || 'completed',
          accuracy: entry.accuracy ?? entry.score ?? '—',
          ...entry,
        });
      }
    }
  }

  return (
    <div className="space-y-6">
      <h2 className="text-xl font-bold">ML / AI Analysis</h2>

      {/* Algorithm Summary Table */}
      {algorithms.length > 0 && (
        <div className="bg-white dark:bg-ci-dark-card rounded-lg border border-gray-200 dark:border-gray-700 p-4">
          <h3 className="text-sm font-medium mb-3">Algorithm Summary ({algorithms.length} models)</h3>
          <div className="overflow-auto max-h-[400px]">
            <table className="w-full text-sm">
              <thead className="bg-gray-50 dark:bg-gray-800 sticky top-0">
                <tr>
                  <th className="px-3 py-2 text-left text-xs font-medium text-ci-gray">Algorithm</th>
                  <th className="px-3 py-2 text-left text-xs font-medium text-ci-gray">Category</th>
                  <th className="px-3 py-2 text-left text-xs font-medium text-ci-gray">Status</th>
                  <th className="px-3 py-2 text-left text-xs font-medium text-ci-gray">Metric</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100 dark:divide-gray-700">
                {algorithms.map((algo, i) => {
                  const status = String(algo.status || 'completed');
                  const metric = algo.accuracy ?? algo.score ?? algo.r2 ?? '—';
                  return (
                    <tr key={i} className="hover:bg-gray-50 dark:hover:bg-gray-700/30">
                      <td className="px-3 py-1.5 font-medium capitalize">{String(algo.algorithm)}</td>
                      <td className="px-3 py-1.5 text-ci-gray capitalize">{String(algo.category || '—')}</td>
                      <td className={`px-3 py-1.5 capitalize ${STATUS_COLORS[status] || ''}`}>{status}</td>
                      <td className="px-3 py-1.5 font-mono">{typeof metric === 'number' ? metric.toFixed(4) : String(metric)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* PNG Charts */}
      <h3 className="text-sm font-medium">Analysis Charts</h3>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {ML_CHARTS.map((chart) => (
          <div key={chart.file}>
            <p className="text-xs text-ci-gray mb-1">{chart.label}</p>
            <ChartImage src={api.getChartURL(batchId, chart.file)} alt={chart.label} />
          </div>
        ))}
      </div>
    </div>
  );
}
