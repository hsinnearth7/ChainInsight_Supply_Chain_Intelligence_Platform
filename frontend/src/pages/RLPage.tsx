import { useEffect, useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { api } from '../api/client';
import { useAppStore } from '../stores/appStore';
import KPICard from '../components/KPICard';
import ChartImage from '../components/ChartImage';
import DataTable from '../components/DataTable';
import LoadingSpinner from '../components/LoadingSpinner';
import type { AnalysisResult } from '../types/api';

const RL_CHARTS = [
  { file: 'chart_23_rl_reward_curves.png', label: 'Reward Learning Curves' },
  { file: 'chart_24_rl_service_levels.png', label: 'Service Level Over Training' },
  { file: 'chart_25_rl_comparison.png', label: 'Agent Comparison' },
  { file: 'chart_26_rl_convergence.png', label: 'Convergence Analysis' },
  { file: 'chart_27_rl_reward_distribution.png', label: 'Reward Distribution' },
  { file: 'chart_28_rl_summary.png', label: 'Summary Comparison Table' },
];

const AGENT_COLORS = ['#2E86C1', '#27AE60', '#E74C3C', '#F39C12', '#8E44AD', '#1ABC9C'];
const AGENT_NAMES = ['Q-Learning', 'SARSA', 'DQN', 'PPO', 'A2C', 'GA-RL Hybrid'];

export default function RLPage() {
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
      const data = await api.getAnalysis(bid, 'rl');
      setAnalysis(data);
    } catch {
      // no data
    } finally {
      setLoading(false);
    }
  }

  if (loading) return <LoadingSpinner text="Loading RL results..." />;
  if (!batchId) return <div className="text-ci-gray text-center py-12">No data available.</div>;

  const rlData = (analysis?.kpis || {}) as Record<string, unknown>;
  const comparisonData = rlData.comparison_data as Record<string, unknown> | undefined;

  // Extract best agent info
  const bestAgent = rlData.best_agent as Record<string, unknown> | undefined;

  // Build interactive reward curves from comparison_data
  const rewardCurveData: Record<string, number>[] = [];
  if (comparisonData) {
    // comparison_data has agent_name -> { reward_history: number[], ... }
    const agents = Object.entries(comparisonData);
    const maxLen = Math.max(...agents.map(([, data]) => {
      const rh = (data as Record<string, unknown>).reward_history as number[] | undefined;
      return rh?.length || 0;
    }));

    for (let ep = 0; ep < maxLen; ep++) {
      const point: Record<string, number> = { episode: ep };
      for (const [name, data] of agents) {
        const rh = (data as Record<string, unknown>).reward_history as number[] | undefined;
        if (rh && ep < rh.length) {
          // Smooth with running average (window of 10)
          const start = Math.max(0, ep - 9);
          const slice = rh.slice(start, ep + 1);
          point[name] = slice.reduce((a, b) => a + b, 0) / slice.length;
        }
      }
      rewardCurveData.push(point);
    }
  }

  // Agent comparison table
  const agentTableData: Record<string, unknown>[] = [];
  if (comparisonData) {
    for (const [name, data] of Object.entries(comparisonData)) {
      const d = data as Record<string, unknown>;
      agentTableData.push({
        agent: name,
        final_reward: typeof d.final_reward === 'number' ? d.final_reward.toFixed(2) : '—',
        avg_reward: typeof d.avg_reward === 'number' ? d.avg_reward.toFixed(2) : '—',
        best_reward: typeof d.best_reward === 'number' ? d.best_reward.toFixed(2) : '—',
        service_level: typeof d.service_level === 'number' ? `${(d.service_level * 100).toFixed(1)}%` : '—',
        episodes: d.episodes ?? '—',
      });
    }
  }

  const agentKeys = comparisonData ? Object.keys(comparisonData) : [];

  return (
    <div className="space-y-6">
      <h2 className="text-xl font-bold">RL Optimization</h2>

      {/* Best Agent KPIs */}
      {bestAgent && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <KPICard title="Best Agent" value={String(bestAgent.name || '—')} icon="🏆" color="ci-primary" />
          <KPICard title="Final Reward" value={typeof bestAgent.final_reward === 'number' ? bestAgent.final_reward.toFixed(2) : '—'} icon="💰" color="ci-success" />
          <KPICard title="Service Level" value={typeof bestAgent.service_level === 'number' ? `${(bestAgent.service_level as number * 100).toFixed(1)}%` : '—'} icon="📊" color="ci-teal" />
          <KPICard title="Agents Trained" value={agentKeys.length || '6'} icon="🤖" color="ci-purple" />
        </div>
      )}

      {/* Agent Comparison Table */}
      {agentTableData.length > 0 && (
        <DataTable data={agentTableData} title="Agent Comparison" columns={['agent', 'final_reward', 'avg_reward', 'best_reward', 'service_level', 'episodes']} />
      )}

      {/* Interactive Reward Curves */}
      {rewardCurveData.length > 0 && (
        <div className="bg-white dark:bg-ci-dark-card rounded-lg border border-gray-200 dark:border-gray-700 p-4">
          <h3 className="text-sm font-medium mb-3">Reward Learning Curves (Interactive)</h3>
          <ResponsiveContainer width="100%" height={350}>
            <LineChart data={rewardCurveData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="episode" label={{ value: 'Episode', position: 'bottom' }} />
              <YAxis label={{ value: 'Reward', angle: -90, position: 'insideLeft' }} />
              <Tooltip />
              <Legend />
              {agentKeys.map((name, i) => (
                <Line
                  key={name}
                  type="monotone"
                  dataKey={name}
                  stroke={AGENT_COLORS[i % AGENT_COLORS.length]}
                  dot={false}
                  strokeWidth={2}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Environment Specs */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-white dark:bg-ci-dark-card rounded-lg border border-gray-200 dark:border-gray-700 p-4">
          <h3 className="text-sm font-medium mb-2">Environment Specification</h3>
          <table className="text-sm w-full">
            <tbody className="divide-y divide-gray-100 dark:divide-gray-700">
              {[
                ['State Space', '5-dim (stock, pending, days_since, demand_trend, stockout_days)'],
                ['Action Space', '5 discrete (0x, 0.5x, 1x, 1.5x, 2x EOQ)'],
                ['Episodes', '300'],
                ['Steps/Episode', '90'],
                ['Reward', '-(holding + stockout + ordering cost)'],
              ].map(([key, val]) => (
                <tr key={key}>
                  <td className="py-1 font-medium text-ci-gray">{key}</td>
                  <td className="py-1">{val}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="bg-white dark:bg-ci-dark-card rounded-lg border border-gray-200 dark:border-gray-700 p-4">
          <h3 className="text-sm font-medium mb-2">Agents</h3>
          <table className="text-sm w-full">
            <tbody className="divide-y divide-gray-100 dark:divide-gray-700">
              {AGENT_NAMES.map((name, i) => (
                <tr key={name}>
                  <td className="py-1">
                    <span className="inline-block w-2 h-2 rounded-full mr-2" style={{ backgroundColor: AGENT_COLORS[i] }} />
                    {name}
                  </td>
                  <td className="py-1 text-ci-gray text-xs">
                    {['Off-policy tabular', 'On-policy tabular', 'Deep Q-Network', 'Proximal Policy Opt.', 'Advantage Actor-Critic', 'GA + RL Hybrid'][i]}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* PNG Charts */}
      <h3 className="text-sm font-medium">Pipeline Charts</h3>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {RL_CHARTS.map((chart) => (
          <div key={chart.file}>
            <p className="text-xs text-ci-gray mb-1">{chart.label}</p>
            <ChartImage src={api.getChartURL(batchId, chart.file)} alt={chart.label} />
          </div>
        ))}
      </div>
    </div>
  );
}
