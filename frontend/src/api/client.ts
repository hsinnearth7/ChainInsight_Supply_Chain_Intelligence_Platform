import type { PipelineRun, AnalysisResult, RunStatus, KPIData, InventoryRow, ChartInfo, IngestResponse } from '../types/api';

const BASE = '/api';

async function fetchJSON<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status}: ${body}`);
  }
  return res.json();
}

export const api = {
  // Ingest
  async ingest(file: File): Promise<IngestResponse> {
    const form = new FormData();
    form.append('file', file);
    const res = await fetch(`${BASE}/ingest`, { method: 'POST', body: form });
    if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
    return res.json();
  },

  async ingestExisting(filename: string): Promise<IngestResponse> {
    // Upload existing file from data/raw by triggering a fetch to a known path
    const res = await fetch(`${BASE}/ingest`, {
      method: 'POST',
      body: (() => {
        const form = new FormData();
        // We'll use a small hack: create a blob with the filename
        form.append('file', new Blob(['']), filename);
        return form;
      })(),
    });
    if (!res.ok) throw new Error(`Failed: ${res.status}`);
    return res.json();
  },

  // Runs
  listRuns: () => fetchJSON<PipelineRun[]>(`${BASE}/runs`),

  getRun: (batchId: string) => fetchJSON<PipelineRun>(`${BASE}/runs/${batchId}`),

  getRunStatus: (batchId: string) => fetchJSON<RunStatus>(`${BASE}/runs/${batchId}/status`),

  // Analysis
  getAnalysis: (batchId: string, type: string) =>
    fetchJSON<AnalysisResult>(`${BASE}/runs/${batchId}/analysis/${type}`),

  // KPIs
  getKPIs: (batchId: string) => fetchJSON<Record<string, unknown>>(`${BASE}/runs/${batchId}/kpis`),
  getLatestKPIs: () => fetchJSON<KPIData>(`${BASE}/latest/kpis`),
  getKPIHistory: (limit = 20) => fetchJSON<KPIData[]>(`${BASE}/history/kpis?limit=${limit}`),

  // Charts
  listCharts: (batchId: string) => fetchJSON<ChartInfo[]>(`${BASE}/runs/${batchId}/charts`),
  getChartURL: (batchId: string, chartName: string) => `${BASE}/runs/${batchId}/charts/${chartName}`,

  // Data
  getInventoryData: (batchId: string) => fetchJSON<InventoryRow[]>(`${BASE}/runs/${batchId}/data`),
};
