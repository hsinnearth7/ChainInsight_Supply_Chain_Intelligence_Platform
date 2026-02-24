import { useState, useCallback } from 'react';
import { useWebSocket } from './useWebSocket';
import type { WSMessage } from '../types/websocket';

export interface StageStatus {
  stage: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress_pct: number;
}

const STAGES = ['etl', 'stats', 'supply_chain', 'ml', 'rl'];

const initialStages: StageStatus[] = STAGES.map((s) => ({
  stage: s,
  status: 'pending',
  progress_pct: 0,
}));

export function usePipelineProgress(batchId: string | null) {
  const [stages, setStages] = useState<StageStatus[]>(initialStages);
  const [overallPct, setOverallPct] = useState(0);
  const [pipelineStatus, setPipelineStatus] = useState<'idle' | 'running' | 'completed' | 'failed'>('idle');

  const handleMessage = useCallback((msg: WSMessage) => {
    const { stage, status, progress_pct } = msg.payload;

    if (msg.type === 'pipeline:failed') {
      setPipelineStatus('failed');
      return;
    }

    setOverallPct(progress_pct);

    setStages((prev) =>
      prev.map((s) => {
        if (s.stage === stage) {
          return { ...s, status: status as StageStatus['status'], progress_pct };
        }
        return s;
      })
    );

    // Check if all done
    if (stage === 'rl' && status === 'completed') {
      setPipelineStatus('completed');
      setOverallPct(100);
    } else if (status === 'running' && pipelineStatus !== 'running') {
      setPipelineStatus('running');
    }
  }, [pipelineStatus]);

  const { connected } = useWebSocket({
    url: batchId ? `/ws/pipeline/${batchId}` : '',
    onMessage: handleMessage,
    enabled: !!batchId,
  });

  const reset = useCallback(() => {
    setStages(initialStages);
    setOverallPct(0);
    setPipelineStatus('idle');
  }, []);

  return { stages, overallPct, pipelineStatus, connected, reset };
}
