export type WSMessageType =
  | 'pipeline:progress'
  | 'pipeline:completed'
  | 'pipeline:failed'
  | 'watchdog:detected';

export interface WSPayload {
  stage: string;
  status: string;
  progress_pct: number;
  data: Record<string, unknown>;
}

export interface WSMessage {
  type: WSMessageType;
  batch_id: string;
  timestamp: string;
  payload: WSPayload;
}
