import type { StageStatus } from '../hooks/usePipelineProgress';
import { useTranslation } from '../i18n/useTranslation';
import type { TranslationKey } from '../i18n/translations';

const STAGE_LABEL_KEYS: Record<string, TranslationKey> = {
  etl: 'stage.etl',
  stats: 'stage.stats',
  supply_chain: 'stage.supply_chain',
  ml: 'stage.ml',
  rl: 'stage.rl',
};

interface PipelineProgressProps {
  stages: StageStatus[];
  overallPct: number;
  pipelineStatus: string;
  connected: boolean;
}

export default function PipelineProgress({ stages, overallPct, pipelineStatus, connected }: PipelineProgressProps) {
  const { t } = useTranslation();

  return (
    <div className="bg-white dark:bg-ci-dark-card rounded-lg border border-gray-200 dark:border-gray-700 p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium">{t('pipeline.title')}</h3>
        <div className="flex items-center gap-2">
          <span className={`w-2 h-2 rounded-full ${connected ? 'bg-ci-success' : 'bg-ci-gray'}`} />
          <span className="text-xs text-ci-gray">{connected ? t('pipeline.live') : t('pipeline.disconnected')}</span>
        </div>
      </div>

      {/* Overall progress bar */}
      <div className="mb-4">
        <div className="flex justify-between text-xs text-ci-gray mb-1">
          <span className="capitalize">{pipelineStatus}</span>
          <span>{overallPct}%</span>
        </div>
        <div className="h-2 bg-gray-100 dark:bg-gray-700 rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-500 ${
              pipelineStatus === 'failed' ? 'bg-ci-danger' :
              pipelineStatus === 'completed' ? 'bg-ci-success' : 'bg-ci-primary'
            }`}
            style={{ width: `${overallPct}%` }}
          />
        </div>
      </div>

      {/* Stage breakdown */}
      <div className="space-y-2">
        {stages.map((s) => (
          <div key={s.stage} className="flex items-center gap-3">
            <span className="text-sm w-5 text-center">
              {s.status === 'completed' ? '✅' : s.status === 'running' ? '⏳' : s.status === 'failed' ? '❌' : '⬜'}
            </span>
            <span className={`text-sm flex-1 ${s.status === 'running' ? 'font-medium text-ci-primary' : ''}`}>
              {STAGE_LABEL_KEYS[s.stage] ? t(STAGE_LABEL_KEYS[s.stage]) : s.stage}
            </span>
            <span className="text-xs text-ci-gray capitalize">{s.status}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
