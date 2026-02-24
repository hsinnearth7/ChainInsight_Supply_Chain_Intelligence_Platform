export type Language = 'en' | 'zh' | 'ja';

export const LANGUAGE_LABELS: Record<Language, string> = {
  en: 'EN',
  zh: '中文',
  ja: '日本語',
};

const translations = {
  // ── Header ──
  'header.subtitle': {
    en: 'Real-Time Supply Chain Inventory Analytics',
    zh: '即時供應鏈庫存分析',
    ja: 'リアルタイムサプライチェーン在庫分析',
  },
  'header.light': { en: 'Light', zh: '亮色', ja: 'ライト' },
  'header.dark': { en: 'Dark', zh: '暗色', ja: 'ダーク' },

  // ── Sidebar ──
  'sidebar.subtitle': {
    en: 'Supply Chain Analytics',
    zh: '供應鏈分析平台',
    ja: 'サプライチェーン分析',
  },
  'sidebar.latestRun': { en: 'Latest Run', zh: '最新執行', ja: '最新実行' },
  'nav.dashboard': { en: 'Dashboard', zh: '儀表板', ja: 'ダッシュボード' },
  'nav.upload': { en: 'Upload & Run', zh: '上傳與執行', ja: 'アップロード' },
  'nav.stats': { en: 'Statistics', zh: '統計分析', ja: '統計分析' },
  'nav.supplyChain': { en: 'Supply Chain', zh: '供應鏈', ja: 'サプライチェーン' },
  'nav.ml': { en: 'ML / AI', zh: '機器學習', ja: '機械学習' },
  'nav.rl': { en: 'RL Optimization', zh: '強化學習', ja: '強化学習' },
  'nav.history': { en: 'History', zh: '歷史記錄', ja: '履歴' },

  // ── Dashboard ──
  'dashboard.title': { en: 'Dashboard', zh: '儀表板', ja: 'ダッシュボード' },
  'dashboard.loading': {
    en: 'Loading dashboard...',
    zh: '載入儀表板中...',
    ja: 'ダッシュボード読込中...',
  },
  'dashboard.noData': {
    en: 'No completed pipeline runs yet. Go to Upload & Run to start.',
    zh: '尚無已完成的管線執行。請前往「上傳與執行」開始。',
    ja: 'パイプライン実行がありません。「アップロード」から開始してください。',
  },
  'dashboard.inventoryTurnover': {
    en: 'Inventory Turnover',
    zh: '庫存周轉率',
    ja: '在庫回転率',
  },
  'dashboard.avgDSI': { en: 'Avg DSI', zh: '平均庫存天數', ja: '平均在庫日数' },
  'dashboard.oosRate': { en: 'OOS Rate', zh: '缺貨率', ja: '欠品率' },
  'dashboard.slowMovingValue': {
    en: 'Slow-Moving Value',
    zh: '滯銷品金額',
    ja: '滞留在庫額',
  },
  'dashboard.totalValue': { en: 'Total Value', zh: '庫存總值', ja: '在庫総額' },
  'dashboard.days': { en: 'Days', zh: '天', ja: '日' },
  'dashboard.stockStatus': {
    en: 'Stock Status Distribution',
    zh: '庫存狀態分布',
    ja: '在庫ステータス分布',
  },
  'dashboard.categoryValue': {
    en: 'Inventory Value by Category',
    zh: '各類別庫存金額',
    ja: 'カテゴリ別在庫額',
  },
  'dashboard.vendorPerf': {
    en: 'Vendor Performance',
    zh: '供應商績效',
    ja: 'ベンダー実績',
  },
  'dashboard.stockoutAlerts': {
    en: 'Stockout Alerts (DSI < Lead Time)',
    zh: '缺貨警報（DSI < 前置時間）',
    ja: '欠品アラート（DSI < リードタイム）',
  },

  // ── Upload ──
  'upload.title': {
    en: 'Upload & Run Pipeline',
    zh: '上傳與執行管線',
    ja: 'アップロード & パイプライン実行',
  },
  'upload.dragDrop': {
    en: 'Drag & drop a CSV file here',
    zh: '拖曳 CSV 檔案至此',
    ja: 'CSVファイルをドラッグ＆ドロップ',
  },
  'upload.browse': { en: 'or click to browse', zh: '或點擊瀏覽', ja: 'またはクリックして選択' },
  'upload.useExisting': {
    en: 'Use existing dirty_inventory_10000.csv',
    zh: '使用既有的 dirty_inventory_10000.csv',
    ja: '既存の dirty_inventory_10000.csv を使用',
  },
  'upload.etlSummary': { en: 'ETL Summary', zh: 'ETL 摘要', ja: 'ETL サマリー' },
  'upload.pipelineComplete': {
    en: 'Pipeline complete! Visit Dashboard to view results.',
    zh: '管線已完成！請前往儀表板查看結果。',
    ja: 'パイプライン完了！ダッシュボードで結果をご確認ください。',
  },

  // ── Pipeline Progress ──
  'pipeline.title': {
    en: 'Pipeline Progress',
    zh: '管線進度',
    ja: 'パイプライン進捗',
  },
  'pipeline.live': { en: 'Live', zh: '即時', ja: 'ライブ' },
  'pipeline.disconnected': { en: 'Disconnected', zh: '已斷線', ja: '切断' },
  'stage.etl': { en: 'ETL Pipeline', zh: 'ETL 管線', ja: 'ETL パイプライン' },
  'stage.stats': { en: 'Statistical Analysis', zh: '統計分析', ja: '統計分析' },
  'stage.supply_chain': { en: 'Supply Chain', zh: '供應鏈優化', ja: 'サプライチェーン' },
  'stage.ml': { en: 'ML / AI Analysis', zh: '機器學習分析', ja: '機械学習分析' },
  'stage.rl': { en: 'RL Optimization', zh: '強化學習優化', ja: '強化学習最適化' },

  // ── Stats ──
  'stats.title': { en: 'Statistical Analysis', zh: '統計分析', ja: '統計分析' },
  'stats.loading': {
    en: 'Loading statistics...',
    zh: '載入統計資料中...',
    ja: '統計情報読込中...',
  },
  'stats.noData': {
    en: 'No data available. Run a pipeline first.',
    zh: '無可用資料。請先執行管線。',
    ja: 'データがありません。先にパイプラインを実行してください。',
  },
  'stats.tabInteractive': { en: 'Interactive', zh: '互動圖表', ja: 'インタラクティブ' },
  'stats.tabPNG': { en: 'PNG Charts', zh: 'PNG 圖表', ja: 'PNG チャート' },
  'stats.tabData': { en: 'Raw Data', zh: '原始資料', ja: '生データ' },
  'stats.correlationHeatmap': {
    en: 'Correlation Heatmap',
    zh: '相關性熱力圖',
    ja: '相関ヒートマップ',
  },
  'stats.unitCostDist': {
    en: 'Unit Cost Distribution',
    zh: '單位成本分布',
    ja: '単位原価分布',
  },
  'stats.vendorInventoryAvg': {
    en: 'Vendor Inventory Value (Avg)',
    zh: '供應商平均庫存金額',
    ja: 'ベンダー別平均在庫額',
  },
  'stats.inventorySnapshot': {
    en: 'Inventory Snapshot',
    zh: '庫存快照',
    ja: '在庫スナップショット',
  },

  // ── Supply Chain ──
  'sc.title': {
    en: 'Supply Chain Optimization',
    zh: '供應鏈優化',
    ja: 'サプライチェーン最適化',
  },
  'sc.loading': {
    en: 'Loading supply chain data...',
    zh: '載入供應鏈資料中...',
    ja: 'サプライチェーンデータ読込中...',
  },
  'sc.noData': { en: 'No data available.', zh: '無可用資料。', ja: 'データがありません。' },
  'sc.eoqCalc': { en: 'EOQ Calculator', zh: 'EOQ 計算器', ja: 'EOQ 計算機' },
  'sc.annualDemand': { en: 'Annual Demand', zh: '年需求量', ja: '年間需要量' },
  'sc.orderCost': { en: 'Order Cost ($)', zh: '訂購成本 ($)', ja: '発注コスト ($)' },
  'sc.holdingCost': {
    en: 'Holding Cost ($/unit/yr)',
    zh: '持有成本 ($/單位/年)',
    ja: '保管コスト ($/個/年)',
  },
  'sc.optimalQty': {
    en: 'Optimal Order Quantity',
    zh: '最佳訂購量',
    ja: '最適発注量',
  },
  'sc.units': { en: 'units', zh: '單位', ja: '個' },
  'sc.holdingCostLabel': { en: 'Holding Cost', zh: '持有成本', ja: '保管コスト' },
  'sc.orderingCostLabel': { en: 'Ordering Cost', zh: '訂購成本', ja: '発注コスト' },
  'sc.totalCostLabel': { en: 'Total Cost', zh: '總成本', ja: '総コスト' },
  'sc.monteCarlo': {
    en: 'Monte Carlo Stockout Simulation',
    zh: '蒙地卡羅缺貨模擬',
    ja: 'モンテカルロ欠品シミュレーション',
  },
  'sc.simulations': { en: 'Simulations', zh: '模擬次數', ja: 'シミュレーション回数' },
  'sc.dailyDemandMean': { en: 'Daily Demand (mean)', zh: '日需求量 (均值)', ja: '日次需要 (平均)' },
  'sc.demandStdDev': { en: 'Demand Std Dev', zh: '需求標準差', ja: '需要標準偏差' },
  'sc.leadTimeDays': { en: 'Lead Time (days)', zh: '前置時間 (天)', ja: 'リードタイム (日)' },
  'sc.currentStock': { en: 'Current Stock', zh: '當前庫存', ja: '現在庫' },
  'sc.runSimulation': { en: 'Run Simulation', zh: '執行模擬', ja: 'シミュレーション実行' },
  'sc.stockoutProb': {
    en: 'Stockout Probability',
    zh: '缺貨機率',
    ja: '欠品確率',
  },
  'sc.pipelineCharts': { en: 'Pipeline Charts', zh: '管線圖表', ja: 'パイプラインチャート' },

  // ── ML ──
  'ml.title': { en: 'ML / AI Analysis', zh: '機器學習分析', ja: '機械学習分析' },
  'ml.loading': { en: 'Loading ML results...', zh: '載入 ML 結果中...', ja: 'ML結果読込中...' },
  'ml.noData': { en: 'No data available.', zh: '無可用資料。', ja: 'データがありません。' },
  'ml.algorithmSummary': {
    en: 'Algorithm Summary',
    zh: '演算法摘要',
    ja: 'アルゴリズムサマリー',
  },
  'ml.models': { en: 'models', zh: '個模型', ja: 'モデル' },
  'ml.algorithm': { en: 'Algorithm', zh: '演算法', ja: 'アルゴリズム' },
  'ml.category': { en: 'Category', zh: '類別', ja: 'カテゴリ' },
  'ml.status': { en: 'Status', zh: '狀態', ja: 'ステータス' },
  'ml.metric': { en: 'Metric', zh: '指標', ja: 'メトリクス' },
  'ml.analysisCharts': { en: 'Analysis Charts', zh: '分析圖表', ja: '分析チャート' },

  // ── RL ──
  'rl.title': { en: 'RL Optimization', zh: '強化學習優化', ja: '強化学習最適化' },
  'rl.loading': { en: 'Loading RL results...', zh: '載入 RL 結果中...', ja: 'RL結果読込中...' },
  'rl.noData': { en: 'No data available.', zh: '無可用資料。', ja: 'データがありません。' },
  'rl.bestAgent': { en: 'Best Agent', zh: '最佳代理', ja: '最良エージェント' },
  'rl.finalReward': { en: 'Final Reward', zh: '最終獎勵', ja: '最終報酬' },
  'rl.serviceLevel': { en: 'Service Level', zh: '服務水準', ja: 'サービスレベル' },
  'rl.agentsTrained': { en: 'Agents Trained', zh: '已訓練代理數', ja: '訓練済エージェント' },
  'rl.agentComparison': { en: 'Agent Comparison', zh: '代理比較', ja: 'エージェント比較' },
  'rl.rewardCurves': {
    en: 'Reward Learning Curves (Interactive)',
    zh: '獎勵學習曲線（互動）',
    ja: '報酬学習曲線（インタラクティブ）',
  },
  'rl.envSpec': { en: 'Environment Specification', zh: '環境規格', ja: '環境仕様' },
  'rl.agents': { en: 'Agents', zh: '代理', ja: 'エージェント' },
  'rl.pipelineCharts': { en: 'Pipeline Charts', zh: '管線圖表', ja: 'パイプラインチャート' },

  // ── RL Environment Spec ──
  'rl.stateSpace': { en: 'State Space', zh: '狀態空間', ja: '状態空間' },
  'rl.actionSpace': { en: 'Action Space', zh: '動作空間', ja: '行動空間' },
  'rl.episodes': { en: 'Episodes', zh: '回合數', ja: 'エピソード数' },
  'rl.stepsPerEpisode': { en: 'Steps/Episode', zh: '每回合步數', ja: 'ステップ/エピソード' },
  'rl.reward': { en: 'Reward', zh: '獎勵', ja: '報酬' },

  // ── History ──
  'history.title': { en: 'Pipeline History', zh: '管線歷史', ja: 'パイプライン履歴' },
  'history.loading': { en: 'Loading history...', zh: '載入歷史記錄中...', ja: '履歴読込中...' },
  'history.kpiTrends': { en: 'KPI Trends', zh: 'KPI 趨勢', ja: 'KPI トレンド' },
  'history.pipelineRuns': { en: 'Pipeline Runs', zh: '管線執行記錄', ja: 'パイプライン実行' },
  'history.batchId': { en: 'Batch ID', zh: '批次 ID', ja: 'バッチ ID' },
  'history.status': { en: 'Status', zh: '狀態', ja: 'ステータス' },
  'history.sourceFile': { en: 'Source File', zh: '來源檔案', ja: 'ソースファイル' },
  'history.started': { en: 'Started', zh: '開始時間', ja: '開始日時' },
  'history.completed': { en: 'Completed', zh: '完成時間', ja: '完了日時' },
  'history.duration': { en: 'Duration', zh: '耗時', ja: '所要時間' },
  'history.noRuns': {
    en: 'No pipeline runs yet.',
    zh: '尚無管線執行記錄。',
    ja: 'パイプライン実行履歴はありません。',
  },

  // ── Common ──
  'common.noData': { en: 'No data available', zh: '無可用資料', ja: 'データなし' },
  'common.downloadCSV': { en: 'Download CSV', zh: '下載 CSV', ja: 'CSV ダウンロード' },
  'common.orderQuantity': { en: 'Order Quantity', zh: '訂購數量', ja: '発注数量' },
} as const;

export type TranslationKey = keyof typeof translations;

export function getTranslation(key: TranslationKey, lang: Language): string {
  const entry = translations[key];
  return entry?.[lang] ?? entry?.en ?? key;
}

export default translations;
