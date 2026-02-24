import { NavLink } from 'react-router-dom';
import { useAppStore } from '../stores/appStore';

const navItems = [
  { to: '/', label: 'Dashboard', icon: '📊' },
  { to: '/upload', label: 'Upload & Run', icon: '📤' },
  { to: '/stats', label: 'Statistics', icon: '📈' },
  { to: '/supply-chain', label: 'Supply Chain', icon: '🔗' },
  { to: '/ml', label: 'ML / AI', icon: '🤖' },
  { to: '/rl', label: 'RL Optimization', icon: '🎯' },
  { to: '/history', label: 'History', icon: '📋' },
];

export default function Sidebar() {
  const latestBatchId = useAppStore((s) => s.latestBatchId);

  return (
    <aside className="w-56 shrink-0 bg-white dark:bg-ci-dark-card border-r border-gray-200 dark:border-gray-700 flex flex-col h-screen sticky top-0">
      <div className="p-4 border-b border-gray-200 dark:border-gray-700">
        <h1 className="text-lg font-bold text-ci-primary">ChainInsight</h1>
        <p className="text-xs text-ci-gray mt-0.5">Supply Chain Analytics</p>
      </div>

      <nav className="flex-1 py-2 overflow-y-auto">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === '/'}
            className={({ isActive }) =>
              `flex items-center gap-2 px-4 py-2.5 text-sm transition-colors ${
                isActive
                  ? 'bg-ci-primary/10 text-ci-primary font-medium border-r-2 border-ci-primary'
                  : 'text-gray-600 dark:text-gray-400 hover:bg-gray-50 dark:hover:bg-gray-700/50'
              }`
            }
          >
            <span>{item.icon}</span>
            {item.label}
          </NavLink>
        ))}
      </nav>

      {latestBatchId && (
        <div className="p-3 border-t border-gray-200 dark:border-gray-700 text-xs text-ci-gray">
          <p>Latest Run</p>
          <p className="font-mono text-ci-text dark:text-ci-dark-text truncate" title={latestBatchId}>
            {latestBatchId.replace('batch_', '').slice(0, 20)}
          </p>
        </div>
      )}
    </aside>
  );
}
