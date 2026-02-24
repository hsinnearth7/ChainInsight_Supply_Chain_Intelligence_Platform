import { useAppStore } from '../stores/appStore';

export default function Header() {
  const { darkMode, toggleDarkMode, watchdogEvents, clearWatchdogEvents } = useAppStore();

  return (
    <header className="h-14 bg-white dark:bg-ci-dark-card border-b border-gray-200 dark:border-gray-700 flex items-center justify-between px-6 sticky top-0 z-10">
      <h2 className="text-sm font-medium text-ci-gray">
        Real-Time Supply Chain Inventory Analytics
      </h2>

      <div className="flex items-center gap-3">
        {/* Watchdog notification badge */}
        {watchdogEvents.length > 0 && (
          <button
            onClick={clearWatchdogEvents}
            className="relative text-sm text-ci-warning hover:text-ci-warning/80 transition"
            title="Watchdog auto-detected files"
          >
            🔔
            <span className="absolute -top-1 -right-1 bg-ci-danger text-white text-[10px] rounded-full w-4 h-4 flex items-center justify-center">
              {watchdogEvents.length}
            </span>
          </button>
        )}

        {/* Dark mode toggle */}
        <button
          onClick={toggleDarkMode}
          className="text-sm px-3 py-1.5 rounded-md border border-gray-200 dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-gray-700 transition"
        >
          {darkMode ? '☀️ Light' : '🌙 Dark'}
        </button>
      </div>
    </header>
  );
}
