import { useAppStore } from '../stores/appStore';
import { useTranslation } from '../i18n/useTranslation';
import { LANGUAGE_LABELS, type Language } from '../i18n/translations';

const LANGS: Language[] = ['en', 'zh', 'ja'];

export default function Header() {
  const { darkMode, toggleDarkMode, watchdogEvents, clearWatchdogEvents, language, setLanguage } = useAppStore();
  const { t } = useTranslation();

  return (
    <header className="h-14 bg-white dark:bg-ci-dark-card border-b border-gray-200 dark:border-gray-700 flex items-center justify-between px-6 sticky top-0 z-10">
      <h2 className="text-sm font-medium text-ci-gray">
        {t('header.subtitle')}
      </h2>

      <div className="flex items-center gap-3">
        {/* Language switcher */}
        <div className="flex bg-gray-100 dark:bg-gray-800 rounded-md overflow-hidden">
          {LANGS.map((lang) => (
            <button
              key={lang}
              onClick={() => setLanguage(lang)}
              className={`px-2 py-1 text-xs font-medium transition-colors ${
                language === lang
                  ? 'bg-ci-primary text-white'
                  : 'text-ci-gray hover:text-ci-text dark:hover:text-ci-dark-text'
              }`}
            >
              {LANGUAGE_LABELS[lang]}
            </button>
          ))}
        </div>

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
          {darkMode ? `☀️ ${t('header.light')}` : `🌙 ${t('header.dark')}`}
        </button>
      </div>
    </header>
  );
}
