import { create } from 'zustand';
import type { Language } from '../i18n/translations';

interface WatchdogEvent {
  batch_id: string;
  file: string;
  timestamp: string;
}

interface AppState {
  darkMode: boolean;
  toggleDarkMode: () => void;

  language: Language;
  setLanguage: (lang: Language) => void;

  latestBatchId: string | null;
  setLatestBatchId: (id: string) => void;

  activePipelineBatchId: string | null;
  setActivePipelineBatchId: (id: string | null) => void;

  watchdogEvents: WatchdogEvent[];
  addWatchdogEvent: (event: WatchdogEvent) => void;
  clearWatchdogEvents: () => void;
}

export const useAppStore = create<AppState>((set) => ({
  darkMode: false,
  toggleDarkMode: () =>
    set((state) => {
      const next = !state.darkMode;
      if (next) {
        document.documentElement.classList.add('dark');
      } else {
        document.documentElement.classList.remove('dark');
      }
      return { darkMode: next };
    }),

  language: (localStorage.getItem('ci-lang') as Language) || 'en',
  setLanguage: (lang) => {
    localStorage.setItem('ci-lang', lang);
    set({ language: lang });
  },

  latestBatchId: null,
  setLatestBatchId: (id) => set({ latestBatchId: id }),

  activePipelineBatchId: null,
  setActivePipelineBatchId: (id) => set({ activePipelineBatchId: id }),

  watchdogEvents: [],
  addWatchdogEvent: (event) =>
    set((state) => ({
      watchdogEvents: [event, ...state.watchdogEvents].slice(0, 20),
    })),
  clearWatchdogEvents: () => set({ watchdogEvents: [] }),
}));
