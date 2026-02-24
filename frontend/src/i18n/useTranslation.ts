import { useAppStore } from '../stores/appStore';
import { getTranslation, type TranslationKey } from './translations';

export function useTranslation() {
  const language = useAppStore((s) => s.language);
  const t = (key: TranslationKey) => getTranslation(key, language);
  return { t, language };
}
