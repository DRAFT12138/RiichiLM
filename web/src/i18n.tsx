import { useCallback, useSyncExternalStore } from "react";
import { messages } from "./messages";

export type Locale = "en" | "zh-CN" | "ja";
export type Translator = (message: string, values?: Record<string, string | number>) => string;
const storageKey = "riichilm.locale";
const supported = (value: string | null): value is Locale => value === "en" || value === "zh-CN" || value === "ja";
function readPreference(): Locale | "" {
  try { const value = localStorage.getItem(storageKey); return supported(value) ? value : ""; }
  catch { return ""; }
}
let preference = readPreference();
const listeners = new Set<() => void>();
const notify = () => listeners.forEach((listener) => listener());
export function browserLocale(languages: readonly string[]): Locale {
  for (const language of languages) {
    const base = language.toLowerCase().split(/[-_]/)[0];
    if (base === "zh") return "zh-CN";
    if (base === "ja") return "ja";
    if (base === "en") return "en";
  }
  return "en";
}
const currentLocale = () => preference || browserLocale(navigator.languages?.length ? navigator.languages : [navigator.language]);
function subscribe(listener: () => void) {
  listeners.add(listener);
  const changed = () => { preference = readPreference(); notify(); };
  window.addEventListener("languagechange", listener);
  window.addEventListener("storage", changed);
  return () => { listeners.delete(listener); window.removeEventListener("languagechange", listener); window.removeEventListener("storage", changed); };
}
export function translate(message: string, locale: Locale, values: Record<string, string | number> = {}): string {
  const match = /^([A-Z_]+: )(.+)$/.exec(message);
  if (match) return match[1] + translate(match[2], locale, values);
  const failed = /^请求失败 \((\d+)\)$/.exec(message);
  if (failed) return translate("请求失败 ({status})", locale, { status: failed[1] });
  const text = locale === "zh-CN" ? message : messages[message]?.[locale === "en" ? 0 : 1] ?? message;
  return text.replace(/\{(\w+)\}/g, (token, key: string) => String(values[key] ?? token));
}
export function useI18n() {
  // Include the preference so choosing Auto also rerenders when the resolved language is unchanged.
  const snapshot = useSyncExternalStore(subscribe, () => `${preference}|${currentLocale()}`);
  const locale = snapshot.split("|")[1] as Locale;
  const t: Translator = useCallback((message, values) => translate(message, locale, values), [locale]);
  const setLocale = (value: Locale | "") => {
    preference = value;
    try { if (value) localStorage.setItem(storageKey, value); else localStorage.removeItem(storageKey); } catch { /* Session preference still works without storage. */ }
    notify();
  };
  return { locale, preference, setLocale, t };
}
export function LanguageSwitcher() {
  const { preference, setLocale, t } = useI18n();
  return <label className="language-switcher">
    <span className="sr-only">{t("界面语言")}</span>
    <select aria-label={t("界面语言")} title={t("界面语言")} value={preference}
      onChange={(event) => setLocale(event.target.value as Locale | "")}>
      <option value="">{t("跟随浏览器")}</option>
      <option value="en">English</option><option value="zh-CN">简体中文</option><option value="ja">日本語</option>
    </select>
  </label>;
}
