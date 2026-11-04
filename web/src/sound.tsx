import { useEffect, useRef, useState, useSyncExternalStore } from "react";
import { Volume2, VolumeX } from "lucide-react";
import { useI18n } from "./i18n";
import type { Table } from "./types";

type Cue = "tile" | "turn" | "call" | "start" | "win" | "draw";
let context: AudioContext | undefined;
let enabled = false;
let volume = 0.35;
try {
  enabled = localStorage.getItem("riichilm.sound") === "on";
  const saved = localStorage.getItem("riichilm.volume");
  if (saved !== null && Number.isFinite(Number(saved))) volume = Math.max(0, Math.min(1, Number(saved)));
} catch { /* Sound remains usable without storage. */ }
const listeners = new Set<() => void>();
const subscribe = (listener: () => void) => { listeners.add(listener); return () => { listeners.delete(listener); }; };
function update() {
  try { localStorage.setItem("riichilm.sound", enabled ? "on" : "off"); localStorage.setItem("riichilm.volume", String(volume)); } catch { /* Session settings still work. */ }
  listeners.forEach((listener) => listener());
}
async function unlock() {
  if (!enabled || !("AudioContext" in window)) return;
  try { context ??= new AudioContext(); if (context.state === "suspended") await context.resume(); } catch { /* Unsupported devices keep the game playable. */ }
}
/** Short original synthesized cues; no downloads and no autoplay before a gesture. */
export function playCue(cue: Cue) {
  if (!enabled || !context || context.state !== "running" || document.hidden) return;
  const notes: Record<Cue, number[]> = {
    tile: [760, 390], turn: [659, 880], call: [440, 659, 880],
    start: [392, 523, 659], win: [523, 659, 784, 1047], draw: [440, 392, 330],
  };
  notes[cue].forEach((frequency, i) => {
    const oscillator = context!.createOscillator(), gain = context!.createGain();
    const time = context!.currentTime + i * (cue === "tile" ? 0.018 : 0.105);
    const duration = cue === "tile" ? 0.055 : 0.28;
    oscillator.type = cue === "tile" ? "triangle" : "sine";
    oscillator.frequency.setValueAtTime(frequency, time);
    if (cue === "tile") oscillator.frequency.exponentialRampToValueAtTime(frequency / 2, time + duration);
    gain.gain.setValueAtTime(0, time);
    gain.gain.linearRampToValueAtTime(volume * (cue === "tile" ? 0.16 : 0.13), time + 0.004);
    gain.gain.exponentialRampToValueAtTime(0.0001, time + duration);
    oscillator.connect(gain); gain.connect(context!.destination);
    oscillator.start(time); oscillator.stop(time + duration);
    oscillator.onended = () => { oscillator.disconnect(); gain.disconnect(); };
  });
}
export function SoundControl() {
  const { t } = useI18n();
  useSyncExternalStore(subscribe, () => `${enabled}:${volume}`);
  const [open, setOpen] = useState(false);
  return <div className="sound-control">
    <button type="button" className="table-tool sound-toggle" aria-label={t(enabled ? "关闭音效" : "开启音效")} aria-pressed={enabled}
      title={t(enabled ? "关闭音效" : "开启音效")} onClick={() => {
        enabled = !enabled; update(); void unlock().then(() => playCue("turn"));
      }}>{enabled ? <Volume2 size={17} /> : <VolumeX size={17} />}</button>
    <button type="button" className="sound-settings" aria-label={t("音量设置")} aria-expanded={open} onClick={() => setOpen(!open)}>⌄</button>
    {open && <label className="sound-popover">{t("音效音量")}<input type="range" aria-label={t("音效音量")} min="0" max="1" step="0.05" value={volume}
      onChange={(event) => { volume = Number(event.target.value); update(); }} onPointerUp={() => playCue("turn")} />
      <small>{Math.round(volume * 100)}%</small></label>}
  </div>;
}
export function useTableSound(table: Table) {
  const previous = useRef<Table | null>(null);
  useEffect(() => {
    const gesture = () => { void unlock(); };
    window.addEventListener("pointerdown", gesture); window.addEventListener("keydown", gesture);
    return () => { window.removeEventListener("pointerdown", gesture); window.removeEventListener("keydown", gesture); };
  }, []);
  useEffect(() => {
    const old = previous.current;
    previous.current = table;
    if (!old || old.table_id !== table.table_id) return;
    if (old.state !== table.state && old.state !== "round_result" && (table.state === "round_result" || table.state === "finished")) {
      playCue(table.result?.winners?.length ? "win" : "draw"); return;
    }
    if (table.state !== "playing") return;
    if (old.round_id !== table.round_id || old.state === "loading") { playCue("start"); return; }
    if (table.players?.some((p) => {
      const before = old.players?.find((entry) => entry.seat === p.seat);
      return before && ((!before.riichi && p.riichi) || p.melds.length > before.melds.length);
    })) { playCue("call"); return; }
    if (table.players?.some((p) => p.river.length > (old.players?.find((entry) => entry.seat === p.seat)?.river.length ?? p.river.length))) playCue("tile");
    if (!table.spectator && table.candidates?.length && !old.candidates?.length) playCue("turn");
  }, [table]);
}
