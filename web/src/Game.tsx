import { useI18n, LanguageSwitcher, type Translator } from "./i18n";
import {
  ArrowRight,
  Bot,
  CircleHelp,
  DoorOpen,
  LoaderCircle,
  Pause,
  Play,
  SkipForward,
  MessageSquareText,
  UserRound,
  X,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import type { Candidate, Meld, Player, Table, SpectatorControl, Winner } from "./types";
import { useLabels } from "./labels";
import { Tile, tileName } from "./Tile";
import type { Cast } from "./cast";
import { characterNames } from "./cast";
import { SoundControl, useTableSound } from "./sound";

const victoryArt = import.meta.glob<string>("./assets/illustrations/victory-*.webp", { eager: true, query: "?url", import: "default" });
const characterVictoryArtwork = (id: number) => victoryArt[`./assets/illustrations/victory-${id}.webp`];

function doraFromIndicator(indicator: string): string {
  const clean = indicator.replace(/r$/, "");
  const winds = ["E", "S", "W", "N"];
  const wi = winds.indexOf(clean);
  if (wi !== -1) return winds[(wi + 1) % 4];
  const dragons = ["P", "F", "C"];
  const di = dragons.indexOf(clean);
  if (di !== -1) return dragons[(di + 1) % 3];
  const m = /^([1-9])([mps])$/.exec(clean);
  if (!m) return clean;
  return `${(Number(m[1]) % 9) + 1}${m[2]}`;
}

function isDoraTile(tile: string, doraSet: Set<string>): boolean {
  return tile.endsWith("r") || doraSet.has(tile.replace(/r$/, ""));
}

function countVisibleTile(table: Table, kind: string, viewer: number): number {
  let count = 0;
  for (const ind of table.dora_indicators || []) {
    if (ind.replace(/r$/, "") === kind) count++;
  }
  for (const p of table.players || []) {
    for (const r of p.river) {
      if (!r.called && r.tile.replace(/r$/, "") === kind) count++;
    }
    for (const m of p.melds) {
      for (const mt of m.tiles) {
        if (mt.replace(/r$/, "") === kind) count++;
      }
    }
    if (p.seat === viewer) {
      for (const ht of p.hand || []) {
        if (ht.replace(/r$/, "") === kind) count++;
      }
      if (p.drawn && p.drawn.replace(/r$/, "") === kind) count++;
    }
  }
  return count;
}

function handTierLabel(w: Winner, t: Translator): string | null {
  if (w.yakuman_units >= 1) return w.yakuman_units > 1 ? null : t("役满");
  const han = w.han ?? 0;
  const fu = w.fu ?? 0;
  if (han >= 13) return t("累计役满");
  if (han >= 11) return t("三倍满");
  if (han >= 8) return t("倍满");
  if (han >= 6) return t("跳满");
  if (han === 5 || (han === 4 && fu >= 40) || (han === 3 && fu >= 70)) return t("满贯");
  return null;
}

function calledTileIndex(meld: Meld, seat: number): number {
  if (meld.from == null || meld.from === seat || meld.name === "MELD_ANKAN") return -1;
  const rel = (meld.from - seat + 4) % 4;
  if (rel === 3) return 0;
  if (rel === 2) return Math.floor(meld.tiles.length / 2);
  if (rel === 1) return meld.tiles.length - 1;
  return -1;
}

function formatScore(score: number, viewerScore: number, isViewer: boolean, showDiff: boolean, locale: string): string {
  if (!showDiff || isViewer) return score.toLocaleString(locale);
  const diff = score - viewerScore;
  if (diff === 0) return "±0";
  return `${diff > 0 ? "+" : ""}${diff.toLocaleString(locale)}`;
}

function SeatPolicy({ table, seat }: { table: Table; seat: number }) {
  const { t } = useI18n();
  const info = table.controllers?.[seat] || table.opponents?.[seat - 1];
  const label = info?.kind === "llm"
    ? [...new Set([info.name, info.model].filter(Boolean))].join(" · ") || "LLM"
    : info?.kind === "model"
      ? info.name || info.checkpoint_id || t("训练模型")
      : info?.kind === "human" || (!table.spectator && seat === 0)
        ? t("真人")
        : t("随机策略");
  return <small className="seat-policy" title={label}>{label}</small>;
}

function Melds({ player, labels, doraSet }: { player: Player; labels: string[]; doraSet: Set<string> }) {
  const { t } = useI18n();
  const { actionNames } = useLabels();
  return (
    <div className="melds">
      {player.melds.map((m, i) => {
        const sidewaysIdx = calledTileIndex(m, player.seat);
        return (
          <div className="meld" key={i}>
            {m.from != null && (
              <small className="meld-source">
                {labels[m.from]} ·{" "}
                {actionNames[m.name.replace("MELD_", "A_")] || t("杠")}
              </small>
            )}
            {m.tiles.map((tile, j) => (
              <Tile
                key={j}
                tile={tile}
                small
                sideways={j === sidewaysIdx}
                dora={isDoraTile(tile, doraSet)}
                back={m.name === "MELD_ANKAN" && (j === 0 || j === 3)}
              />
            ))}
          </div>
        );
      })}
    </div>
  );
}

function Opponent({
  player,
  table,
  position,
  portrait,
  doraSet,
  viewerScore,
  showScoreDiff,
}: {
  player: Player;
  table: Table;
  position: number;
  portrait: number;
  doraSet: Set<string>;
  viewerScore: number;
  showScoreDiff: boolean;
}) {
  const { t, locale } = useI18n();
  const { observerNames, names, winds } = useLabels();
  const labels = table.spectator ? observerNames : names,
    active = table.thinking === player.seat;
  const formattedScore = formatScore(player.score, viewerScore, false, showScoreDiff, locale);
  const diffClass = showScoreDiff
    ? player.score > viewerScore
      ? "score-ahead"
      : player.score < viewerScore
        ? "score-behind"
        : ""
    : "";

  return (
    <>
      <div
        className={`opponent-seat seat-${position} ${active ? "thinking" : ""} ${player.riichi ? "in-riichi" : ""}`}
        data-seat={player.seat}
      >
        <div className="seat-info">
          <span className={`seat-wind ${player.dealer ? "is-dealer" : ""}`}>{winds[player.wind]}</span>
          <span className={`seat-avatar portrait-${portrait}`} title={t(characterNames[portrait])} aria-hidden="true">
            <Bot size={17} />
          </span>
          <div className="seat-identity">
            <strong>{labels[player.seat]}{active && <span className="seat-thinking"> · {t("思考中")}</span>}</strong>
            <SeatPolicy table={table} seat={player.seat} />
          </div>
          <b className={diffClass}>{formattedScore}</b>
          {player.dealer && <span className="dealer-label">{t("庄家")}</span>}
          {player.riichi && <span className="riichi-badge">{t("立直")}</span>}
        </div>
        <div className="opponent-hand">
          {Array.from({ length: player.hand_count }, (_, i) => (
            <Tile key={i} small back />
          ))}
        </div>
        <Melds player={player} labels={labels} doraSet={doraSet} />
      </div>
      {position !== 2 && (
        <div className={`compact-seat-score score-${position} ${active ? "thinking" : ""}`} data-seat={player.seat}>
          <span>
            {labels[player.seat]} · {winds[player.wind]}
          </span>
          <span className={`compact-portrait portrait-${portrait}`} title={t(characterNames[portrait])} aria-hidden="true" />
          <b className={diffClass}>{formattedScore}</b>
          {active && <span className="compact-thinking">{t("思考中")}</span>}
          {player.dealer && <span className="dealer-label">{t("庄家")}</span>}
          {player.riichi && <span className="riichi-badge">{t("立直")}</span>}
          <SeatPolicy table={table} seat={player.seat} />
        </div>
      )}
    </>
  );
}

function River({
  player,
  selectedTile,
  trigger,
  lastDiscardSeat,
  position,
  labels,
}: {
  player: Player;
  selectedTile: string | null;
  trigger: Table["response_trigger"];
  lastDiscardSeat: number | null;
  position: number;
  labels: string[];
}) {
  const { t } = useI18n();
  return (
    <div
      className={`river river-${position}`}
      aria-label={t("{seat}的牌河", { seat: labels[player.seat] })}
    >
      {player.river.map((r, i) => {
        const isLast = !r.called && i === player.river.length - 1 &&
          ((trigger?.kind === "discard" && trigger.source === player.seat && r.tile === trigger.tile) ||
            lastDiscardSeat === player.seat);
        return (
          <span
            className={[
              r.riichi && "sideways",
              r.called && "called",
              r.tsumogiri && !r.called && "tsumogiri",
              isLast && "last-discard",
              trigger?.kind === "discard" &&
                trigger.source === player.seat &&
                i === player.river.length - 1 &&
                !r.called &&
                r.tile === trigger.tile &&
                "response-tile",
              !r.called &&
                selectedTile !== null &&
                r.tile.replace(/r$/, "") === selectedTile &&
                "matching-tile",
            ]
              .filter(Boolean)
              .join(" ")}
            key={i}
          >
            <Tile tile={r.tile} small />
          </span>
        );
      })}
    </div>
  );
}

function LoadingRoom({ cast, spectator, pending, onExit }: {
  cast: Cast;
  spectator?: boolean;
  pending: boolean;
  onExit: () => Promise<boolean>;
}) {
  const { t } = useI18n();
  const { names, observerNames } = useLabels();
  const [seat, setSeat] = useState(0);
  const [cycling, setCycling] = useState(true);
  const labels = spectator ? observerNames : names;
  useEffect(() => {
    if (!cycling) return;
    const timer = window.setTimeout(() => setSeat((current) => (current + 1) % cast.seats.length), 4000);
    return () => window.clearTimeout(timer);
  }, [cycling, seat, cast.seats.length]);
  return (
    <section className="loading-screen" aria-label={t("入席准备")}>
      <div className="loading-room illustrated-loading">
        <div className="loading-character-art">
          {cast.seats.map((character, index) => (
            <img key={character} className={seat === index ? "is-active" : ""}
              src={characterVictoryArtwork(character)}
              alt={seat === index ? t("{name}的角色立绘", { name: t(characterNames[character]) }) : ""}
              aria-hidden={seat !== index} decoding="async" />
          ))}
        </div>
        <div className="loading-copy">
          <div className="eyebrow">RiichiLM · {t("入席准备")}</div>
          <div className="loading-tiles" aria-hidden="true">
            <Tile tile="E" /><Tile tile="5s" /><Tile tile="7p" />
          </div>
          <LoaderCircle className="spin" size={24} aria-hidden="true" />
          <h2>{t("好牌，值得等一等。")}</h2>
          <p role="status">{t("正在校验并加载你的对手，准备这场对局。")}</p>
          <div className="loading-character-name">{labels[seat]} · {t(characterNames[cast.seats[seat]])}</div>
          <div className="loading-character-controls" role="group" aria-label={t("本局角色")}>
            {cast.seats.map((character, index) => (
              <button key={character} type="button" aria-label={`${labels[index]} · ${t(characterNames[character])}`}
                aria-pressed={seat === index} onClick={() => setSeat(index)}>
                <span className={`loading-character-avatar portrait-${character}`} aria-hidden="true" />
                <span>{labels[index]}</span>
              </button>
            ))}
            <button type="button" className="loading-cycle-toggle"
              aria-label={cycling ? t("暂停立绘轮播") : t("继续立绘轮播")}
              title={cycling ? t("暂停立绘轮播") : t("继续立绘轮播")}
              onClick={() => setCycling((current) => !current)}>
              {cycling ? <Pause size={16} /> : <Play size={16} />}
            </button>
          </div>
          <button className="text-button" disabled={pending} onClick={() => void onExit()}>
            {t("取消并返回")}
          </button>
        </div>
      </div>
    </section>
  );
}

interface CallBanner {
  id: number;
  seat: number;
  character: number;
  label: string;
  tone: "riichi" | "call" | "win";
}

export function Game({
  table,
  onAction,
  onContinue,
  onExit,
  onControl,
  pending,
  cast,
  sessionInvalid,
}: {
  table: Table;
  onAction: (c: Candidate) => Promise<boolean>;
  onContinue: () => void;
  onExit: () => Promise<boolean>;
  onControl: (control: SpectatorControl, value?: number) => void;
  pending: boolean;
  cast: Cast;
  sessionInvalid: boolean;
}) {
  useTableSound(table);
  const { t, locale } = useI18n();
  const [introHidden, setIntroHidden] = useState(false);
  const [resultSeat, setResultSeat] = useState<number | null>(null);
  const [showScoreDiff, setShowScoreDiff] = useState(false);
  const [banner, setBanner] = useState<CallBanner | null>(null);
  const [lastDiscardSeat, setLastDiscardSeat] = useState<number | null>(null);

  useEffect(() => {
    setIntroHidden(false);
    const timer = window.setTimeout(() => setIntroHidden(true), 1800);
    return () => window.clearTimeout(timer);
  }, [table.table_id, table.round_id, table.state === "loading"]);

  const { actionNames, modes, names, observerNames, winds, yakuNames } = useLabels();
  const [group, setGroup] = useState<string | null>(null),
    [preview, setPreview] = useState<string | null>(null),
    [submitted, setSubmitted] = useState(false),
    [exitConfirm, setExitConfirm] = useState(false),
    [exitError, setExitError] = useState(false),
    [help, setHelp] = useState(false),
    [resultHidden, setResultHidden] = useState(false);
  const [notesOpen, setNotesOpen] = useState(false);

  const submissionLock = useRef(false),
    submittedVersion = useRef(0),
    modalRef = useRef<HTMLElement>(null),
    resultReopenRef = useRef<HTMLButtonElement>(null),
    exitButtonRef = useRef<HTMLButtonElement>(null),
    helpButtonRef = useRef<HTMLButtonElement>(null),
    exitWasOpen = useRef(false),
    notesRef = useRef<HTMLElement>(null),
    handRef = useRef<HTMLDivElement>(null),
    prevTableRef = useRef<Table | null>(null),
    bannerSeq = useRef(0);

  // Track last discard and anime-style call cut-in banners from snapshot diffs.
  useEffect(() => {
    const prev = prevTableRef.current;
    prevTableRef.current = table;
    if (!prev || prev.table_id !== table.table_id || prev.round_id !== table.round_id) {
      setLastDiscardSeat(null);
      setBanner(null);
      return;
    }
    if (table.state === "playing" && prev.players && table.players) {
      for (const p of table.players) {
        const oldP = prev.players.find((o) => o.seat === p.seat);
        if (!oldP) continue;
        if (p.river.length > oldP.river.length) {
          setLastDiscardSeat(p.seat);
        }
        if (!oldP.riichi && p.riichi) {
          bannerSeq.current++;
          setBanner({ id: bannerSeq.current, seat: p.seat, character: cast.seats[p.seat], label: actionNames.A_RIICHI_DISCARD, tone: "riichi" });
        } else if (p.melds.length > oldP.melds.length) {
          const newest = p.melds[p.melds.length - 1];
          const actionKey = newest.name.replace("MELD_", "A_");
          bannerSeq.current++;
          setBanner({ id: bannerSeq.current, seat: p.seat, character: cast.seats[p.seat], label: actionNames[actionKey] || t("杠"), tone: "call" });
        }
      }
    }
  }, [table, cast.seats, actionNames, t]);

  useEffect(() => {
    if (!banner) return;
    const timer = window.setTimeout(() => setBanner(null), 1150);
    return () => window.clearTimeout(timer);
  }, [banner]);

  useEffect(() => {
    setGroup(null);
    setPreview(null);
    setSubmitted(false);
    submissionLock.current = false;
  }, [table.revision, table.round_id, table.table_id, table.viewer]);

  useEffect(() => {
    if (
      submissionLock.current &&
      !pending &&
      !table.busy &&
      table.view_version > submittedVersion.current
    ) {
      submissionLock.current = false;
      setSubmitted(false);
    }
  }, [table.view_version, table.busy, pending]);

  useEffect(() => {
    setResultHidden(false);
    setResultSeat(null);
  }, [table.round_id, table.state]);

  useEffect(() => {
    if (resultHidden) resultReopenRef.current?.focus();
  }, [resultHidden]);

  useEffect(() => {
    if (notesOpen && !window.matchMedia("(min-width: 1280px)").matches) {
      notesRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }, [notesOpen]);

  useEffect(() => {
    if (exitWasOpen.current && !exitConfirm) exitButtonRef.current?.focus();
    exitWasOpen.current = exitConfirm;
  }, [exitConfirm]);

  useEffect(() => {
    const modal = modalRef.current;
    if (!modal) return;
    const focusable = () => Array.from(modal.querySelectorAll<HTMLButtonElement>("button:not(:disabled)"));
    focusable()[0]?.focus();
    const trapFocus = (event: KeyboardEvent) => {
      if (event.key !== "Tab") return;
      const buttons = focusable();
      if (!buttons.length) return;
      const first = buttons[0], last = buttons[buttons.length - 1];
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
    };
    modal.addEventListener("keydown", trapFocus);
    return () => modal.removeEventListener("keydown", trapFocus);
  }, [exitConfirm, resultHidden, sessionInvalid, table.state]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !sessionInvalid) {
        if (exitConfirm) setExitConfirm(false);
        else if ((table.state === "round_result" || table.state === "finished") && !resultHidden) setResultHidden(true);
        else if (group) setGroup(null);
        else if (help) { setHelp(false); helpButtonRef.current?.focus(); }
        return;
      }
      if (sessionInvalid || exitConfirm || help || event.metaKey || event.ctrlKey || event.altKey) return;
      const tag = (document.activeElement as HTMLElement | null)?.tagName;
      if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;

      if (event.key === "d" || event.key === "D") {
        setShowScoreDiff((v) => !v);
        return;
      }
      if (!table.spectator && (event.key === "ArrowLeft" || event.key === "ArrowRight")) {
        const buttons = Array.from(handRef.current?.querySelectorAll<HTMLButtonElement>("button.tile:not(:disabled)") ?? []);
        if (!buttons.length) return;
        const idx = buttons.indexOf(document.activeElement as HTMLButtonElement);
        event.preventDefault();
        if (idx === -1) buttons[event.key === "ArrowLeft" ? buttons.length - 1 : 0].focus();
        else buttons[(idx + (event.key === "ArrowRight" ? 1 : buttons.length - 1)) % buttons.length].focus();
        return;
      }
      if (table.spectator && table.state === "playing") {
        if (event.key === " " && tag !== "BUTTON") {
          event.preventDefault();
          if (!pending && !table.busy) onControl(table.paused ? "resume" : "pause");
        } else if (event.key === "." && table.paused && !pending && !table.busy) {
          onControl("step");
        } else if (/^[1-4]$/.test(event.key)) {
          onControl("view", Number(event.key) - 1);
        }
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [exitConfirm, group, help, resultHidden, sessionInvalid, table.state, table.spectator, table.paused, table.busy, pending, onControl]);

  const doraSet = useMemo(
    () => new Set((table.dora_indicators || []).map(doraFromIndicator)),
    [table.dora_indicators],
  );

  const viewer = table.viewer ?? 0,
    labels = table.spectator ? observerNames : names,
    position = (seat: number) => (seat - viewer + 4) % 4;
  const candidates = table.spectator ? [] : table.candidates || [],
    busy = pending || table.busy || submitted,
    own = table.players?.find((p) => p.seat === viewer);
  const discards = candidates.filter((c) => c.kind === "A_DISCARD"),
    special = Array.from(
      new Set(
        candidates
          .filter((c) => c.kind !== "A_DISCARD" && c.kind !== "A_PASS")
          .map((c) => c.kind),
      ),
    ),
    pass = candidates.find((c) => c.kind === "A_PASS");

  // Compare tile kinds, so red fives and ordinary fives match across all rivers.
  const selectedTile = busy ? null : (preview?.replace(/r$/, "") ?? null);
  const seenCount = selectedTile ? countVisibleTile(table, selectedTile, viewer) : 0;

  const findDiscard = (tile: string, drawn: boolean) => candidates.find((c) =>
    c.kind === (group === "A_RIICHI_DISCARD" ? "A_RIICHI_DISCARD" : "A_DISCARD") &&
    c.args[0] === `TILE_${tile}` && c.args[1] === (drawn ? "TSUMOGIRI" : "TEDASHI"),
  );
  const showIllegal = !busy && (discards.length > 0 || group === "A_RIICHI_DISCARD");
  const trigger =
    table.response_window && candidates.length ? table.response_trigger : null;
  const windAt = (relativePosition: number) => {
    const player = table.players?.find((p) => position(p.seat) === relativePosition);
    return player ? winds[player.wind] : "";
  };
  const playerAt = (relativePosition: number) =>
    table.players?.find((p) => position(p.seat) === relativePosition);

  async function submit(c: Candidate) {
    if (busy || submissionLock.current) return;
    submissionLock.current = true;
    submittedVersion.current = table.view_version;
    setSubmitted(true);
    setPreview(null);
    setGroup(null);
    if (!(await onAction(c))) {
      submissionLock.current = false;
      setSubmitted(false);
    }
  }
  const chooseTile = (tile: string, drawn = false) => {
    const c = findDiscard(tile, drawn);
    if (c) void submit(c);
  };
  const resultSeats = table.final
    ? table.final.ranks.flatMap((rank, seat) => rank === 1 ? [seat] : [])
    : [...new Set(table.result?.winners.map((winner) => winner.actor) ?? [])];
  const featuredSeat = resultSeat != null && resultSeats.includes(resultSeat) ? resultSeat : resultSeats[0];
  const featuredCharacter = featuredSeat == null ? undefined : cast.seats[featuredSeat];
  const status =
    table.thinking !== null && table.thinking !== undefined
      ? t("{seat}正在思考", { seat: labels[table.thinking] })
      : table.spectator
        ? table.paused ? t("观战已暂停") : t("AI 对战进行中")
        : busy
          ? t("正在处理你的选择")
          : candidates.length
            ? t("轮到你了，慢慢想。")
            : t("牌局进行中");

  const actionToneClass = (kind: string) => {
    if (kind.includes("WIN") || kind === "A_RON") return "win-button";
    if (kind === "A_RIICHI_DISCARD") return group === kind ? "riichi-button active-action" : "riichi-button";
    if (group === kind) return "active-action";
    if (kind === "A_CHI") return "light chi-button";
    if (kind === "A_PON") return "light pon-button";
    if (kind.includes("KAN")) return "light kan-button";
    if (kind === "A_ABORT_KYUUSHU") return "light abort-button";
    return "light";
  };

  return (
    <>
      {table.state === "loading" ? (
        <LoadingRoom key={table.table_id} cast={cast} spectator={table.spectator} pending={pending} onExit={onExit} />
      ) : table.state === "failed" ? (
        <div className="loading-room">
          <span className="error-symbol">!</span>
          <h2>{t("这桌暂时告一段落。")}</h2>
          <p role="alert">{t(table.error || "")}</p>
          <button className="button primary" disabled={pending} onClick={() => void onExit()}>
            {t("返回配置，重新开始")}<ArrowRight size={17} />
          </button>
        </div>
      ) : (
        own && (
          <>
            <div className={`game-stage ${table.spectator && notesOpen ? "with-notes" : ""}`}>
              <div className="felt-frame">
                <div className={`felt-table ${table.spectator ? "spectator-table" : ""}`}>
                  <div className="table-tools">
                    <SoundControl />
                    <LanguageSwitcher />
                    <span className="table-mode">{modes[table.mode].name}</span>
                    {resultHidden && (table.state === "round_result" || table.state === "finished") && (
                      <button ref={resultReopenRef} className="table-tool result-reopen" onClick={() => setResultHidden(false)}>{t("查看结算")}</button>
                    )}
                    <button
                      className="table-tool"
                      ref={helpButtonRef}
                      aria-label={t("操作说明")}
                      title={t("操作说明")}
                      aria-expanded={help}
                      aria-controls="table-help"
                      onClick={() => setHelp(!help)}
                    >
                      <CircleHelp size={17} />
                    </button>
                    <button
                      className="table-tool"
                      ref={exitButtonRef}
                      aria-label={t("退出牌桌")}
                      title={pending ? t("正在处理操作，请稍候") : t("退出牌桌")}
                      disabled={pending}
                      onClick={() => { setExitError(false); setExitConfirm(true); }}
                    >
                      <DoorOpen size={17} />
                    </button>
                  </div>

                  {table.spectator && (
                    <div className="spectator-controls" aria-label={t("观战控制")}>
                      <span className="spectator-badge"><Bot size={14} />{t("AI 观战")}</span>
                      <label>
                        <span className="sr-only">{t("观战视角")}</span>
                        <select aria-label={t("观战视角")} title={t("观战视角")} value={viewer}
                          disabled={sessionInvalid || !["playing", "round_result"].includes(table.state)}
                          onChange={(e) => onControl("view", Number(e.target.value))}>
                          {observerNames.map((name, seat) => <option key={seat} value={seat}>{t("{seat}视角", { seat: name })}</option>)}
                        </select>
                      </label>
                      <button className="table-tool" disabled={busy || table.state !== "playing"}
                        aria-label={table.paused ? t("继续播放") : t("暂停播放")}
                        title={table.paused ? t("继续播放") : t("暂停播放")}
                        onClick={() => onControl(table.paused ? "resume" : "pause")}>
                        {table.paused ? <Play size={17} /> : <Pause size={17} />}
                      </button>
                      <button className="table-tool" aria-label={t("单步")} title={table.paused ? t("单步") : t("暂停后可单步")} disabled={busy || !table.paused || table.state !== "playing"}
                        onClick={() => onControl("step")}><SkipForward size={17} /></button>
                      <select aria-label={t("播放速度")} title={t("播放速度")} value={table.playback_interval ?? 0.75} disabled={busy}
                        onChange={(e) => onControl("speed", Number(e.target.value))}>
                        <option value={1.5}>{t("慢速")}</option><option value={0.75}>{t("常速")}</option><option value={0.25}>{t("快速")}</option>
                      </select>
                      <button className="table-tool spectator-notes-button" aria-label={t("AI决策说明")} title={t("AI决策说明")} aria-expanded={notesOpen}
                        onClick={() => setNotesOpen(!notesOpen)}><MessageSquareText size={17} /><span>{t("决策说明")}</span></button>
                    </div>
                  )}

                  {help && (
                    <div id="table-help" className="table-help" role="region" aria-label={t("操作说明")}>
                      <div className="table-help-heading"><strong>{t("操作说明")}</strong><button className="text-button" onClick={() => { setHelp(false); helpButtonRef.current?.focus(); }}>{t("关闭")}</button></div>
                      <p>
                        {table.spectator
                          ? t("四席自动对战，可暂停、单步和调整播放速度。切换视角后查看该席手牌与已提交的决策说明。暂停会在当前决策完成后生效；每局结算后点击继续。")
                          : t("点击手牌直接打出；悬停或键盘聚焦可查看四家同种弃牌。立直先点「立直」再点舍牌。吃、碰、杠、荣和会显示来源玩家和目标牌；多个组合展开后点选即提交。真人没有倒计时。")}
                      </p>
                      <p>{t("刷新或退出后需重新开局。")}</p>
                      {table.cross_mode && (
                        <p>
                          {t("当前模型训练于半庄，本桌采用东风规则，实际棋力以对局体验为准。")}
                        </p>
                      )}
                      {table.timings && (
                        <p>
                          {t("加载")}{table.timings.load_seconds.toFixed(2)}{t("s · 最近决策")}{" "}
                          {table.timings.last_decision_seconds?.toFixed(2) ?? "—"}
                          {t("s · 内存峰值")}{table.timings.peak_rss_mib.toFixed(0)} MiB
                        </p>
                      )}
                    </div>
                  )}

                  {!introHidden && table.state === "playing" && (
                    <div className="round-intro" key={table.round_id} aria-hidden="true">
                      <span>RiichiLM</span>
                      <strong>{t("{wind}{round}局", { wind: winds[table.round_wind], round: ["", t("一"), t("二"), t("三"), t("四")][table.round_number] })}</strong>
                      <small>{table.honba} {t("本场")} · {t("好牌，值得等一等。")}</small>
                    </div>
                  )}

                  {banner && table.state === "playing" && (
                    <div className={`action-banner tone-${banner.tone}`} key={banner.id} aria-hidden="true">
                      <span className={`action-banner-portrait portrait-${banner.character}`} />
                      <div className="action-banner-copy">
                        <small>{labels[banner.seat]} · {t(characterNames[banner.character])}</small>
                        <strong>{banner.label}</strong>
                      </div>
                    </div>
                  )}

                  <div className="felt-rail" aria-hidden="true" />

                  {table.players!.filter((p) => p.seat !== viewer).map((p) => (
                    <Opponent
                      key={p.seat}
                      player={p}
                      table={table}
                      position={position(p.seat)}
                      portrait={cast.seats[p.seat]}
                      doraSet={doraSet}
                      viewerScore={own.score}
                      showScoreDiff={showScoreDiff}
                    />
                  ))}

                  <div
                    className={`table-center ${showScoreDiff ? "diff-active" : ""}`}
                    title={t("点击切换点差")}
                    onClick={() => setShowScoreDiff((v) => !v)}
                  >
                    <div
                      className="center-compass"
                      data-turn={table.thinking != null ? position(table.thinking) : !table.response_window && candidates.length ? 0 : undefined}
                      aria-hidden="true"
                    >
                      <span>{windAt(2)}</span>
                      <span>{windAt(3)}</span>
                      <span>{windAt(1)}</span>
                      <span>{windAt(0)}</span>
                    </div>
                    <div className="riichi-sticks-ring" aria-hidden="true">
                      {[2, 3, 1, 0].map((pos) =>
                        playerAt(pos)?.riichi ? <span key={pos} className={`riichi-stick pos-${pos}`} /> : null,
                      )}
                    </div>
                    <div className="round-label">
                      {t("{wind}{round}局", { wind: winds[table.round_wind], round: ["", t("一"), t("二"), t("三"), t("四")][table.round_number] })}
                    </div>
                    <div className="round-counters">
                      <span><i className="stick-icon honba-stick" aria-hidden="true" />{table.honba} {t("本场")}</span>
                      <span><i className="stick-icon kyotaku-stick" aria-hidden="true" />{table.kyotaku} {t("供托")}</span>
                    </div>
                    <div className="dora-label">{t("宝牌指示牌")}</div>
                    <div className="dora-tiles">
                      {table.dora_indicators.map((tile, i) => (
                        <Tile tile={tile} small key={i} />
                      ))}
                      {Array.from(
                        { length: Math.max(0, 5 - table.dora_indicators.length) },
                        (_, i) => (
                          <Tile small back key={`back${i}`} />
                        ),
                      )}
                    </div>
                    <div className={`remaining ${table.live_remaining <= 10 ? "low" : ""}`}>
                      {t("余牌")}<strong>{table.live_remaining}</strong>
                    </div>
                  </div>

                  {table.players!.map((p) => (
                    <River
                      key={p.seat}
                      player={p}
                      selectedTile={selectedTile}
                      trigger={trigger}
                      lastDiscardSeat={lastDiscardSeat}
                      position={position(p.seat)}
                      labels={labels}
                    />
                  ))}

                  <div className="self-seat">
                    <div className={`self-info ${own.riichi ? "in-riichi" : ""}`} data-seat={own.seat}>
                      <span className={`seat-wind ${own.dealer ? "is-dealer" : ""}`}>{winds[own.wind]}</span>
                      <span className={`seat-avatar portrait-${cast.seats[own.seat]}`} aria-hidden="true">
                        {table.spectator ? <Bot size={17} /> : <UserRound size={17} />}
                      </span>
                      <div className="self-seat-identity">
                        <strong>{labels[own.seat]}{table.spectator ? t(" · 观战视角") : ""}</strong>
                        <SeatPolicy table={table} seat={own.seat} />
                      </div>
                      <b>{own.score.toLocaleString(locale)}</b>
                      {showScoreDiff && <span className="diff-mode-tag">{t("点差")}</span>}
                      {own.dealer && <span className="dealer-label">{t("庄家")}</span>}
                      {own.riichi && <span className="riichi-badge">{t("立直")}</span>}
                      <Melds player={own} labels={labels} doraSet={doraSet} />
                    </div>
                    <div className="self-hand" ref={handRef}>
                      {(own.hand || []).map((tile, i) => (
                        <Tile
                          key={`${i}-${tile}`}
                          tile={tile}
                          dora={isDoraTile(tile, doraSet)}
                          onClick={table.spectator ? undefined : () => chooseTile(tile)}
                          onPreview={setPreview}
                          disabled={busy || !findDiscard(tile, false)}
                          dim={showIllegal && !findDiscard(tile, false)}
                        />
                      ))}
                      {own.drawn && (
                        <span className="drawn-tile">
                          <Tile
                            tile={own.drawn}
                            dora={isDoraTile(own.drawn, doraSet)}
                            onClick={table.spectator ? undefined : () => chooseTile(own.drawn!, true)}
                            onPreview={setPreview}
                            disabled={busy || !findDiscard(own.drawn, true)}
                            dim={showIllegal && !findDiscard(own.drawn, true)}
                          />
                        </span>
                      )}
                    </div>
                  </div>

                  <div className="table-controls" aria-label={t("牌桌操作")}>
                    {trigger && (
                      <div className="response-context" role="status">
                        <span className="source-direction">
                          {["", t("右侧"), t("对面"), t("左侧")][trigger.source]}
                        </span>
                        <strong>{names[trigger.source]}</strong>
                        <span>
                          {trigger.kind === "kan" ? t("杠牌 · 可抢杠") : t("打出")}
                        </span>
                        <Tile tile={trigger.tile} small />
                        <span className="response-hint">{t("请选择响应")}</span>
                      </div>
                    )}
                    {group && group !== "A_RIICHI_DISCARD" && (
                      <div
                        className="choice-panel"
                        aria-label={t("选择{action}的组合", { action: actionNames[group] })}
                      >
                        <span>{t("选择{action}的组合 · 点选即提交", { action: actionNames[group] })}</span>
                        <div className="choice-options">
                          {candidates
                            .filter((c) => c.kind === group)
                            .map((c) => {
                              const tiles = c.args.filter(
                                (a): a is string =>
                                  typeof a === "string" && a.startsWith("TILE_"),
                              );
                              const source = [
                                "SELF",
                                "SHIMOCHA",
                                "TOIMEN",
                                "KAMICHA",
                              ].indexOf(String(c.args[0]));
                              return (
                                <button
                                  key={c.id}
                                  className="choice"
                                  disabled={busy}
                                  onClick={() => void submit(c)}
                                >
                                  <strong>
                                    {actionNames[c.kind]}
                                    {source > 0 ? names[source] : ""}
                                  </strong>
                                  {tiles.map((a, i) => (
                                    <span
                                      key={i}
                                      className={
                                        source > 0 && i === 0
                                          ? "called-preview"
                                          : ""
                                      }
                                    >
                                      <Tile small tile={a.slice(5)} />
                                    </span>
                                  ))}
                                </button>
                              );
                            })}
                        </div>
                        <button
                          className="text-button"
                          onClick={() => setGroup(null)}
                        >
                          <X size={14} />
                          {t("取消选择")}
                        </button>
                      </div>
                    )}
                    <div className="action-dock">
                      {!trigger && (
                        <div className="turn-status" role="status">
                          <span
                            className={`turn-dot ${candidates.length && !busy ? "your-turn" : ""}`}
                          />
                          <span>
                            {group === "A_RIICHI_DISCARD"
                              ? t("请选择立直舍牌")
                              : table.state === "playing"
                                ? discards.length && !busy
                                  ? t("轮到你 · 点击手牌直接打出")
                                  : status
                                : t("本局已结算")}
                          </span>
                          {selectedTile && (
                            <span className="seen-pill">
                              {tileName(selectedTile, t, locale)} · {t("已见 {count}/4", { count: seenCount })}
                            </span>
                          )}
                        </div>
                      )}
                      <div className="action-buttons">
                        {special.map((kind) => {
                          const cs = candidates.filter((c) => c.kind === kind);
                          const source = [
                            "SELF",
                            "SHIMOCHA",
                            "TOIMEN",
                            "KAMICHA",
                          ].indexOf(String(cs[0].args[0]));
                          const tile =
                            source > 0
                              ? String(cs[0].args[1]).replace(/^TILE_/, "")
                              : null;
                          return (
                            <button
                              key={kind}
                              className={`button ${actionToneClass(kind)}`}
                              aria-label={source > 0 ? t("{action}{seat}的 {tile}", { action: actionNames[kind], seat: names[source], tile: tile || "" }) : actionNames[kind]}
                              aria-expanded={
                                cs.length > 1 || kind === "A_RIICHI_DISCARD"
                                  ? group === kind
                                  : undefined
                              }
                              disabled={busy}
                              onClick={() => {
                                if (
                                  cs.length === 1 &&
                                  kind !== "A_RIICHI_DISCARD"
                                )
                                  void submit(cs[0]);
                                else setGroup(group === kind ? null : kind);
                              }}
                            >
                              <span>
                                {actionNames[kind]}
                                {source > 0 && <small>{names[source]}</small>}
                              </span>
                            </button>
                          );
                        })}
                        {pass && (
                          <button
                            className="button light pass-button"
                            disabled={busy}
                            onClick={() => void submit(pass)}
                          >
                            {t("跳过")}
                          </button>
                        )}
                        {group === "A_RIICHI_DISCARD" && (
                          <button
                            className="button light pass-button"
                            disabled={busy}
                            onClick={() => setGroup(null)}
                          >
                            {t("取消立直")}
                          </button>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {table.spectator && notesOpen && (
                <section ref={notesRef} className="decision-panel" aria-label={t("AI决策说明")}>
                  <div className="decision-panel-heading">
                    <div><MessageSquareText size={19} /><h2>{labels[viewer]} {t(" · 决策说明")}</h2></div>
                    <span>{t("显示该席已完成的决策")}</span>
                  </div>
                  {!(table.decision_notes?.length) ? (
                    <p className="decision-empty">{t("等待该席完成决策。可切换视角查看其他 AI。")}</p>
                  ) : (
                    <div className="decision-notes">
                      {[...(table.decision_notes || [])].reverse().slice(0, 8).map((note) => (
                        <article className="decision-note" key={`${note.round_id}:${note.revision}:${note.seat}`}>
                          <div className="decision-note-action">
                            <strong>{actionNames[note.action.kind] || note.action.kind}</strong>
                            {note.action.args.filter((a): a is string => typeof a === "string" && a.startsWith("TILE_")).map((tile, i) => <Tile small tile={tile.slice(5)} key={i} />)}
                            <span>{note.sampling === "categorical" ? t("AI 出牌") : note.sampling === "uniform" ? t("随机出牌") : t("LLM 决策")}</span>
                            <small>{note.seconds.toFixed(2)}s · {note.candidate_count} {t("个可选项")}</small>
                          </div>
                          <p>{(note.sampling === "uniform" ? t("从 {count} 个可选动作中随机选择。", { count: note.candidate_count }) : note.summary) || (note.sampling === "categorical" ? t("本地模型直接给出推荐动作，不含文字解说。") : t("该模型未提供简短决策说明。"))}</p>
                          {note.selected_probability != null && (
                            <div className="prob-row">
                              <small>{t("AI 对该手的支持度 ")}{(note.selected_probability * 100).toFixed(1)}%</small>
                              <div className="prob-meter" aria-hidden="true">
                                <span style={{ width: `${Math.max(2, Math.min(100, note.selected_probability * 100))}%` }} />
                              </div>
                            </div>
                          )}
                        </article>
                      ))}
                    </div>
                  )}
                </section>
              )}
            </div>

            {(table.state === "round_result" || table.state === "finished") && !resultHidden && !sessionInvalid && (
              <div className="modal-backdrop">
                <section
                  className={`result-modal cinematic-result ${featuredCharacter != null ? "has-winner" : "draw-result"}`}
                  ref={exitConfirm ? undefined : modalRef}
                  role="dialog"
                  aria-modal="true"
                  aria-labelledby="result-heading"
                >
                  <div className="result-art" data-character={featuredCharacter} data-seat={featuredSeat}>
                    {featuredCharacter != null ? <>
                      <img key={featuredCharacter} src={characterVictoryArtwork(featuredCharacter)} alt={t("{name}的结算立绘", { name: t(characterNames[featuredCharacter]) })} />
                      <span>{t(characterNames[featuredCharacter])}<small>{labels[featuredSeat]}</small></span>
                    </> : <span>RiichiLM</span>}
                  </div>
                  <div className="result-content">
                    <div className="eyebrow">
                      {table.state === "finished" ? t("感谢对局") : t("回顾这一局")}
                    </div>
                    <h2 id="result-heading">
                      {table.final
                        ? t("本场终了")
                        : table.result?.kind === "win"
                          ? table.result.winners
                              .map((w) => (w.tsumo ? t("自摸和牌") : t("荣和")))
                              .join(" · ")
                          : table.result?.kind === "abort"
                            ? t("途中流局")
                            : table.result?.kind === "nagashi"
                              ? t("流局满贯")
                              : t("荒牌流局")}
                    </h2>
                    <p className="result-subtitle">
                      {table.mode === "single"
                        ? t("单局练习已结束")
                        : t("{wind}{round}局 · {honba}本场", { wind: winds[table.round_wind], round: ["", t("一"), t("二"), t("三"), t("四")][table.round_number], honba: table.honba })}
                      {table.final ? t(" · 最终点数与排名") : t(" · 本局结算")}
                    </p>
                    {resultSeats.length > 1 && <div className="result-character-tabs" aria-label={t("和牌角色")}>
                      {resultSeats.map((seat) => <button key={seat} type="button" aria-pressed={seat === featuredSeat}
                        onClick={() => setResultSeat(seat)}>{labels[seat]} · {t(characterNames[cast.seats[seat]])}</button>)}
                    </div>}
                    {table.result?.winners.map((w) => {
                      const tier = handTierLabel(w, t);
                      return (
                        <div className="winner" key={w.actor}>
                          <div className="winner-title">
                            <strong>
                              {labels[w.actor]}
                              {w.tsumo ? t(" 自摸") : t(" 荣和 · {seat}放铳", { seat: labels[w.target] })}
                            </strong>
                            <b>
                              {w.yakuman_units
                                ? t("{count} 倍役满", { count: w.yakuman_units })
                                : t("{han} 番 {fu} 符", { han: w.han ?? "—", fu: w.fu ?? "—" })}
                              {tier && <span className="tier-badge">{tier}</span>}
                            </b>
                          </div>
                          <div className="winner-hand">
                            {w.hand.map((tile, i) => (
                              <Tile small tile={tile} dora={isDoraTile(tile, doraSet)} key={i} />
                            ))}
                            <span className="drawn-tile">
                              <Tile small tile={w.winning_tile} dora={isDoraTile(w.winning_tile, doraSet)} />
                            </span>
                            {w.melds.map((m, i) => (
                              <span className="meld" key={i}>
                                {m.tiles.map((tile, j) => (
                                  <Tile small tile={tile} dora={isDoraTile(tile, doraSet)} key={j} />
                                ))}
                              </span>
                            ))}
                          </div>
                          <div className="yaku-tags">
                            {w.yaku.map((y) => (
                              <span key={y}>{yakuNames[y] || y}</span>
                            ))}
                            {!w.yakuman_units &&
                              [
                                [t("宝牌"), w.dora],
                                [t("赤宝牌"), w.aka],
                                [t("里宝牌"), w.ura],
                              ]
                                .filter(([, v]) => Number(v) > 0)
                                .map(([n, v]) => (
                                  <span key={n}>
                                    {n} × {v}
                                  </span>
                                ))}
                          </div>
                          {w.ura_indicators.length > 0 && (
                            <div className="ura">
                              <small>{t("里宝牌指示牌")}</small>
                              {w.ura_indicators.map((tile, i) => (
                                <Tile small tile={tile} key={i} />
                              ))}
                            </div>
                          )}
                        </div>
                      );
                    })}
                    <div className="score-results">
                      {table.players!.map((p) => (
                        <div key={p.seat} className={p.seat === 0 ? "you" : ""}>
                          <span>
                            {table.final && (
                              <b className="rank">{table.final.ranks[p.seat]}</b>
                            )}
                            {labels[p.seat]}
                          </span>
                          <strong>
                            {(table.final?.scores[p.seat] ?? p.score).toLocaleString(locale)}
                          </strong>
                          <span
                            className={
                              (table.result?.score_delta[p.seat] ?? 0) >= 0
                                ? "positive"
                                : "negative"
                            }
                          >
                            {(table.result?.score_delta[p.seat] ?? 0) > 0 ? "+" : ""}
                            {table.result?.score_delta[p.seat] ?? 0}
                          </span>
                        </div>
                      ))}
                    </div>
                    {table.mode === "single" && (
                      <p className="result-footnote">
                        {t("未领取供托：{count} 根（{points} 点），保留在池中，不额外分配。", { count: table.kyotaku, points: table.kyotaku * 1000 })}
                      </p>
                    )}
                    <div className="result-actions">
                      <span>{t("结果仅展示于当前页面")}</span>
                      <button className="button light" onClick={() => setResultHidden(true)}>{t("查看牌桌")}</button>
                      <button
                        className="button primary"
                        disabled={busy}
                        onClick={table.state === "finished" ? onExit : onContinue}
                      >
                        {table.state === "finished"
                          ? t("返回配置")
                          : table.result?.game_over
                            ? t("查看终场")
                            : t("继续下一局")}
                        <ArrowRight size={17} />
                      </button>
                    </div>
                  </div>
                </section>
              </div>
            )}
          </>
        )
      )}
      {exitConfirm && !sessionInvalid && (
        <div className="modal-backdrop">
          <section
            className="confirm-modal"
            ref={modalRef}
            role="dialog"
            aria-modal="true"
            aria-labelledby="exit-heading"
          >
            <DoorOpen size={28} />
            <h2 id="exit-heading">{t("要离开这张牌桌吗？")}</h2>
            <p>{t("当前对局不会保存，再次开局将从头开始。")}</p>
            {exitError && <p role="alert">{t("退出未确认，请重试。")}</p>}
            <div>
              <button
                className="button light"
                autoFocus
                disabled={pending}
                onClick={() => setExitConfirm(false)}
              >
                {t("继续对局")}
              </button>
              <button
                className="button primary"
                disabled={pending}
                onClick={async () => {
                  setExitError(false);
                  if (!(await onExit())) setExitError(true);
                }}
              >
                {t("退出牌桌")}
              </button>
            </div>
          </section>
        </div>
      )}
    </>
  );
}
