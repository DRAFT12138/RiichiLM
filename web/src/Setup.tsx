import { useI18n } from "./i18n";
import {
  ArrowLeft,
  ArrowRight,
  Bot,
  Check,
  ChevronDown,
  Cpu,
  Layers3,
  Play,
  RefreshCw,
  Search,
  Shuffle,
  UserRound,
  Wind,
} from "lucide-react";
import { useMemo, useState, type Dispatch, type SetStateAction } from "react";
import type { Catalog, Mode, SetupConfig } from "./types";
import { useLabels } from "./labels";
import type { Cast } from "./cast";
import { characterNames } from "./cast";

function useCatalogOptions(catalog: Catalog) {
  const { t, locale } = useI18n();
  const checkpoints = useMemo(
    () =>
      catalog.runs.flatMap((run) =>
        run.checkpoints.map((cp) => ({
          ...cp,
          experiment: run.experiment,
          run_id: run.run_id,
          custom: run.training_type === "Weights",
        })),
      ),
    [catalog],
  );
  const selectable = useMemo(() => checkpoints.filter((c) => c.selectable), [checkpoints]);
  const checkpointLabels = useMemo(() => {
    const base = (cp: (typeof checkpoints)[number]) =>
      cp.custom
        ? `${t("自定义权重")} · ${cp.checkpoint_id}`
        : `${cp.experiment} · ${t("{step} 步", { step: cp.step.toLocaleString(locale) })} · ${cp.run_id} · ${cp.attempt_id} · ${cp.checkpoint_id}`;
    const counts = new Map<string, number>();
    checkpoints.forEach((cp) => counts.set(base(cp), (counts.get(base(cp)) || 0) + 1));
    return new Map(
      checkpoints.map((cp) => {
        const label = base(cp);
        return [
          cp.id,
          `${label}${counts.get(label)! > 1 ? ` · ${cp.id}` : ""}${!cp.selectable ? ` · ${t(cp.status)}` : ""}`,
        ];
      }),
    );
  }, [checkpoints, t, locale]);

  return { checkpoints, selectable, checkpointLabels };
}

export function Setup({
  catalog,
  refresh,
  start,
  pending,
  ready,
  cast,
  shuffleCast,
  config,
  setConfig,
  createUncertain,
  onOpenModels,
}: {
  catalog: Catalog;
  refresh: () => void;
  start: (mode: Mode, opponents: string[], spectator: boolean, temperature: number, topP: number) => void;
  pending: boolean;
  ready: boolean;
  cast: Cast;
  shuffleCast: () => void;
  config: SetupConfig;
  setConfig: Dispatch<SetStateAction<SetupConfig>>;
  createUncertain: boolean;
  onOpenModels: () => void;
}) {
  const { t, locale } = useI18n();
  const { modes, names, observerNames } = useLabels();
  const { mode, opponents, spectator, primary, temperature, top_p } = config;
  const configLocked = pending || createUncertain;
  const { checkpoints, selectable, checkpointLabels } = useCatalogOptions(catalog);

  const seats = spectator ? [primary, ...opponents] : opponents;
  const seatLabels = spectator ? observerNames : names.slice(1);
  const seatWinds = spectator ? [t("东"), t("南"), t("西"), t("北")] : [t("南"), t("西"), t("北")];
  const invalidSeat = seats.findIndex(
    (id) =>
      id !== "random" &&
      !selectable.some((c) => c.id === id) &&
      !catalog.llms?.some((c) => c.id === id),
  );

  function setSeat(index: number, value: string) {
    setConfig((old) =>
      old.spectator && index === 0
        ? { ...old, primary: value }
        : {
            ...old,
            opponents: old.opponents.map((o, j) =>
              j === index - (old.spectator ? 1 : 0) ? value : o,
            ),
          },
    );
  }

  function assignAllSeats(value: string) {
    setConfig((old) => ({
      ...old,
      primary: old.spectator ? value : old.primary,
      opponents: old.opponents.map(() => value),
    }));
  }

  const allSamePolicy = seats.every((s) => s === seats[0]) ? seats[0] : "";

  function renderPolicySelect(index: number, value: string) {
    return (
      <div className="select-wrap">
        <select
          id={`opponent-${index}`}
          value={value}
          disabled={configLocked}
          title={
            checkpointLabels.get(value) ||
            (value === "random"
              ? t("纯随机策略")
              : catalog.llms?.find((llm) => llm.id === value)?.name ||
                t("所选策略不可用"))
          }
          onChange={(e) => setSeat(index, e.target.value)}
        >
          {value !== "random" &&
            !selectable.some((c) => c.id === value) &&
            !catalog.llms?.some((c) => c.id === value) && (
              <option value={value} disabled>
                {t("已不可用，请重新选择")}
              </option>
            )}
          <option value="random">{t("纯随机策略")}</option>
          {!!catalog.llms?.length && (
            <optgroup label={t("大语言模型 (LLM)")}>
              {catalog.llms.map((llm) => (
                <option key={llm.id} value={llm.id}>
                  LLM · {llm.name} · {llm.model}
                </option>
              ))}
            </optgroup>
          )}
          {!!checkpoints.length && (
            <optgroup label={t("训练模型")}>
              {checkpoints.map((cp) => (
                <option key={cp.id} value={cp.id} disabled={!cp.selectable}>
                  {checkpointLabels.get(cp.id)}
                </option>
              ))}
            </optgroup>
          )}
        </select>
        <ChevronDown size={14} />
      </div>
    );
  }

  function renderSeatCard(index: number, placement: "top" | "left" | "right" | "bottom") {
    const value = seats[index];
    const castIndex = spectator ? index : index + 1;
    const relHint = spectator ? t("AI席位") : [t("右侧"), t("对面"), t("左侧")][index];
    return (
      <section
        className={`opponent-card seat-slot-${placement}`}
        data-seat={castIndex}
        key={index}
      >
        <div className="opponent-top">
          <span className={`bot-icon portrait-${cast.seats[castIndex]}`} aria-hidden="true">
            <Bot size={18} />
          </span>
          <div className="opponent-identity">
            <strong>{seatLabels[index]}</strong>
            <small>
              {t(characterNames[cast.seats[castIndex]])} · {relHint}
            </small>
          </div>
          <span className="seat-marker">{seatWinds[index]}</span>
        </div>
        <label className="sr-only" htmlFor={`opponent-${index}`}>
          {t("{seat}策略", { seat: seatLabels[index] })}
        </label>
        {renderPolicySelect(index, value)}
        <p className="opponent-note">
          {value === "random"
            ? t("随机出牌（从当前可选动作中随机挑选）")
            : value.startsWith("llm:")
              ? t("大语言模型 · {name}", {
                  name: catalog.llms?.find((c) => c.id === value)?.name || t("请刷新重选"),
                })
              : t("本地 AI 模型 · 训练 {step} 步", {
                  step:
                    checkpoints.find((c) => c.id === value)?.step.toLocaleString(locale) ||
                    t("请刷新重选"),
                })}
        </p>
      </section>
    );
  }

  return (
    <div className="play-lobby">
      <section className="lobby-stage">
        <div className="lobby-toolbar">
          <div className="play-mode-switch" aria-label={t("参与方式")}>
            <button
              type="button"
              aria-pressed={!spectator}
              disabled={configLocked}
              onClick={() => setConfig((old) => ({ ...old, spectator: false }))}
            >
              {t("真人对战")}
            </button>
            <button
              type="button"
              aria-pressed={spectator}
              disabled={configLocked}
              onClick={() => setConfig((old) => ({ ...old, spectator: true }))}
            >
              {t("纯 AI 观战")}
            </button>
          </div>

          <div className="mode-grid" role="group" aria-label={t("选择你的牌局")}>
            {(Object.entries(modes) as [Mode, (typeof modes)[Mode]][]).map(
              ([key, value], i) => (
                <button
                  key={key}
                  className={`mode-card ${mode === key ? "chosen" : ""}`}
                  onClick={() => setConfig((old) => ({ ...old, mode: key }))}
                  disabled={configLocked}
                  aria-pressed={mode === key}
                >
                  <span className="mode-icon">
                    {i === 0 ? (
                      <Layers3 size={15} />
                    ) : i === 1 ? (
                      <Wind size={15} />
                    ) : (
                      <Play size={15} />
                    )}
                  </span>
                  <span className="mode-title">{value.name}</span>
                  <small className="mode-tag">{value.tag}</small>
                  <span className="radio-check">
                    {mode === key && <Check size={10} />}
                  </span>
                </button>
              ),
            )}
          </div>

          <button
            type="button"
            className="shuffle-cast"
            disabled={configLocked}
            onClick={shuffleCast}
          >
            <Shuffle size={13} /> {t("换一批角色")}
          </button>
        </div>

        <div className={`lobby-table opponent-grid ${spectator ? "spectator-grid" : ""}`}>
          {spectator ? (
            <>
              {renderSeatCard(0, "bottom")}
              {renderSeatCard(1, "right")}
              {renderSeatCard(2, "top")}
              {renderSeatCard(3, "left")}
            </>
          ) : (
            <>
              {renderSeatCard(0, "right")}
              {renderSeatCard(1, "top")}
              {renderSeatCard(2, "left")}
              <section className="human-seat-card seat-slot-bottom" data-seat={0}>
                <div className="opponent-top">
                  <span className={`bot-icon portrait-${cast.seats[0]}`} aria-hidden="true">
                    <UserRound size={18} />
                  </span>
                  <div className="opponent-identity">
                    <strong>{names[0]}</strong>
                    <small>
                      {t(characterNames[cast.seats[0]])} · {t("真人")}
                    </small>
                  </div>
                  <span className="seat-marker is-dealer">{t("东")}</span>
                </div>
                <p className="human-seat-note">
                  {t("真人起始东家 · 对局仅保留在当前页面，退出后重新开始")}
                </p>
              </section>
            </>
          )}

          <div className="lobby-table-center">
            <div className="lobby-guide">
              <span className={`welcome-character portrait-${cast.host}`} aria-hidden="true" />
              <div>
                <span className="lobby-guide-name">
                  {t(characterNames[cast.host])} · {t("牌馆向导")}
                </span>
                <strong>
                  {spectator ? t("四家就位，静候开局。") : t("东家，等你入席。")}
                </strong>
                <small>
                  {t("四人立直麻将")} · {t("本地运行")}
                </small>
              </div>
            </div>
          </div>
        </div>
      </section>

      <aside className="launch-panel">
        <div className="launch-head">
          <h2>{t("选择你的牌局")}</h2>
          <p>{modes[mode].desc}</p>
        </div>

        <div className="launch-summary" aria-label={t("牌局配置摘要")}>
          <div>
            <span>{t("赛制")}</span>
            <strong>{modes[mode].name}</strong>
          </div>
          <div>
            <span>{t("参与方式")}</span>
            <strong>{spectator ? t("纯 AI 观战") : t("真人对战")}</strong>
          </div>
          <div>
            <span>{t("AI 席位")}</span>
            <strong>
              {seats.length} {t("席")}
            </strong>
          </div>
        </div>

        <div className="sampling-box" aria-label={t("模型推理采样")}>
          <div className="sampling-head">
            <span>{t("模型推理采样")}</span>
            {(temperature !== 0.7 || top_p !== 0.9) && (
              <button
                type="button"
                className="text-button sampling-reset"
                disabled={configLocked}
                onClick={() => setConfig((old) => ({ ...old, temperature: 0.7, top_p: 0.9 }))}
              >
                {t("恢复默认")}
              </button>
            )}
          </div>
          <div className="sampling-row">
            <label htmlFor="sampling-temperature">{t("温度 (Temperature)")}</label>
            <div className="sampling-controls">
              <input
                id="sampling-temperature-range"
                type="range"
                min={0}
                max={2}
                step={0.05}
                value={temperature}
                disabled={configLocked}
                aria-label={t("温度滑块")}
                onChange={(e) => {
                  const v = Number(e.target.value);
                  if (Number.isFinite(v)) setConfig((old) => ({ ...old, temperature: Math.max(0, Math.min(2, v)) }));
                }}
              />
              <input
                id="sampling-temperature"
                type="number"
                min={0}
                max={2}
                step={0.05}
                value={temperature}
                disabled={configLocked}
                onChange={(e) => {
                  const v = Number(e.target.value);
                  if (Number.isFinite(v)) setConfig((old) => ({ ...old, temperature: Math.max(0, Math.min(2, v)) }));
                }}
              />
            </div>
          </div>
          <div className="sampling-row">
            <label htmlFor="sampling-top-p">Top-P</label>
            <div className="sampling-controls">
              <input
                id="sampling-top-p-range"
                type="range"
                min={0.05}
                max={1}
                step={0.05}
                value={top_p}
                disabled={configLocked}
                aria-label={t("Top-P 滑块")}
                onChange={(e) => {
                  const v = Number(e.target.value);
                  if (Number.isFinite(v)) setConfig((old) => ({ ...old, top_p: Math.max(0.05, Math.min(1, v)) }));
                }}
              />
              <input
                id="sampling-top-p"
                type="number"
                min={0.05}
                max={1}
                step={0.05}
                value={top_p}
                disabled={configLocked}
                onChange={(e) => {
                  const v = Number(e.target.value);
                  if (Number.isFinite(v)) setConfig((old) => ({ ...old, top_p: Math.max(0.05, Math.min(1, v)) }));
                }}
              />
            </div>
          </div>
          <small className="sampling-hint">{t("作用于本地模型 · 0 为贪心，推荐 0.7 / 0.9")}</small>
        </div>

        {(selectable.length > 0 || (catalog.llms?.length ?? 0) > 0) && (
          <div className="quick-fill-box">
            <label htmlFor="quick-fill-all">{t("一键全席")}</label>
            <div className="select-wrap">
              <select
                id="quick-fill-all"
                value={allSamePolicy}
                disabled={configLocked}
                onChange={(e) => {
                  if (e.target.value) assignAllSeats(e.target.value);
                }}
              >
                <option value="" disabled>
                  {t("全席设为该模型")}…
                </option>
                <option value="random">{t("纯随机策略")}</option>
                {catalog.llms?.map((llm) => (
                  <option key={llm.id} value={llm.id}>
                    LLM · {llm.name}
                  </option>
                ))}
                {selectable.map((cp) => (
                  <option key={cp.id} value={cp.id}>
                    {checkpointLabels.get(cp.id)}
                  </option>
                ))}
              </select>
              <ChevronDown size={14} />
            </div>
          </div>
        )}

        <div className="launch-links">
          <button type="button" className="button light" onClick={onOpenModels}>
            <Cpu size={14} />
            {t("本地模型库")} ({selectable.length})
          </button>
          <button
            type="button"
            className="button light launch-refresh"
            aria-label={t("刷新列表")}
            title={t("刷新列表")}
            disabled={configLocked || !ready}
            onClick={refresh}
          >
            <RefreshCw size={14} />
          </button>
        </div>

        {ready && invalidSeat !== -1 && (
          <p className="launch-error" role="status">
            {t("{seat}所选策略已不可用，请重新选择。", {
              seat: seatLabels[invalidSeat],
            })}
          </p>
        )}

        <div className="launch-cta">
          <button
            className="button primary start-button"
            disabled={!ready || pending || invalidSeat !== -1}
            onClick={() => start(mode, seats, spectator, temperature, top_p)}
          >
            {pending
              ? t("正在处理…")
              : !ready
                ? t("连接牌馆中…")
                : createUncertain
                  ? t("重试确认开局")
                  : spectator
                    ? t("开始观战")
                    : t("开始对局")}
            <ArrowRight size={17} />
          </button>
          <small className="launch-footnote">
            {t("仅本页内存运行 · 不保存对局历史")} ·{" "}
            <a href="https://www.riichilm.top" target="_blank" rel="noopener noreferrer">
              www.riichilm.top
            </a>
          </small>
        </div>
      </aside>
    </div>
  );
}

export function ModelLibrary({
  catalog,
  ready,
  pending,
  refresh,
  config,
  setConfig,
  createUncertain,
  onBack,
}: {
  catalog: Catalog;
  ready: boolean;
  pending: boolean;
  refresh: () => void;
  config: SetupConfig;
  setConfig: Dispatch<SetStateAction<SetupConfig>>;
  createUncertain: boolean;
  onBack: () => void;
}) {
  const { t, locale } = useI18n();
  const { names, observerNames } = useLabels();
  const { opponents, spectator, primary } = config;
  const configLocked = pending || createUncertain;
  const [query, setQuery] = useState(""),
    [showRuns, setShowRuns] = useState(false);

  const { selectable } = useCatalogOptions(catalog);
  const filtered = catalog.runs.filter((r) =>
    `${r.experiment} ${r.run_id} ${r.checkpoints.map((c) => c.checkpoint_id).join(" ")}`
      .toLowerCase()
      .includes(query.toLowerCase()),
  );

  const seats = spectator ? [primary, ...opponents] : opponents;
  const seatLabels = spectator ? observerNames : names.slice(1);

  function setSeat(index: number, value: string) {
    setConfig((old) =>
      old.spectator && index === 0
        ? { ...old, primary: value }
        : {
            ...old,
            opponents: old.opponents.map((o, j) =>
              j === index - (old.spectator ? 1 : 0) ? value : o,
            ),
          },
    );
  }

  function assignAllSeats(value: string) {
    setConfig((old) => ({
      ...old,
      primary: old.spectator ? value : old.primary,
      opponents: old.opponents.map(() => value),
    }));
  }

  return (
    <section className="library-panel">
      <div className="section-heading">
        <div>
          <button type="button" className="button light" onClick={onBack}>
            <ArrowLeft size={14} />
            {t("返回对局配置")}
          </button>
          <span className="library-icon">
            <Cpu size={18} />
          </span>
          <h2>{t("本地模型库")}</h2>
          <span className="count-pill">{selectable.length}</span>
        </div>
        <button className="button light" onClick={refresh} disabled={configLocked || !ready}>
          <RefreshCw size={14} />
          {pending ? t("请稍候…") : t("刷新列表")}
        </button>
      </div>

      {!catalog.runs.length && !catalog.errors.length ? (
        <div className="empty-library">
          <span className="empty-icon">
            <Bot size={22} />
          </span>
          <div>
            <strong>{t("下一位对手，正在成长。")}</strong>
            <p>{t("模型目录暂无训练记录。现在就可以与三位随机对手开一桌。")}</p>
          </div>
          <span className="outline-tag">{t("随机策略可用")}</span>
        </div>
      ) : catalog.runs.length ? (
        <>
          <div className="search">
            <Search size={16} />
            <input
              aria-label={t("搜索模型和训练记录")}
              placeholder={t("搜索实验名、Run 或 checkpoint ID")}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
            <button
              className="text-button"
              aria-expanded={showRuns}
              onClick={() => setShowRuns(!showRuns)}
            >
              {showRuns ? t("收起训练记录") : t("展开训练记录")}
              <ChevronDown size={14} />
            </button>
          </div>
          {!filtered.length && (
            <p className="search-empty" role="status">
              {t("没有找到匹配的训练记录，请换个关键词。")}
            </p>
          )}
          {filtered.map((run) => (
            <div className="run-card" key={run.run_id}>
              <div className="run-heading">
                <strong>{run.experiment}</strong>
                <span>{run.training_type === "Weights" ? t("自定义权重") : run.training_type}</span>
                {run.training_type !== "Weights" && <code title={run.created_at || ""}>{run.run_id}</code>}
              </div>
              <div className="checkpoint-list">
                {run.checkpoints.length ? (
                  run.checkpoints.map((cp) => (
                    <div key={cp.id} className="checkpoint">
                      <div>
                        {run.training_type !== "Weights" && (
                          <b>
                            {cp.step.toLocaleString(locale)} <small>{t("步")}</small>
                          </b>
                        )}
                        <code>{cp.checkpoint_id}</code>
                        {run.training_type !== "Weights" && (
                          <small title={cp.attempt_id}>Attempt {cp.attempt_id.slice(0, 8)}</small>
                        )}
                        <small>{t(cp.status)}</small>
                      </div>
                      <div className="assign-buttons">
                        {seatLabels.map((n, i) => (
                          <button
                            className="button subtle"
                            aria-pressed={seats[i] === cp.id}
                            key={n}
                            disabled={!cp.selectable || configLocked}
                            onClick={() => setSeat(i, cp.id)}
                          >
                            {t("选为")}
                            {n}
                          </button>
                        ))}
                        <button
                          className="button subtle assign-all"
                          aria-pressed={seats.every((s) => s === cp.id)}
                          title={t("全席设为该模型")}
                          disabled={!cp.selectable || configLocked}
                          onClick={() => assignAllSeats(cp.id)}
                        >
                          {t("全席")}
                        </button>
                      </div>
                    </div>
                  ))
                ) : (
                  <p className="muted">{t("此训练任务没有剩余 checkpoint。")}</p>
                )}
              </div>
              {showRuns &&
                run.attempts.map((a) => (
                  <div className="attempt" key={a.attempt_id}>
                    <code>{a.attempt_id}</code>
                    <span>{a.outcome === "unknown" ? t("状态未知") : t(a.outcome)}</span>
                    <span>
                      {t("最近记录")}
                      {a.progress?.global_step ?? "—"} {t("步")}
                    </span>
                    {a.source_checkpoint && (
                      <small title={a.source_checkpoint?.run_id}>
                        {t("恢复自")}
                        {a.source_checkpoint?.checkpoint_id} · {a.source_checkpoint?.global_step}{" "}
                        {t("步")}
                      </small>
                    )}
                  </div>
                ))}
              {run.errors.map((e, i) => (
                <p className="inline-error" key={i}>
                  {t(e)}
                </p>
              ))}
            </div>
          ))}
        </>
      ) : null}
      {catalog.errors.length > 0 &&
        catalog.errors.map((e, i) => (
          <p className="inline-error" role="alert" key={i}>
            {t(e)}
          </p>
        ))}
    </section>
  );
}
