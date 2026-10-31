import { useI18n } from "./i18n";
import {
  ArrowRight,
  Bot,
  Check,
  ChevronDown,
  Cpu,
  Layers3,
  RefreshCw,
  Search,
  Shuffle,
  Sparkles,
  Wind,
} from "lucide-react";
import { useMemo, useState, type Dispatch, type SetStateAction } from "react";
import type { Catalog, Mode, SetupConfig } from "./types";
import { useLabels } from "./labels";
import type { Cast } from "./cast";
import { characterNames } from "./cast";
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
}: {
  catalog: Catalog;
  refresh: () => void;
  start: (mode: Mode, opponents: string[], spectator: boolean) => void;
  pending: boolean;
  ready: boolean;
  cast: Cast;
  shuffleCast: () => void;
  config: SetupConfig;
  setConfig: Dispatch<SetStateAction<SetupConfig>>;
  createUncertain: boolean;
}) {
  const { t, locale } = useI18n();
  const { modes, names, observerNames } = useLabels();
  const { mode, opponents, spectator, primary } = config;
  const configLocked = pending || createUncertain;
  const [query, setQuery] = useState(""),
    [showRuns, setShowRuns] = useState(false);
  const checkpoints = useMemo(
    () =>
      catalog.runs.flatMap((run) =>
        run.checkpoints.map((cp) => ({
          ...cp,
          experiment: run.experiment,
          run_id: run.run_id,
        })),
      ),
    [catalog],
  );
  const filtered = catalog.runs.filter((r) =>
    `${r.experiment} ${r.run_id} ${r.checkpoints.map((c) => c.checkpoint_id).join(" ")}`
      .toLowerCase()
      .includes(query.toLowerCase()),
  );
  const selectable = checkpoints.filter((c) => c.selectable);
  const checkpointLabels = useMemo(() => {
    const base = (cp: (typeof checkpoints)[number]) => `${cp.experiment} · ${t("{step} 步", { step: cp.step.toLocaleString(locale) })} · ${cp.run_id} · ${cp.attempt_id} · ${cp.checkpoint_id}`;
    const counts = new Map<string, number>();
    checkpoints.forEach((cp) => counts.set(base(cp), (counts.get(base(cp)) || 0) + 1));
    return new Map(checkpoints.map((cp) => {
      const label = base(cp);
      return [cp.id, `${label}${counts.get(label)! > 1 ? ` · ${cp.id}` : ""}${!cp.selectable ? ` · ${t(cp.status)}` : ""}`];
    }));
  }, [checkpoints, t, locale]);
  const seats = spectator ? [primary, ...opponents] : opponents;
  const seatLabels = spectator ? observerNames : names.slice(1);
  const invalidSeat = seats.findIndex((id) => id !== "random" &&
    !selectable.some((c) => c.id === id) && !catalog.llms?.some((c) => c.id === id));
  function setSeat(index: number, value: string) {
    setConfig((old) => old.spectator && index === 0
      ? { ...old, primary: value }
      : { ...old, opponents: old.opponents.map((o, j) => j === index - (old.spectator ? 1 : 0) ? value : o) });
  }
  return (
    <>
      <section className="welcome">
        <div className="welcome-copy">
          <span className="welcome-kicker"><span /> {t("RIICHILM · 学习与对局")}</span>
          <h1>{t("让机器学打麻将。")}</h1>
          <p>
            {t("大语言模型 · 监督学习 · 强化学习")}<br />
            {t("带上自训模型或外接 LLM，对局或观战。")}</p>
          <div className="welcome-tags">
            <span>
              <span className="status-dot" />
              {t("本地运行")}</span>
            <span>{t("四人立直麻将")}</span>
            <span>{t("仅供学习与研究")}</span>
          </div>
          <p className="welcome-scope">{t("请自备训练数据与模型；项目不提供数据集或预训练权重。")}</p>
        </div>
        <div className="welcome-art" aria-hidden="true">
          <span className="welcome-art-ring" />
          <span className={`welcome-character portrait-${cast.host}`} />
          <span className="welcome-art-label">{t(characterNames[cast.host])} <b>✦</b> {t("牌馆向导")}</span>
        </div>
      </section>
      <div className="setup-layout">
      <div className="setup-options">
      <div className="section-heading">
        <div>
          <span className="section-ornament">{t("一")}</span>
          <h2>{t("选择你的牌局")}</h2>
        </div>
        <span>{t("按自己的节奏，开始一场对局")}</span>
      </div>
      <div className="mode-grid">
        {(Object.entries(modes) as [Mode, (typeof modes)[Mode]][]).map(
          ([key, value], i) => (
            <button
              key={key}
              className={`mode-card ${mode === key ? "chosen" : ""}`}
              onClick={() => setConfig((old) => ({ ...old, mode: key }))}
              disabled={configLocked}
              aria-pressed={mode === key}
            >
              <div className="mode-card-top">
                <span className="mode-icon">
                  {i === 0 ? (
                    <Layers3 size={22} />
                  ) : i === 1 ? (
                    <Wind size={24} />
                  ) : (
                    <Sparkles size={22} />
                  )}
                </span>
                <span className="radio-check">
                  {mode === key && <Check size={13} />}
                </span>
              </div>
              <div className="mode-title">
                {value.name}
                <small>{value.tag}</small>
              </div>
              <p>{value.desc}</p>
            </button>
          ),
        )}
      </div>
      <div className="section-heading opponent-heading">
        <div>
          <span className="section-ornament">{t("二")}</span>
          <h2>{t("落座席位")}</h2>
        </div>
        <button type="button" className="shuffle-cast" disabled={configLocked} onClick={shuffleCast}>
          <Shuffle size={14} /> {t("换一批角色")}</button>
      </div>
      <div className="play-mode-switch" aria-label={t("参与方式")}>
        <button type="button" aria-pressed={!spectator} disabled={configLocked} onClick={() => setConfig((old) => ({ ...old, spectator: false }))}>{t("真人对战")}</button>
        <button type="button" aria-pressed={spectator} disabled={configLocked} onClick={() => setConfig((old) => ({ ...old, spectator: true }))}>{t("纯 AI 观战")}</button>
      </div>
      <div className={`opponent-grid ${spectator ? "spectator-grid" : ""}`}>
        {seats.map((value, i) => (
          <section className="opponent-card" data-seat={spectator ? i : i + 1} key={i}>
            <div className="opponent-top">
              <span className={`bot-icon portrait-${cast.seats[spectator ? i : i + 1]}`} aria-hidden="true">
                <Bot size={23} />
              </span>
              <div>
                <strong>{seatLabels[i]}</strong>
                <small>{t(characterNames[cast.seats[spectator ? i : i + 1]])} · {spectator ? t("AI席位") : [t("右侧"), t("对面"), t("左侧")][i]}</small>
              </div>
              <span className="seat-marker">{(spectator ? [t("东"), t("南"), t("西"), t("北")] : [t("南"), t("西"), t("北")])[i]}</span>
            </div>
            <label htmlFor={`opponent-${i}`}>{t("{seat}策略", { seat: seatLabels[i] })}</label>
            <div className="select-wrap">
              <Shuffle size={15} />
              <select
                id={`opponent-${i}`}
                value={value}
                disabled={configLocked}
                title={checkpointLabels.get(value) || (value === "random" ? t("纯随机策略") : catalog.llms?.find((llm) => llm.id === value)?.name || t("所选策略不可用"))}
                onChange={(e) =>
                  setSeat(i, e.target.value)
                }
              >
                {value !== "random" && !selectable.some((c) => c.id === value) && !catalog.llms?.some((c) => c.id === value) && (
                  <option value={value} disabled>{t("已不可用，请重新选择")}</option>
                )}
                <option value="random">{t("纯随机策略")}</option>
                {!!catalog.llms?.length && <optgroup label={t("通用 LLM")}>{catalog.llms.map((llm) => (
                  <option key={llm.id} value={llm.id}>
                    LLM · {llm.name} · {llm.model}
                  </option>
                ))}</optgroup>}
                {!!checkpoints.length && <optgroup label={t("训练模型")}>{checkpoints.map((cp) => (
                  <option key={cp.id} value={cp.id} disabled={!cp.selectable}>
                    {checkpointLabels.get(cp.id)}
                  </option>
                ))}</optgroup>}
              </select>
              <ChevronDown size={14} />
            </div>
            <p className="opponent-note">
              {value === "random"
                ? t("从完整合法动作中等概率选择")
                : value.startsWith("llm:")
                  ? t("通用 LLM · {name}", { name: catalog.llms?.find((c) => c.id === value)?.name || t("请刷新重选") })
                : t("已选固定模型 · {step} 步", { step: checkpoints.find((c) => c.id === value)?.step.toLocaleString(locale) || t("请刷新重选") })}
            </p>
          </section>
        ))}
      </div>
      </div>
      <div className="launch-panel">
        <div className="launch-tip">
          <div className="launch-symbol">{t("東")}</div>
          <div>
            <strong>{spectator ? t("四家就位，静候开局。") : t("东家，等你入席。")}</strong>
            <p>{spectator ? t("四席自动对战 · 可暂停、单步与查看AI决策说明") : t("真人起始东家 · 对局仅保留在当前页面，退出后重新开始")}</p>
          </div>
        </div>
        <div className="launch-summary" aria-label={t("牌局配置摘要")}>
          <div><span>{t("赛制")}</span><strong>{modes[mode].name}</strong></div>
          <div><span>{t("参与方式")}</span><strong>{spectator ? t("纯 AI 观战") : t("真人对战")}</strong></div>
          <div><span>{t("AI 席位")}</span><strong>{seats.length} {t("席")}</strong></div>
        </div>
        {ready && invalidSeat !== -1 && (
          <p className="launch-error" role="status">{t("{seat}所选策略已不可用，请重新选择。", { seat: seatLabels[invalidSeat] })}</p>
        )}
        <button
          className="button primary start-button"
          disabled={
            !ready ||
            pending ||
            invalidSeat !== -1
          }
          onClick={() => start(mode, seats, spectator)}
        >
          {pending ? t("正在处理…") : !ready ? t("连接牌馆中…") : createUncertain ? t("重试确认开局") : spectator ? t("开始观战") : t("开始对局")}
          <ArrowRight size={18} />
        </button>
      </div>
      </div>
      <section className="library-panel">
        <div className="section-heading">
          <div>
            <span className="library-icon">
              <Cpu size={19} />
            </span>
            <h2>{t("本地模型库")}</h2>
            <span className="count-pill">{selectable.length}</span>
          </div>
          <button className="text-button" onClick={refresh} disabled={configLocked || !ready}>
            <RefreshCw size={14} />
            {pending ? t("请稍候…") : t("刷新列表")}
          </button>
        </div>
        {!catalog.runs.length ? (
          <div className="empty-library">
            <span className="empty-icon">
              <Bot size={23} />
            </span>
            <div>
              <strong>{t("下一位对手，正在成长。")}</strong>
              <p>{t("模型目录暂无训练记录。现在就可以与三位随机对手开一桌。")}</p>
            </div>
            <span className="outline-tag">{t("随机策略可用")}</span>
          </div>
        ) : (
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
                onClick={() => setShowRuns(!showRuns)}
              >
                {showRuns ? t("收起训练记录") : t("展开训练记录")}
                <ChevronDown size={14} />
              </button>
            </div>
            {!filtered.length && <p className="search-empty" role="status">{t("没有找到匹配的训练记录，请换个关键词。")}</p>}
            {filtered.map((run) => (
              <div className="run-card" key={run.run_id}>
                <div className="run-heading">
                  <strong>{run.experiment}</strong>
                  <span>{run.training_type}</span>
                  <code title={run.created_at || ""}>{run.run_id}</code>
                </div>
                <div className="checkpoint-list">
                  {run.checkpoints.length ? (
                    run.checkpoints.map((cp) => (
                      <div key={cp.id} className="checkpoint">
                        <div>
                          <b>
                            {cp.step.toLocaleString(locale)} <small>{t("步")}</small>
                          </b>
                          <code>{cp.checkpoint_id}</code>
                          <small title={cp.attempt_id}>
                            Attempt {cp.attempt_id.slice(0, 8)}
                          </small>
                          <small>{t(cp.status)}</small>
                        </div>
                        <div className="assign-buttons">
                          {seatLabels.map((n, i) => (
                            <button
                              className="button subtle"
                              key={n}
                              disabled={!cp.selectable || configLocked}
                              onClick={() =>
                                setSeat(i, cp.id)
                              }
                            >
                              {t("选为")}{n}
                            </button>
                          ))}
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
                      <span>
                        {a.outcome === "unknown" ? t("状态未知") : t(a.outcome)}
                      </span>
                      <span>{t("最近记录")}{a.progress?.global_step ?? "—"} {t("步")}</span>
                      {a.source_checkpoint && (
                        <small title={a.source_checkpoint?.run_id}>
                          {t("恢复自")}{a.source_checkpoint?.checkpoint_id} ·{" "}
                          {a.source_checkpoint?.global_step} {t("步")}</small>
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
        )}
        {catalog.errors.length > 0 &&
          catalog.runs.length > 0 &&
          catalog.errors.map((e, i) => (
            <p className="inline-error" key={i}>
              {t(e)}
            </p>
          ))}
      </section>
      <footer className="page-footer">
        <span>{t("一张牌，一次选择。")}</span>
        <span>
          <a href="https://www.riichilm.top" target="_blank" rel="noopener noreferrer" aria-label={t("RiichiLM 官网")}>
            www.riichilm.top
          </a> · {t("仅供学习与研究")}
        </span>
      </footer>
    </>
  );
}
