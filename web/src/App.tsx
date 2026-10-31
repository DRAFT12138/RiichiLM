import { useI18n, LanguageSwitcher } from "./i18n";
import { useEffect, useRef, useState } from "react";
import { ArrowUpRight, CircleDot, Sparkles, X } from "lucide-react";
import * as api from "./api";
import type { Candidate, Catalog, Mode, Table, SpectatorControl, SetupConfig } from "./types";
import { Setup } from "./Setup";
import { Game } from "./Game";
import { randomCast } from "./cast";
const invalidPage = (error: unknown) => error instanceof api.ApiError &&
  (error.status === 403 || (error.status === 409 && error.code === "STALE_DECISION" && error.message === "页面已失效，请刷新并重新开局"));
export function App() {
  const { locale, t } = useI18n();
  useEffect(() => {
    document.documentElement.lang = locale;
    document.title = t("RiichiLM · 学习、对局与观战");
    document.querySelector('meta[name="description"]')?.setAttribute("content", t("用大语言模型、监督学习与强化学习，让机器学打立直麻将。"));
  }, [locale, t]);
  const [catalog, setCatalog] = useState<Catalog>({ runs: [], errors: [] }),
    [table, setTable] = useState<Table | null>(null),
    [ready, setReady] = useState(false),
    [pending, setPending] = useState(false),
    [error, setError] = useState(""),
    [device, setDevice] = useState("cpu"),
    [cast, setCast] = useState(() => randomCast()),
    [setupConfig, setSetupConfig] = useState<SetupConfig>({ mode: "hanchan", opponents: ["random", "random", "random"], spectator: false, primary: "random" }),
    [createUncertain, setCreateUncertain] = useState(false),
    [sessionInvalid, setSessionInvalid] = useState(false);
  const active = useRef<string | null>(null),
    locked = useRef(false),
    pollError = useRef(false),
    sessionInvalidRef = useRef(false),
    refreshPageRef = useRef<HTMLButtonElement>(null),
    mounted = useRef(true),
    createAttempt = useRef<{ id: string; mode: Mode; opponents: string[]; spectator: boolean; uncertain: boolean } | null>(null);
  useEffect(() => {
    mounted.current = true;
    let cancelled = false;
    void api
      .initialize()
      .then(async () => {
        const [c, b] = await Promise.all([
          api.catalog(),
          api.request<{ device: string }>("bootstrap"),
        ]);
        if (!cancelled) {
          setCatalog(c);
          setDevice(b.device);
          setReady(true);
        }
      })
      .catch((e) => {
        if (!cancelled) setError(e.message);
      });
    return () => {
      cancelled = true;
      mounted.current = false;
    };
  }, []);
  useEffect(() => {
    if (!ready) return;
    let stopped = false;
    let timer: ReturnType<typeof setTimeout>;
    const controller = new AbortController();
    async function poll() {
      const polledTableId = active.current;
      try {
        const t = await api.table(controller.signal);
        if (!stopped && !sessionInvalidRef.current && active.current === polledTableId) {
          if (pollError.current && !locked.current && !sessionInvalidRef.current && !createAttempt.current?.uncertain) {
            pollError.current = false;
            setError("");
          }
          if (t && polledTableId === t.table_id) {
            active.current = t.table_id;
            setTable((old) => old?.table_id === t.table_id && old.view_version >= t.view_version ? old : t);
          } else if (!t && polledTableId && !locked.current) {
            active.current = null;
            setTable(null);
            setCast((previous) => randomCast(previous));
            setError("当前牌桌已失效，请重新开局。");
          }
        }
      } catch (e) {
        if (!stopped && active.current === polledTableId && !locked.current && !sessionInvalidRef.current && !createAttempt.current?.uncertain) {
          pollError.current = true;
          setError((e as Error).message);
        }
      } finally {
        if (!stopped) timer = setTimeout(poll, document.hidden ? 5000 : 500);
      }
    }
    void poll();
    return () => {
      stopped = true;
      controller.abort();
      clearTimeout(timer);
    };
  }, [ready]);
  useEffect(() => {
    const leave = () => {
      if (active.current) api.leave(active.current);
    };
    window.addEventListener("pagehide", leave);
    return () => window.removeEventListener("pagehide", leave);
  }, []);
  useEffect(() => {
    if (sessionInvalid) refreshPageRef.current?.focus();
  }, [sessionInvalid]);
  async function run(operation: () => Promise<void>) {
    if (locked.current) return false;
    locked.current = true;
    pollError.current = false;
    setPending(true);
    setError("");
    try {
      await operation();
      return true;
    } catch (e) {
      setError((e as Error).message);
      return false;
    } finally {
      locked.current = false;
      if (mounted.current) setPending(false);
    }
  }
  function invalidateSession() {
    sessionInvalidRef.current = true;
    setSessionInvalid(true);
  }
  function start(mode: Mode, opponents: string[], spectator: boolean) {
    if (locked.current) return;
    const attempt = createAttempt.current ?? { id: crypto.randomUUID(), mode, opponents: [...opponents], spectator, uncertain: false };
    createAttempt.current = attempt;
    void run(async () => {
      try {
        const result = await api.create(attempt.mode, attempt.opponents, attempt.id, attempt.spectator);
        active.current = result.table_id;
        createAttempt.current = null;
        setCreateUncertain(false);
        setTable({ table_id: result.table_id, page_id: api.pageId, view_version: 0, state: "loading", mode: attempt.mode, spectator: attempt.spectator, busy: false } as Table);
      } catch (error) {
        // A lost response can hide an accepted table; only the same request ID
        // and immutable payload can safely resolve that uncertainty.
        const closedAttempt = error instanceof api.ApiError && error.status === 409 &&
          error.code === "STALE_DECISION" && error.message === "本次开局已结束，请重新开局";
        const ownershipConflict = error instanceof api.ApiError && error.status === 409 &&
          error.code === "STALE_DECISION" && ["请先退出当前桌", "页面已失效，请刷新并重新开局", "同一请求ID不能更换内容"].includes(error.message);
        const definitelyRejected = error instanceof api.ApiError && (closedAttempt ||
          (!attempt.uncertain && error.status >= 400 && error.status < 500 &&
            !ownershipConflict));
        if (definitelyRejected) {
          createAttempt.current = null;
          setCreateUncertain(false);
          throw error;
        }
        attempt.uncertain = true;
        setCreateUncertain(true);
        throw new Error("开局结果尚未确认。请重试原配置；如需放弃，可刷新页面重新配置。");
      }
    });
  }
  async function exit() {
    return run(async () => {
      const id = active.current;
      if (id) {
        try { await api.close(id); }
        catch (error) {
          // A lost close response is harmless to retry; a confirmed empty table
          // means the close did succeed.
          const current = await api.table().catch(() => undefined);
          if (current !== null) throw error;
        }
      }
      active.current = null;
      setTable(null);
      setCast((previous) => randomCast(previous));
    });
  }
  async function causalSnapshot(expected: Table, requireAdvance = true) {
    while (mounted.current && active.current === expected.table_id) {
      try {
        const current = await api.table();
        if (!current || current.table_id !== expected.table_id) {
          active.current = null;
          setTable(null);
          setCast((previous) => randomCast(previous));
          throw new api.ApiError("当前牌桌已失效，请重新开局。", 409, "TABLE_ENDED");
        }
        if (requireAdvance && current.view_version <= expected.view_version) {
          await new Promise((resolve) => setTimeout(resolve, 500));
          continue;
        }
        setTable((old) => old?.table_id === current.table_id && old.view_version >= current.view_version ? old : current);
        return;
      } catch (error) {
        if (error instanceof api.ApiError && error.code === "TABLE_ENDED") throw error;
        if (invalidPage(error)) {
          invalidateSession();
          throw new Error("页面连接已失效，请刷新页面重新开始。");
        }
        setError("正在等待牌桌状态恢复，请稍候；如长时间无法恢复，可刷新页面。");
        await new Promise((resolve) => setTimeout(resolve, 1000));
      }
    }
  }
  async function command(expected: Table, send: (requestId: string) => Promise<api.CommandReceipt>) {
    const requestId = crypto.randomUUID();
    let uncertain = false;
    let receipt: api.CommandReceipt | undefined;
    while (mounted.current && active.current === expected.table_id) {
      try {
        receipt = await send(requestId);
        break;
      } catch (error) {
        if (invalidPage(error)) {
          invalidateSession();
          throw new Error("页面连接已失效，请刷新页面重新开始。");
        }
        if (uncertain && error instanceof api.ApiError && error.status === 409 &&
            error.message === "牌桌已结束或已更换") {
          await causalSnapshot(expected, false);
        }
        if (!uncertain && error instanceof api.ApiError && error.status >= 400 && error.status < 500) {
          await causalSnapshot(expected, false);
          throw error;
        }
        uncertain = true;
        setError("连接中断，正在用原请求确认操作；如长时间无法恢复，可刷新页面。");
        await new Promise((resolve) => setTimeout(resolve, 1000));
      }
    }
    await causalSnapshot(expected, receipt?.status !== "rejected");
    if (receipt?.status === "rejected") throw new Error(receipt.error || "操作未执行，请重试。");
    if (mounted.current) setError("");
  }
  async function action(c?: Candidate) {
    if (table)
      return run(async () => {
        await command(table, (requestId) => api.submit(table, c, requestId));
      });
    return false;
  }
  function control(control: SpectatorControl, value?: number) {
    if (table) void run(async () => { await command(table, (requestId) => api.control(table, control, value, requestId)); });
  }
  const refresh = () => {
    void run(async () => setCatalog(await api.refreshCatalog()));
  };
  const reconnect = () => {
    void run(async () => {
      await api.initialize();
      const [c, b] = await Promise.all([
        api.catalog(),
        api.request<{ device: string }>("bootstrap"),
      ]);
      setCatalog(c);
      setDevice(b.device);
      setReady(true);
    });
  };
  return (
    <div className={`app-shell ${table ? "in-game" : ""}`}>
      <div className="main-wrap">
        {!table && (
          <header className="topbar">
            <a
              className="brand"
              href="#"
              onClick={(e) => e.preventDefault()}
              aria-label="RiichiLM"
            >
              <span className="brand-symbol" aria-hidden="true"><Sparkles size={23} /></span>
              <span>
                <strong>RiichiLM</strong>
                <small>{t("学习 · 对局 · 探索")}</small>
              </span>
            </a>
            <span className="header-place">{t("对局大厅")}</span>
            <div className="topbar-status">
              <LanguageSwitcher />
              <span className="device-pill">
                <CircleDot size={13} />
                {ready ? t("本地牌馆") : t("连接中")} · {device.toUpperCase()}
              </span>
              <button
                className="header-library"
                onClick={() =>
                  document
                    .querySelector(".library-panel")
                    ?.scrollIntoView({ behavior: "smooth" })
                }
              >
                {t("模型与训练")}<ArrowUpRight size={14} />
              </button>
            </div>
          </header>
        )}
        <main className={table ? "game-content" : "setup-content"}>
          {error && (
            <div className="error-banner" role="alert">
              <span>{t(error)}</span>
              {!ready && <button className="retry-connect" disabled={pending} onClick={reconnect}>{t("重试连接")}</button>}
              {sessionInvalid && <button ref={refreshPageRef} className="retry-connect" onClick={() => window.location.reload()}>{t("刷新页面")}</button>}
              {ready && !createUncertain && !sessionInvalid && <button aria-label={t("关闭提示")} onClick={() => setError("")}>
                <X size={16} />
              </button>}
            </div>
          )}
          {table?.command_error && table.command_error !== error && (
            <div className="error-banner" role="alert">{t(table.command_error)}</div>
          )}
          {table ? (
            <Game
              table={table}
              pending={pending || sessionInvalid}
              sessionInvalid={sessionInvalid}
              onAction={action}
              onContinue={() => action()}
              onExit={exit}
              onControl={control}
              cast={cast}
            />
          ) : (
            <Setup
              catalog={catalog}
              ready={ready}
              refresh={refresh}
              start={start}
              pending={pending || sessionInvalid}
              cast={cast}
              shuffleCast={() => setCast((previous) => randomCast(previous))}
              config={setupConfig}
              setConfig={setSetupConfig}
              createUncertain={createUncertain}
            />
          )}
        </main>
      </div>
    </div>
  );
}
