import type { Candidate, Catalog, Mode, Table, SpectatorControl } from "./types";
export const pageId = crypto.randomUUID();
let token = "";
let bootstrapPromise: Promise<void> | undefined;
export class ApiError extends Error {
  constructor(message: string, public status: number, public code?: string) {
    super(message);
  }
}
export interface CommandReceipt {
  status: "processing" | "done" | "rejected";
  request_id: string;
  error?: string | null;
}
export async function request<T>(
  path: string,
  body?: unknown,
  signal?: AbortSignal,
): Promise<T> {
  const response = await fetch(`/api/${path}`, {
    method: body === undefined ? "GET" : "POST",
    signal,
    headers: { "Content-Type": "application/json", "X-Session-Token": token },
    body: body === undefined ? undefined : JSON.stringify(body),
    cache: "no-store",
  });
  const value = await response.json();
  if (!response.ok)
    throw new ApiError(value.error || `请求失败 (${response.status})`, response.status, value.code);
  return value as T;
}
export function initialize(): Promise<void> {
  bootstrapPromise ??= (async () => {
    const config = await request<{ token: string; generation: number }>(
      "bootstrap",
    );
    token = config.token;
    await request("page", { page_id: pageId, generation: config.generation });
  })().catch((error) => {
    bootstrapPromise = undefined;
    throw error;
  });
  return bootstrapPromise;
}
export const catalog = () => request<Catalog>("catalog");
export const refreshCatalog = () => request<Catalog>("catalog/refresh", {});
export const table = (signal?: AbortSignal) =>
  request<Table | null>(`table?page_id=${pageId}`, undefined, signal);
export const create = (mode: Mode, opponents: string[], requestId: string, spectator = false) =>
  request<{ table_id: string }>("table", {
    page_id: pageId,
    request_id: requestId,
    mode,
    opponents,
    spectator,
  });
export const control = (t: Table, control: SpectatorControl, value?: number, requestId: string = crypto.randomUUID()) =>
  request<CommandReceipt>("table/control", {
    page_id: pageId, table_id: t.table_id, request_id: requestId, control, value,
  });
export const close = (tableId: string) =>
  request("table/close", { page_id: pageId, table_id: tableId });
export const submit = (t: Table, candidate?: Candidate, requestId: string = crypto.randomUUID()) =>
  request<CommandReceipt>(candidate ? "table/action" : "table/continue", {
    page_id: pageId,
    table_id: t.table_id,
    request_id: requestId,
    ...(candidate
      ? { window: t.window, candidate: candidate.id }
      : { round_id: t.round_id, revision: t.revision }),
  });
export function leave(tableId: string) {
  void fetch("/api/table/close", {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-Session-Token": token },
    body: JSON.stringify({ page_id: pageId, table_id: tableId }),
    keepalive: true,
  }).catch(() => {});
}
