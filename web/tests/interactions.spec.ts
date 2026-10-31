import { expect, test, type Page } from "@playwright/test";
import { characterNames, randomCast } from "../src/cast";
import type { Candidate, Table } from "../src/types";

const discard: Candidate = {
  id: "discard",
  kind: "A_DISCARD",
  args: ["TILE_5pr", "TEDASHI"],
};
const pass: Candidate = { id: "pass", kind: "A_PASS", args: [] };
const seats = ["SELF", "SHIMOCHA", "TOIMEN", "KAMICHA"];

test("all generated portraits can be drawn and render in every character slot", async ({ page }) => {
  let state = 12345;
  const random = () => ((state = (state * 16807) % 2147483647) - 1) / 2147483646;
  const seenSeats = new Set<number>();
  const seenHosts = new Set<number>();
  for (let i = 0; i < 200; i++) {
    const cast = randomCast(undefined, random);
    expect(new Set(cast.seats).size).toBe(4);
    cast.seats.forEach((seat) => seenSeats.add(seat));
    seenHosts.add(cast.host);
  }
  expect(seenSeats.size).toBe(characterNames.length);
  expect(seenHosts.size).toBe(characterNames.length);
  expect(characterNames.length).toBe(13);

  await page.route("**/api/**", async (route) => {
    const path = new URL(route.request().url()).pathname;
    await route.fulfill({ json: path === "/api/bootstrap"
      ? { token: "fixture", generation: 1, device: "cpu" }
      : path === "/api/catalog" ? { runs: [], errors: [] } : {} });
  });
  await page.goto("/");
  const portraits = await page.evaluate(async (count) => {
    const contexts = ["bot-icon", "welcome-character", "compact-portrait", "seat-1", "seat-2", "self-info"];
    const result: string[][] = [];
    const urls = new Set<string>();
    for (let id = 0; id < count; id++) {
      const styles: string[] = [];
      for (const context of contexts) {
        const parent = document.createElement("div");
        const element = document.createElement("span");
        parent.className = context;
        element.className = `${context.startsWith("seat-") || context === "self-info" ? "seat-avatar" : context} portrait-${id}`;
        parent.append(element);
        document.body.append(parent);
        const style = getComputedStyle(element);
        const url = style.backgroundImage.match(/url\("?([^"\)]+)"?\)/)?.[1];
        if (!url) throw new Error(`Portrait ${id} has no image in ${context}`);
        urls.add(url);
        styles.push(`${style.backgroundImage}|${style.backgroundSize}|${style.backgroundPosition}`);
        parent.remove();
      }
      result.push(styles);
    }
    await Promise.all([...urls].map(async (url) => {
      if (!(await fetch(url)).ok) throw new Error(`Portrait failed to load: ${url}`);
    }));
    return result;
  }, characterNames.length);
  for (const contexts of portraits) expect(new Set(contexts).size).toBe(1);
  expect(new Set(portraits.map((contexts) => contexts[0])).size).toBe(characterNames.length);
});

function snapshot(candidates = [discard]): Table {
  return {
    table_id: "fixture",
    page_id: "fixture",
    view_version: 1,
    state: "playing",
    mode: "single",
    busy: false,
    round_id: 0,
    revision: 1,
    round_wind: "E",
    round_number: 1,
    honba: 0,
    kyotaku: 0,
    live_remaining: 40,
    dora_indicators: ["3s"],
    active_seats: [0],
    response_window: false,
    response_trigger: null,
    window: { window_id: 1 },
    candidates,
    thinking: null,
    result: null,
    cross_mode: false,
    players: [0, 1, 2, 3].map((seat) => ({
      seat,
      wind: "ESWN"[seat],
      score: 25000,
      dealer: seat === 0,
      riichi: false,
      hand:
        seat === 0
          ? [
              "5pr",
              "1m",
              "2m",
              "3m",
              "4m",
              "5m",
              "6m",
              "7m",
              "8m",
              "9m",
              "1s",
              "2s",
              "3s",
            ]
          : null,
      drawn: seat === 0 ? "9s" : null,
      hand_count: 13,
      melds: [],
      river: [
        { tile: "5p", tsumogiri: false, riichi: false, called: false },
        { tile: "5pr", tsumogiri: false, riichi: false, called: true },
        { tile: "5p", tsumogiri: true, riichi: false, called: false },
      ],
    })),
  };
}

async function openFixture(page: Page, table: Table) {
  const actions: string[] = [];
  let failure = false;
  await page.route("**/api/**", async (route) => {
    const path = new URL(route.request().url()).pathname;
    let body: unknown = {};
    let status = 200;
    if (path === "/api/bootstrap")
      body = { token: "fixture", generation: 1, device: "cpu" };
    if (path === "/api/catalog") body = { runs: [], errors: [] };
    if (path === "/api/table")
      body =
        route.request().method() === "POST"
          ? { table_id: table.table_id }
          : table;
    if (path === "/api/table/action") {
      actions.push(route.request().postDataJSON().candidate);
      if (failure) {
        failure = false;
        status = 409;
        body = { error: "请重试" };
      }
    }
    await route.fulfill({ status, json: body });
  });
  await page.goto("/");
  if (table.spectator) await page.getByRole("button", { name: "纯 AI 观战" }).click();
  await page.locator(".start-button").click();
  await expect(page.locator(".self-hand")).toBeVisible();
  return {
    actions,
    failNext: () => {
      failure = true;
    },
  };
}

test("lobby keeps configuration and launch summary usable on desktop and mobile", async ({ page }) => {
  await page.route("**/api/**", async (route) => {
    const path = new URL(route.request().url()).pathname;
    await route.fulfill({ json: path === "/api/bootstrap"
      ? { token: "fixture", generation: 1, device: "cpu" }
      : path === "/api/catalog" ? { runs: [], errors: [] } : {} });
  });
  for (const width of [1280, 390]) {
    await page.setViewportSize({ width, height: 800 });
    await page.goto("/");
    await expect(page.locator(".start-button")).toBeEnabled();
    await expect.poll(() => page.locator(".welcome-character").evaluate(async (el) => {
      const url = getComputedStyle(el).backgroundImage.match(/url\("?([^"\)]+)"?\)/)?.[1];
      return url ? (await fetch(url)).ok : false;
    })).toBe(true);
    await page.getByRole("button", { name: /单局练习/ }).click();
    await page.getByRole("button", { name: "纯 AI 观战" }).click();
    await expect(page.getByLabel("牌局配置摘要")).toContainText("单局练习");
    await expect(page.getByLabel("牌局配置摘要")).toContainText("4 席");
    await expect(page.locator(".opponent-card")).toHaveCount(4);
    expect(await page.locator("body").evaluate((e) => e.scrollWidth)).toBeLessThanOrEqual(width);
    const button = (await page.locator(".start-button").boundingBox())!;
    expect(button.x).toBeGreaterThanOrEqual(0);
    expect(button.x + button.width).toBeLessThanOrEqual(width);
    if (width === 1280) {
      expect(button.y + button.height).toBeLessThanOrEqual(800);
      const modes = (await page.locator(".mode-grid").boundingBox())!;
      const seats = (await page.locator(".opponent-grid").boundingBox())!;
      expect(modes.x + modes.width).toBeLessThanOrEqual(seats.x);
    }
  }
});

test("mobile spectator scores do not obscure the opposite hand", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  const table = snapshot();
  table.spectator = true;
  table.viewer = 0;
  table.paused = true;
  table.playback_interval = 0.75;
  await openFixture(page, table);
  const opposite = (await page.locator(".seat-2").boundingBox())!;
  const avatars = await page.locator(".seat-2 .seat-avatar, .self-info .seat-avatar, .compact-portrait").evaluateAll((elements) =>
    elements.map((el) => el.className.match(/portrait-(\d+)/)?.[1]),
  );
  expect(new Set(avatars).size).toBe(4);
  for (const score of await page.locator(".compact-seat-score").all()) {
    const box = (await score.boundingBox())!;
    expect(box.y).toBeGreaterThanOrEqual(opposite.y + opposite.height);
  }
  await expect(page.getByRole("combobox", { name: "观战视角" })).toBeVisible();
  for (const width of [900, 1280]) {
    await page.setViewportSize({ width, height: 900 });
    const controls = (await page.locator(".spectator-controls").boundingBox())!;
    const seat = (await page.locator(".seat-2").boundingBox())!;
    const overlap = controls.x < seat.x + seat.width && controls.x + controls.width > seat.x && controls.y < seat.y + seat.height && controls.y + controls.height > seat.y;
    expect(overlap).toBe(false);
  }
});

test("desktop spectator notes stay beside the table without covering the play area", async ({ page }) => {
  const table = snapshot([]);
  table.spectator = true;
  table.viewer = 0;
  table.paused = true;
  table.players!.forEach((player) => {
    player.river = Array.from({ length: 24 }, () => ({ tile: "5p", tsumogiri: false, riichi: false, called: false }));
  });
  table.players![0].hand = table.players![0].hand!.slice(0, 10);
  table.players![0].melds = [{ name: "MELD_PON", tiles: ["5m", "5m", "5mr"], base_tile: "5m", from: 3 }];
  await openFixture(page, table);
  await page.getByRole("button", { name: "AI决策说明" }).click();
  for (const [width, height] of [[1280, 800], [1280, 900], [1440, 800], [1920, 900]]) {
    await page.setViewportSize({ width, height });
    const felt = (await page.locator(".felt-table").boundingBox())!;
    const notes = (await page.locator(".decision-panel").boundingBox())!;
    expect(notes.x).toBeGreaterThanOrEqual(felt.x + felt.width);
    const toolbar = (await page.locator(".spectator-controls").boundingBox())!;
    const topSeat = (await page.locator(".seat-2").boundingBox())!;
    expect(toolbar.x < topSeat.x + topSeat.width && toolbar.x + toolbar.width > topSeat.x && toolbar.y < topSeat.y + topSeat.height && toolbar.y + toolbar.height > topSeat.y, `${width}×${height}: toolbar overlaps the top seat`).toBe(false);
    for (const locator of [".self-hand", ".table-center", ".river-0", ".river-1", ".river-2", ".river-3"]) {
      const box = (await page.locator(locator).boundingBox())!;
      expect(box.x, locator).toBeGreaterThanOrEqual(felt.x);
      expect(box.x + box.width, locator).toBeLessThanOrEqual(felt.x + felt.width);
      expect(box.y, locator).toBeGreaterThanOrEqual(felt.y);
      expect(box.y + box.height, locator).toBeLessThanOrEqual(felt.y + felt.height);
    }
    for (const element of await page.locator(".turn-status, .compact-seat-score, .seat-2, .self-info .melds, .table-center, .self-hand, .spectator-controls, .table-tools").all()) {
      const a = (await element.boundingBox())!;
      for (const river of await page.locator(".river").all()) {
        const b = (await river.boundingBox())!;
        expect(a.x < b.x + b.width && a.x + a.width > b.x && a.y < b.y + b.height && a.y + a.height > b.y,
          `${width}: ${await element.getAttribute("class")} overlaps ${await river.getAttribute("class")}`).toBe(false);
      }
    }
    expect(await page.locator("body").evaluate((el) => el.scrollWidth)).toBeLessThanOrEqual(width);
  }
  await page.getByRole("button", { name: "AI决策说明" }).click();
  await expect(page.locator(".decision-panel")).toHaveCount(0);
});

test("characters reshuffle in the lobby and stay with their seats throughout a table", async ({ page }) => {
  const table = snapshot();
  table.spectator = true;
  table.viewer = 0;
  table.paused = true;
  await page.route("**/api/**", async (route) => {
    const path = new URL(route.request().url()).pathname;
    const body = path === "/api/bootstrap" ? { token: "fixture", generation: 1, device: "cpu" }
      : path === "/api/catalog" ? { runs: [], errors: [] }
      : path === "/api/table" ? (route.request().method() === "POST" ? { table_id: table.table_id } : table)
      : {};
    await route.fulfill({ json: body });
  });
  const portrait = (selector: string) => page.locator(selector).getAttribute("class").then((c) => c?.match(/portrait-\d+/)?.[0]);
  const lobbyCast = async () => Promise.all([0, 1, 2, 3].map((seat) => portrait(`.opponent-card[data-seat="${seat}"] .bot-icon`)));
  await page.goto("/");
  await page.getByRole("button", { name: "纯 AI 观战" }).click();
  const before = await lobbyCast();
  const host = await portrait(".welcome-character");
  await page.getByRole("button", { name: "换一批角色" }).click();
  const chosen = await lobbyCast();
  const nextHost = await portrait(".welcome-character");
  expect(new Set(chosen).size).toBe(4);
  expect(chosen).not.toEqual(before);
  expect(nextHost).not.toBe(host);
  await page.getByRole("button", { name: "开始观战" }).click();
  await expect(page.locator(".self-hand")).toBeVisible();
  for (let seat = 0; seat < 4; seat++) {
    const selector = seat === 0 ? ".self-info .seat-avatar" : `.opponent-seat[data-seat="${seat}"] .seat-avatar`;
    expect(await portrait(selector)).toBe(chosen[seat]);
  }
  table.viewer = 2;
  table.view_version++;
  await expect(page.locator('.self-info[data-seat="2"]')).toBeVisible();
  expect(await portrait(".self-info .seat-avatar")).toBe(chosen[2]);
  expect(await portrait('.opponent-seat[data-seat="0"] .seat-avatar')).toBe(chosen[0]);
  await page.getByRole("button", { name: "退出牌桌", exact: true }).click();
  await page.getByRole("dialog").getByRole("button", { name: "退出牌桌", exact: true }).click();
  await page.getByRole("button", { name: "纯 AI 观战" }).click();
  expect(await lobbyCast()).not.toEqual(chosen);
  expect(await portrait(".welcome-character")).not.toBe(nextHost);
});

test("connection failure offers a working retry", async ({ page }) => {
  let attempts = 0;
  await page.route("**/api/**", async (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === "/api/bootstrap" && attempts++ === 0) {
      await route.fulfill({ status: 503, json: { error: "连接暂不可用" } });
      return;
    }
    await route.fulfill({ json: path === "/api/bootstrap"
      ? { token: "fixture", generation: 1, device: "cpu" }
      : path === "/api/catalog" ? { runs: [], errors: [] } : {} });
  });
  await page.goto("/");
  await expect(page.getByRole("alert")).toContainText("连接暂不可用");
  await page.getByRole("button", { name: "重试连接" }).click();
  await expect(page.locator(".start-button")).toBeEnabled();
  await expect(page.getByRole("alert")).toHaveCount(0);
});

test("a lost create response retries the same request and freezes its configuration", async ({ page }) => {
  const table = snapshot();
  const requests: any[] = [];
  await page.route("**/api/**", async (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === "/api/table" && route.request().method() === "POST") {
      requests.push(route.request().postDataJSON());
      if (requests.length === 1) { await route.abort("failed"); return; }
      await route.fulfill({ json: { table_id: table.table_id } });
      return;
    }
    await route.fulfill({ json: path === "/api/bootstrap" ? { token: "fixture", generation: 1, device: "cpu" }
      : path === "/api/catalog" ? { runs: [], errors: [] }
      : path === "/api/table" ? table : {} });
  });
  await page.goto("/");
  await expect(page.locator(".start-button")).toBeEnabled();
  await page.locator(".start-button").click();
  await expect(page.getByRole("alert")).toContainText("开局结果尚未确认");
  await expect(page.getByRole("button", { name: /东风战/ })).toBeDisabled();
  await expect(page.locator("#opponent-0")).toBeDisabled();
  await page.getByRole("button", { name: "重试确认开局" }).click();
  await expect(page.locator(".self-hand")).toBeVisible();
  expect(requests).toHaveLength(2);
  expect(requests[1]).toEqual(requests[0]);
});

test("configuration cannot change while a create request is in flight", async ({ page }) => {
  let release!: () => void, captured!: () => void;
  const waiting = new Promise<void>((resolve) => { captured = resolve; });
  const hold = new Promise<void>((resolve) => { release = resolve; });
  await page.route("**/api/**", async (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === "/api/table" && route.request().method() === "POST") {
      captured(); await hold;
      await route.fulfill({ json: { table_id: "fixture" } });
      return;
    }
    await route.fulfill({ json: path === "/api/bootstrap" ? { token: "fixture", generation: 1, device: "cpu" }
      : path === "/api/catalog" ? { runs: [], errors: [] } : path === "/api/table" ? snapshot() : {} });
  });
  await page.goto("/");
  await expect(page.locator(".start-button")).toBeEnabled();
  await page.locator(".start-button").click();
  await waiting;
  await expect(page.getByRole("button", { name: /东风战/ })).toBeDisabled();
  await expect(page.getByRole("button", { name: "纯 AI 观战" })).toBeDisabled();
  await expect(page.locator("#opponent-0")).toBeDisabled();
  await expect(page.getByRole("button", { name: "换一批角色" })).toBeDisabled();
  release();
  await expect(page.locator(".self-hand")).toBeVisible();
});

test("a late empty poll started in the lobby cannot close a newly created table", async ({ page }) => {
  let release!: () => void, captured!: () => void;
  const gotOldPoll = new Promise<void>((resolve) => { captured = resolve; });
  const hold = new Promise<void>((resolve) => { release = resolve; });
  let firstPoll = true;
  await page.route("**/api/**", async (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === "/api/table" && route.request().method() === "GET" && firstPoll) {
      firstPoll = false; captured(); await hold;
      await route.fulfill({ json: null });
      return;
    }
    await route.fulfill({ json: path === "/api/bootstrap" ? { token: "fixture", generation: 1, device: "cpu" }
      : path === "/api/catalog" ? { runs: [], errors: [] }
      : path === "/api/table" ? (route.request().method() === "POST" ? { table_id: "fixture" } : snapshot()) : {} });
  });
  await page.goto("/");
  await expect(page.locator(".start-button")).toBeEnabled();
  await gotOldPoll;
  await page.locator(".start-button").click();
  await expect(page.locator(".loading-room")).toBeVisible();
  release();
  await expect(page.locator(".self-hand")).toBeVisible();
  await expect(page.locator(".error-banner")).toHaveCount(0);
});

test("a definite configuration rejection permits changing and submitting a new request", async ({ page }) => {
  const requests: any[] = [];
  await page.route("**/api/**", async (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === "/api/table" && route.request().method() === "POST") {
      requests.push(route.request().postDataJSON());
      if (requests.length === 1) { await route.fulfill({ status: 409, json: { error: "配置无效", code: "CONFIG_INVALID" } }); return; }
      await route.fulfill({ json: { table_id: "fixture" } });
      return;
    }
    await route.fulfill({ json: path === "/api/bootstrap" ? { token: "fixture", generation: 1, device: "cpu" }
      : path === "/api/catalog" ? { runs: [], errors: [] } : path === "/api/table" ? snapshot() : {} });
  });
  await page.goto("/");
  await expect(page.locator(".start-button")).toBeEnabled();
  await page.locator(".start-button").click();
  await expect(page.getByRole("alert")).toContainText("配置无效");
  await page.getByRole("button", { name: /东风战/ }).click();
  await page.locator(".start-button").click();
  await expect(page.locator(".self-hand")).toBeVisible();
  expect(requests[0].request_id).not.toBe(requests[1].request_id);
  expect(requests[1].mode).toBe("tonpuu");
});

test("a first explicit stale-decision validation rejection does not lock configuration", async ({ page }) => {
  await page.route("**/api/**", async (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === "/api/table" && route.request().method() === "POST") {
      await route.fulfill({ status: 409, json: { error: "请选择有效的玩家策略", code: "STALE_DECISION" } });
      return;
    }
    await route.fulfill({ json: path === "/api/bootstrap" ? { token: "fixture", generation: 1, device: "cpu" }
      : path === "/api/catalog" ? { runs: [], errors: [] } : null });
  });
  await page.goto("/");
  await expect(page.locator(".start-button")).toBeEnabled();
  await page.locator(".start-button").click();
  await expect(page.getByRole("alert")).toContainText("请选择有效的玩家策略");
  await expect(page.getByRole("button", { name: /东风战/ })).toBeEnabled();
  await expect(page.locator("#opponent-0")).toBeEnabled();
});

test("returning to the lobby preserves all selected seats and mode", async ({ page }) => {
  const table = snapshot();
  table.spectator = true;
  table.viewer = 0;
  await page.route("**/api/**", async (route) => {
    const path = new URL(route.request().url()).pathname;
    await route.fulfill({ json: path === "/api/bootstrap" ? { token: "fixture", generation: 1, device: "cpu" }
      : path === "/api/catalog" ? { runs: [], errors: [], llms: [{ id: "llm:one", name: "模型", model: "example" }] }
      : path === "/api/table" ? (route.request().method() === "POST" ? { table_id: table.table_id } : table) : {} });
  });
  await page.goto("/");
  await expect(page.locator(".start-button")).toBeEnabled();
  await page.getByRole("button", { name: /东风战/ }).click();
  await page.getByRole("button", { name: "纯 AI 观战" }).click();
  await page.locator("#opponent-0").selectOption("llm:one");
  await page.locator("#opponent-2").selectOption("llm:one");
  await page.getByRole("button", { name: "开始观战" }).click();
  await expect(page.locator(".self-hand")).toBeVisible();
  await page.getByRole("button", { name: "退出牌桌", exact: true }).click();
  await page.getByRole("dialog").getByRole("button", { name: "退出牌桌", exact: true }).click();
  await expect(page.getByRole("button", { name: "纯 AI 观战" })).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByRole("button", { name: /东风战/ })).toHaveAttribute("aria-pressed", "true");
  await expect(page.locator("#opponent-0")).toHaveValue("llm:one");
  await expect(page.locator("#opponent-2")).toHaveValue("llm:one");
});

test("checkpoint options remain distinguishable even when their visible metadata matches", async ({ page }) => {
  const common = { checkpoint_id: "checkpoint", attempt_id: "attempt", step: 100, created_at: "", selectable: true, status: "ready" };
  await page.route("**/api/**", async (route) => {
    const path = new URL(route.request().url()).pathname;
    await route.fulfill({ json: path === "/api/bootstrap" ? { token: "fixture", generation: 1, device: "cpu" }
      : path === "/api/catalog" ? { runs: [{ run_id: "run", experiment: "experiment", created_at: "", training_type: "NTP", checkpoints: [{ ...common, id: "first-full-id" }, { ...common, id: "second-full-id" }], attempts: [], errors: [] }], errors: [] } : {} });
  });
  await page.goto("/");
  const select = page.locator("#opponent-0");
  await expect(select.locator('option[value="first-full-id"]')).toContainText("first-full-id");
  await expect(select.locator('option[value="second-full-id"]')).toContainText("second-full-id");
  await select.selectOption("second-full-id");
  await expect(select).toHaveValue("second-full-id");
});

test("a removed strategy explains why starting is disabled and can be replaced", async ({ page }) => {
  await page.route("**/api/**", async (route) => {
    const path = new URL(route.request().url()).pathname;
    const body = path === "/api/bootstrap" ? { token: "fixture", generation: 1, device: "cpu" }
      : path === "/api/catalog" ? { runs: [], errors: [], llms: [{ id: "llm:one", name: "模型", model: "example" }] }
      : path === "/api/catalog/refresh" ? { runs: [], errors: [], llms: [] } : {};
    await route.fulfill({ json: body });
  });
  await page.goto("/");
  await expect(page.locator(".start-button")).toBeEnabled();
  await page.locator("#opponent-0").selectOption("llm:one");
  await page.getByRole("button", { name: "刷新列表" }).click();
  await expect(page.locator(".launch-error")).toContainText("下家所选策略已不可用");
  await expect(page.locator(".start-button")).toBeDisabled();
  await expect(page.locator("#opponent-0 option:checked")).toHaveText("已不可用，请重新选择");
  await page.locator("#opponent-0").selectOption("random");
  await expect(page.locator(".start-button")).toBeEnabled();
});

test("model search explains when there are no matches", async ({ page }) => {
  await page.route("**/api/**", async (route) => {
    const path = new URL(route.request().url()).pathname;
    const body = path === "/api/bootstrap" ? { token: "fixture", generation: 1, device: "cpu" }
      : path === "/api/catalog" ? { runs: [{ run_id: "run-one", experiment: "experiment", created_at: "", training_type: "test", checkpoints: [], attempts: [], errors: [] }], errors: [] } : {};
    await route.fulfill({ json: body });
  });
  await page.goto("/");
  await page.getByRole("textbox", { name: "搜索模型和训练记录" }).fill("不存在的记录");
  await expect(page.locator(".search-empty")).toContainText("没有找到匹配的训练记录");
  await page.getByRole("textbox", { name: "搜索模型和训练记录" }).fill("");
  await expect(page.locator(".run-card")).toHaveCount(1);
});

test("result can be hidden to inspect the table and reopened", async ({ page }) => {
  const table = snapshot([]);
  table.state = "round_result";
  table.result = { round_id: 0, revision: 1, kind: "draw", reason: null, winners: [], score_delta: [0, 0, 0, 0], scores: [25000, 25000, 25000, 25000], kyotaku: 0, game_over: false };
  await openFixture(page, table);
  await expect(page.locator(".result-modal")).toBeVisible();
  await expect(page.getByRole("button", { name: "查看牌桌" })).toBeFocused();
  await page.keyboard.press("Shift+Tab");
  await expect(page.getByRole("button", { name: "继续下一局" })).toBeFocused();
  await page.keyboard.press("Tab");
  await expect(page.getByRole("button", { name: "查看牌桌" })).toBeFocused();
  await page.getByRole("button", { name: "查看牌桌" }).click();
  await expect(page.locator(".result-modal")).toHaveCount(0);
  await expect(page.locator(".river-0")).toBeVisible();
  await page.getByRole("button", { name: "查看结算" }).click();
  await expect(page.locator(".result-modal")).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(page.getByRole("button", { name: "查看结算" })).toBeVisible();
  await expect(page.getByRole("button", { name: "查看结算" })).toBeFocused();
});

test("exit confirmation is keyboard accessible and returns focus", async ({ page }) => {
  await openFixture(page, snapshot());
  const exit = page.getByRole("button", { name: "退出牌桌", exact: true });
  await exit.click();
  await expect(page.getByRole("button", { name: "继续对局" })).toBeFocused();
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog")).toHaveCount(0);
  await expect(exit).toBeFocused();
});

test("preview matches red and ordinary tiles; clicking submits once without confirmation", async ({
  page,
}) => {
  const { actions } = await openFixture(page, snapshot());
  await expect(page.locator(".topbar, .game-heading")).toHaveCount(0);
  const tile = page.getByRole("button", { name: "打出 5pr", exact: true });
  await tile.hover();
  await expect(page.locator(".river .matching-tile")).toHaveCount(8);
  await page.locator(".table-mode").hover();
  await expect(page.locator(".matching-tile")).toHaveCount(0);
  await tile.focus();
  await expect(page.locator(".river .matching-tile")).toHaveCount(8);
  await tile.press("Enter");
  await expect.poll(() => actions.length).toBe(1);
  expect(actions).toEqual(["discard"]);
  // Even after HTTP acknowledgement, the old window must remain locked until polling advances it.
  await expect(tile).toBeDisabled();
  await tile.evaluate((e) => {
    e.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  });
  expect(actions).toHaveLength(1);
  await expect(page.locator(".matching-tile")).toHaveCount(0);
  await expect(page.getByRole("button", { name: /确认打出/ })).toHaveCount(0);
});

test("a failed submission can be retried", async ({ page }) => {
  const { actions, failNext } = await openFixture(page, snapshot());
  failNext();
  const tile = page.getByRole("button", { name: "打出 5pr", exact: true });
  await tile.click();
  await expect(page.getByRole("alert")).toContainText("请重试");
  await expect(tile).toBeEnabled();
  await tile.click();
  await expect.poll(() => actions.length).toBe(2);
});

test("a worker rejection unlocks the same decision window for retry", async ({
  page,
}) => {
  const table = snapshot();
  const { actions } = await openFixture(page, table);
  const tile = page.getByRole("button", { name: "打出 5pr", exact: true });
  await tile.click();
  await expect.poll(() => actions.length).toBe(1);
  await expect(tile).toBeDisabled();
  table.view_version += 1;
  table.command_error = "动作未执行，请重试";
  await expect(tile).toBeEnabled();
  await tile.click();
  await expect.poll(() => actions.length).toBe(2);
});

test("a lost action response retries the same request without duplicate commands", async ({ page }) => {
  const table = snapshot();
  await openFixture(page, table);
  const requests: any[] = [];
  await page.route("**/api/table/action", async (route) => {
    requests.push(route.request().postDataJSON());
    if (requests.length === 1) {
      // The server accepted the action, but its HTTP response was lost.
      table.view_version++;
      table.revision++;
      table.candidates = [];
      await route.abort("failed");
      return;
    }
    await route.fulfill({ json: { status: "processing" } });
  });
  await page.getByRole("button", { name: "打出 5pr", exact: true }).click();
  await expect.poll(() => requests.length, { timeout: 8000 }).toBe(2);
  expect(requests[1]).toEqual(requests[0]);
  await expect(page.getByRole("alert")).toHaveCount(0);
});

test("a late pre-command poll cannot restore an older table snapshot", async ({ page }) => {
  const table = snapshot();
  await openFixture(page, table);
  const old = structuredClone(table);
  let captured!: () => void, release!: () => void;
  const gotOld = new Promise<void>((resolve) => { captured = resolve; });
  const hold = new Promise<void>((resolve) => { release = resolve; });
  let used = false;
  await page.route("**/api/table?*", async (route) => {
    if (used) { await route.fallback(); return; }
    used = true;
    captured();
    await hold;
    await route.fulfill({ json: old });
  });
  await gotOld;
  await page.route("**/api/table/action", async (route) => {
    table.view_version++;
    table.revision++;
    table.live_remaining = 39;
    table.candidates = [];
    await route.fulfill({ json: { status: "processing" } });
  });
  await page.getByRole("button", { name: "打出 5pr", exact: true }).click();
  await expect(page.locator(".remaining strong")).toHaveText("39");
  release();
  await page.waitForTimeout(700);
  await expect(page.locator(".remaining strong")).toHaveText("39");
});

test("an ambiguous command stays locked after a later rejection until the same request resolves", async ({ page }) => {
  const table = snapshot();
  await openFixture(page, table);
  const requests: any[] = [];
  await page.route("**/api/table/action", async (route) => {
    requests.push(route.request().postDataJSON());
    if (requests.length === 1) { await route.abort("failed"); return; }
    if (requests.length === 2) { await route.fulfill({ status: 409, json: { error: "请稍候", code: "STALE_DECISION" } }); return; }
    table.view_version++;
    table.revision++;
    table.candidates = [];
    await route.fulfill({ json: { status: "processing" } });
  });
  const tile = page.getByRole("button", { name: "打出 5pr", exact: true });
  await tile.click();
  await expect.poll(() => requests.length, { timeout: 8000 }).toBeGreaterThanOrEqual(2);
  await expect(tile).toBeDisabled();
  await expect.poll(() => requests.length, { timeout: 8000 }).toBe(3);
  expect(requests[1]).toEqual(requests[0]);
  expect(requests[2]).toEqual(requests[0]);
});

test("a rejected idempotent command reports its recorded reason", async ({ page }) => {
  const table = snapshot();
  await openFixture(page, table);
  let requests = 0;
  await page.route("**/api/table/action", async (route) => {
    requests++;
    if (requests === 1) { await route.abort("failed"); return; }
    table.view_version++;
    await route.fulfill({ json: { status: "rejected", error: "动作已失效" } });
  });
  await page.getByRole("button", { name: "打出 5pr", exact: true }).click();
  await expect(page.getByRole("alert")).toContainText("动作已失效", { timeout: 8000 });
  await expect(page.getByRole("button", { name: "打出 5pr", exact: true })).toBeEnabled();
});

test("an ambiguous command stops waiting when the table has ended", async ({ page }) => {
  await openFixture(page, snapshot());
  let requests = 0, ended = false;
  await page.route("**/api/table?*", async (route) => {
    if (ended) await route.fulfill({ json: null }); else await route.fallback();
  });
  await page.route("**/api/table/action", async (route) => {
    requests++;
    if (requests === 1) { await route.abort("failed"); return; }
    ended = true;
    await route.fulfill({ status: 409, json: { error: "牌桌已结束或已更换", code: "STALE_DECISION" } });
  });
  await page.getByRole("button", { name: "打出 5pr", exact: true }).click();
  await expect(page.locator(".start-button")).toBeEnabled({ timeout: 8000 });
  await expect(page.getByRole("alert")).toContainText("当前牌桌已失效");
});

test("an expired page stops retrying and offers a refresh", async ({ page }) => {
  await openFixture(page, snapshot());
  let requests = 0;
  await page.route("**/api/table/action", async (route) => {
    requests++;
    if (requests === 1) { await route.abort("failed"); return; }
    await route.fulfill({ status: 403, json: { error: "会话令牌无效，请刷新" } });
  });
  await page.getByRole("button", { name: "打出 5pr", exact: true }).click();
  await expect(page.getByRole("alert")).toContainText("页面连接已失效", { timeout: 8000 });
  await expect(page.getByRole("button", { name: "刷新页面" })).toBeVisible();
  await expect(page.getByRole("button", { name: "打出 5pr", exact: true })).toBeDisabled();
  expect(requests).toBe(2);
});

test("an expired page exposes refresh even when a result dialog was open", async ({ page }) => {
  const table = snapshot([]);
  table.state = "round_result";
  table.result = { round_id: 0, revision: 1, kind: "draw", reason: null, winners: [], score_delta: [0, 0, 0, 0], scores: [25000, 25000, 25000, 25000], kyotaku: 0, game_over: false };
  await openFixture(page, table);
  await page.route("**/api/table/continue", (route) => route.fulfill({ status: 403, json: { error: "会话令牌无效，请刷新" } }));
  await page.getByRole("button", { name: "继续下一局" }).click();
  await expect(page.locator(".result-modal")).toHaveCount(0);
  await expect(page.getByRole("button", { name: "刷新页面" })).toBeFocused();
});

test("spectator controls remain locked until a newer post-command snapshot", async ({ page }) => {
  const table = snapshot([]);
  table.spectator = true;
  table.viewer = 0;
  table.paused = false;
  await openFixture(page, table);
  let requests = 0;
  await page.route("**/api/table/control", async (route) => {
    requests++;
    await route.fulfill({ json: { status: "processing" } });
  });
  const pause = page.getByRole("button", { name: "暂停播放" });
  await pause.click();
  await expect(pause).toBeDisabled();
  await expect(page.getByRole("button", { name: "退出牌桌", exact: true })).toBeDisabled();
  await page.waitForTimeout(700);
  await expect(pause).toBeDisabled();
  expect(requests).toBe(1);
  table.paused = true;
  table.view_version++;
  await expect(page.getByRole("button", { name: "继续播放" })).toBeEnabled();
  expect(requests).toBe(1);
});

test("a lost view-control response keeps the old projection until the same request resolves", async ({ page }) => {
  const table = snapshot([]);
  table.spectator = true;
  table.viewer = 0;
  table.paused = true;
  await openFixture(page, table);
  const requests: any[] = [];
  await page.route("**/api/table/control", async (route) => {
    requests.push(route.request().postDataJSON());
    if (requests.length === 1) { await route.abort("failed"); return; }
    table.viewer = 2;
    table.view_version++;
    table.players!.forEach((p) => { p.hand = p.seat === 2 ? ["1m"] : null; p.drawn = null; });
    await route.fulfill({ json: { status: "processing" } });
  });
  await page.getByRole("combobox", { name: "观战视角" }).selectOption("2");
  await expect(page.locator('.self-info[data-seat="0"]')).toBeVisible();
  await expect(page.locator('.self-info[data-seat="2"]')).toBeVisible({ timeout: 8000 });
  expect(requests).toHaveLength(2);
  expect(requests[1]).toEqual(requests[0]);
  await expect(page.locator(".self-hand [aria-label='1m']")).toBeVisible();
});

test("a lost continue response retries the same round command", async ({ page }) => {
  const table = snapshot([]);
  table.state = "round_result";
  table.result = { round_id: 0, revision: 1, kind: "draw", reason: null, winners: [], score_delta: [0, 0, 0, 0], scores: [25000, 25000, 25000, 25000], kyotaku: 0, game_over: false };
  await openFixture(page, table);
  const requests: any[] = [];
  await page.route("**/api/table/continue", async (route) => {
    requests.push(route.request().postDataJSON());
    if (requests.length === 1) { await route.abort("failed"); return; }
    table.state = "playing";
    table.round_id++;
    table.view_version++;
    table.result = null;
    await route.fulfill({ json: { status: "processing" } });
  });
  await page.getByRole("button", { name: "继续下一局" }).click();
  await expect(page.locator(".result-modal")).toHaveCount(0, { timeout: 8000 });
  expect(requests).toHaveLength(2);
  expect(requests[1]).toEqual(requests[0]);
});

test("failed exit remains open and can be retried", async ({ page }) => {
  await openFixture(page, snapshot());
  let attempts = 0;
  await page.route("**/api/table/close", async (route) => {
    attempts++;
    if (attempts === 1) await route.fulfill({ status: 503, json: { error: "退出失败" } });
    else await route.fulfill({ json: { status: "closed" } });
  });
  await page.getByRole("button", { name: "退出牌桌", exact: true }).click();
  const dialog = page.getByRole("dialog");
  await dialog.getByRole("button", { name: "退出牌桌", exact: true }).click();
  await expect(dialog.getByRole("alert")).toContainText("退出未确认");
  await dialog.getByRole("button", { name: "退出牌桌", exact: true }).click();
  await expect(page.locator(".start-button")).toBeEnabled();
});

test("riichi selects a legal discard and drawn tiles keep their execution identity", async ({
  page,
}) => {
  const table = snapshot([
    discard,
    { id: "riichi", kind: "A_RIICHI_DISCARD", args: ["TILE_9s", "TSUMOGIRI"] },
  ]);
  const { actions } = await openFixture(page, table);
  await page.getByRole("button", { name: "立直", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "打出 5pr", exact: true }),
  ).toBeDisabled();
  await expect(page.getByRole("button", { name: "打出 5pr", exact: true })).toHaveClass(/dim/);
  await expect(page.getByRole("button", { name: "打出 9s", exact: true })).not.toHaveClass(/dim/);
  expect(actions).toHaveLength(0);
  await page.getByRole("button", { name: "取消立直", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "打出 5pr", exact: true }),
  ).toBeEnabled();
  await page.getByRole("button", { name: "立直", exact: true }).click();
  await page.getByRole("button", { name: "打出 9s", exact: true }).click();
  await expect.poll(() => actions).toEqual(["riichi"]);
});

for (const [label, candidates, handAllowed, drawAllowed, selected] of [
  ["draw only", [{ id: "draw", kind: "A_DISCARD", args: ["TILE_5pr", "TSUMOGIRI"] }], false, true, "draw"],
  ["hand only", [{ id: "hand", kind: "A_DISCARD", args: ["TILE_5pr", "TEDASHI"] }], true, false, "hand"],
  ["both, choose drawn", [{ id: "hand", kind: "A_DISCARD", args: ["TILE_5pr", "TEDASHI"] }, { id: "draw", kind: "A_DISCARD", args: ["TILE_5pr", "TSUMOGIRI"] }], true, true, "draw"],
  ["both, choose hand", [{ id: "hand", kind: "A_DISCARD", args: ["TILE_5pr", "TEDASHI"] }, { id: "draw", kind: "A_DISCARD", args: ["TILE_5pr", "TSUMOGIRI"] }], true, true, "hand"],
] as [string, Candidate[], boolean, boolean, string][]) {
  test(`identical hand and drawn tiles retain their action identity: ${label}`, async ({ page }) => {
    const table = snapshot(candidates);
    table.players![0].drawn = "5pr";
    const { actions } = await openFixture(page, table);
    const hand = page.locator(".self-hand > button").first();
    const drawn = page.locator(".self-hand .drawn-tile button");
    if (handAllowed) await expect(hand).toBeEnabled(); else await expect(hand).toBeDisabled();
    if (drawAllowed) await expect(drawn).toBeEnabled(); else await expect(drawn).toBeDisabled();
    await (selected === "draw" ? drawn : hand).click();
    await expect.poll(() => actions).toEqual([selected]);
  });
}

test("compass follows actual winds and spectator position", async ({ page }) => {
  const table = snapshot([]);
  table.spectator = true;
  table.viewer = 0;
  table.paused = true;
  table.players!.forEach((player, i) => { player.wind = "SWNE"[i]; });
  await openFixture(page, table);
  for (let viewer = 0; viewer < 4; viewer++) {
    table.viewer = viewer;
    table.view_version++;
    table.players!.forEach((p) => { p.hand = p.seat === viewer ? ["1m"] : null; });
    await expect(page.locator(`.self-info[data-seat="${viewer}"]`)).toBeVisible();
    const labels: Record<string, string> = { E: "东", S: "南", W: "西", N: "北" };
    const expected = [2, 3, 1, 0].map((position) => labels[table.players![(viewer + position) % 4].wind]);
    await expect(page.locator(".center-compass span")).toHaveText(expected);
  }
});

for (const [kind, label, source] of [
  ["A_CHI", "吃", 3],
  ["A_PON", "碰", 1],
  ["A_RON", "荣和", 2],
  ["A_DAIMINKAN", "明杠", 3],
] as const) {
  test(`${label} identifies the source and target tile before direct submission`, async ({
    page,
  }) => {
    const tiles =
      kind === "A_RON"
        ? []
        : kind === "A_CHI"
          ? ["TILE_4p", "TILE_6p"]
          : kind === "A_PON"
            ? ["TILE_5p", "TILE_5pr"]
            : ["TILE_5p", "TILE_5p", "TILE_5pr"];
    const table = snapshot([
      { id: "claim", kind, args: [seats[source], "TILE_5p", ...tiles] },
      pass,
    ]);
    table.response_window = true;
    table.response_trigger = { source, tile: "5p", kind: "discard" };
    const { actions } = await openFixture(page, table);
    const name = ["你", "下家", "对家", "上家"][source];
    await expect(page.locator(".response-context")).toContainText(name);
    await expect(
      page.locator(".response-context [aria-label='5p']"),
    ).toBeVisible();
    await expect(page.locator(`.river-${source} .response-tile`)).toHaveCount(
      1,
    );
    await expect(page.locator(".response-tile")).toHaveCount(1);
    await page
      .getByRole("button", { name: `${label}${name}的 5p`, exact: true })
      .click();
    await expect.poll(() => actions).toEqual(["claim"]);
  });
}

test("multiple chi combinations expose consumed tiles and submit the chosen candidate", async ({
  page,
}) => {
  const table = snapshot([
    {
      id: "chi-one",
      kind: "A_CHI",
      args: ["KAMICHA", "TILE_5p", "TILE_3p", "TILE_4p"],
    },
    {
      id: "chi-two",
      kind: "A_CHI",
      args: ["KAMICHA", "TILE_5p", "TILE_4p", "TILE_6p"],
    },
    pass,
  ]);
  table.response_window = true;
  table.response_trigger = { source: 3, tile: "5p", kind: "discard" };
  const { actions } = await openFixture(page, table);
  await page.getByRole("button", { name: "吃上家的 5p", exact: true }).click();
  expect(actions).toHaveLength(0);
  await expect(page.locator(".choice")).toHaveCount(2);
  await page.keyboard.press("Escape");
  await expect(page.locator(".choice")).toHaveCount(0);
  await page.getByRole("button", { name: "吃上家的 5p", exact: true }).click();
  await expect(
    page.locator(".choice").nth(1).locator("[aria-label='6p']"),
  ).toBeVisible();
  await page.locator(".choice").nth(1).click();
  await expect.poll(() => actions).toEqual(["chi-two"]);
});

test("robbed kan identifies the player without highlighting an old discard", async ({
  page,
}) => {
  const table = snapshot([
    { id: "ron", kind: "A_RON", args: ["SHIMOCHA", "TILE_5p"] },
    pass,
  ]);
  table.response_window = true;
  table.response_trigger = { source: 1, tile: "5p", kind: "kan" };
  await openFixture(page, table);
  await expect(page.locator(".response-context")).toContainText(
    "杠牌 · 可抢杠",
  );
  await expect(page.locator(".response-tile")).toHaveCount(0);
  await expect(
    page.getByRole("button", { name: "荣和下家的 5p", exact: true }),
  ).toBeVisible();
});

test("floating controls fit desktop and mobile without covering hand, rivers or melds", async ({
  page,
}) => {
  const table = snapshot([
    {
      id: "pon",
      kind: "A_PON",
      args: ["KAMICHA", "TILE_5p", "TILE_5p", "TILE_5pr"],
    },
    {
      id: "chi",
      kind: "A_CHI",
      args: ["KAMICHA", "TILE_5p", "TILE_4p", "TILE_6p"],
    },
    {
      id: "kan",
      kind: "A_DAIMINKAN",
      args: ["KAMICHA", "TILE_5p", "TILE_5p", "TILE_5p", "TILE_5pr"],
    },
    { id: "ron", kind: "A_RON", args: ["KAMICHA", "TILE_5p"] },
    pass,
  ]);
  table.response_window = true;
  table.response_trigger = { source: 3, tile: "5p", kind: "discard" };
  for (const player of table.players!) {
    player.river = Array.from({ length: 24 }, () => ({
      tile: "5p",
      tsumogiri: false,
      riichi: false,
      called: false,
    }));
  }
  table.players![0].hand = table.players![0].hand!.slice(0, 10);
  table.players![0].drawn = null;
  table.players![0].melds = [
    { name: "MELD_PON", tiles: ["5m", "5m", "5mr"], base_tile: "5m", from: 3 },
  ];
  await openFixture(page, table);
  await expect(page.locator(".meld-source")).toHaveText("上家 · 碰");
  for (const width of [1440, 900, 390]) {
    await page.setViewportSize({ width, height: 900 });
    await page.screenshot({ path: `/tmp/majang-table-${width}.png` });
    const hand = await page.locator(".self-hand").boundingBox();
    const controls = await page.locator(".table-controls").boundingBox();
    const felt = await page.locator(".felt-table").boundingBox();
    expect(controls!.x).toBeGreaterThanOrEqual(felt!.x);
    expect(controls!.x + controls!.width).toBeLessThanOrEqual(
      felt!.x + felt!.width,
    );
    expect(controls!.y + controls!.height).toBeLessThan(hand!.y);
    for (const control of await page
      .locator(
        ".response-context, .turn-status, .action-buttons .button, .choice-panel",
      )
      .all()) {
      const active = (await control.boundingBox())!;
      for (const river of await page
        .locator(".river, .self-info .melds, .self-info .meld-source")
        .all()) {
        const box = (await river.boundingBox())!;
        const overlaps =
          active.x < box.x + box.width &&
          active.x + active.width > box.x &&
          active.y < box.y + box.height &&
          active.y + active.height > box.y;
        expect(
          overlaps,
          `${width}px: ${await control.getAttribute("class")} over ${await river.getAttribute("class")}`,
        ).toBe(false);
      }
    }
    expect(
      await page.locator("body").evaluate((e) => e.scrollWidth),
    ).toBeLessThanOrEqual(width);
  }
});
