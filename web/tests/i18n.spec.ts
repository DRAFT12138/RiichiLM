import { expect, test } from "@playwright/test";
import { browserLocale, translate } from "../src/i18n";

for (const scenario of [
  { browser: "en-US", language: "en", heading: "Choose your game", start: "Start playing", label: "Language" },
  { browser: "ja-JP", language: "ja", heading: "対局形式を選ぶ", start: "対局開始", label: "表示言語" },
  { browser: "zh-TW", language: "zh-CN", heading: "选择你的牌局", start: "开始对局", label: "界面语言" },
  { browser: "de-DE", language: "en", heading: "Choose your game", start: "Start playing", label: "Language" },
]) {
  test(`browser ${scenario.browser} selects ${scenario.language} without a stored preference`, async ({ browser }) => {
    const context = await browser.newContext({ locale: scenario.browser, viewport: { width: 390, height: 844 } });
    const page = await context.newPage();
    await page.goto("/");
    await expect(page.locator("html")).toHaveAttribute("lang", scenario.language);
    await expect(page).toHaveTitle(/RiichiLM/);
    await expect(page.getByRole("heading", { name: scenario.heading, exact: true })).toBeVisible();
    await expect(page.getByRole("button", { name: scenario.start, exact: true })).toBeEnabled();
    await expect(page.getByRole("combobox", { name: scenario.label, exact: true })).toHaveValue("");
    expect(await page.locator("body").evaluate((element) => element.scrollWidth)).toBeLessThanOrEqual(390);
    await context.close();
  });
}

test("locale negotiation, interpolation and API error codes preserve their meaning", () => {
  expect(browserLocale(["fr-FR", "ja-JP", "en-US"])).toBe("ja");
  expect(browserLocale(["zh-Hant-TW"])).toBe("zh-CN");
  expect(browserLocale([])).toBe("en");
  expect(translate("{seat}的牌河", "en", { seat: "East seat" })).toBe("East seat discards");
  expect(translate("{wind}{round}局 · {honba}本场", "ja", { wind: "東", round: "一", honba: 2 })).toBe("東一局 · 2本場");
  expect(translate("STALE_DECISION: 动作窗口已过期", "en")).toBe("STALE_DECISION: The action window has expired.");
  expect(translate("请求失败 (503)", "ja")).toBe("リクエスト失敗 (503)");
  expect(translate("user-supplied model name", "ja")).toBe("user-supplied model name");
});

test("manual locale persists, Auto follows the browser again, and disabled storage still works", async ({ browser }) => {
  const context = await browser.newContext({ locale: "ja-JP" });
  const page = await context.newPage();
  await page.goto("/");
  await page.getByRole("combobox", { name: "表示言語", exact: true }).selectOption("en");
  await expect(page.locator("html")).toHaveAttribute("lang", "en");
  await page.reload();
  await expect(page.getByRole("heading", { name: "Choose your game", exact: true })).toBeVisible();
  await page.getByRole("combobox", { name: "Language", exact: true }).selectOption("");
  await expect(page.locator("html")).toHaveAttribute("lang", "ja");
  await page.reload();
  await expect(page.locator("html")).toHaveAttribute("lang", "ja");
  await context.close();
  const blocked = await browser.newContext({ locale: "en-US" });
  await blocked.addInitScript(() => {
    Storage.prototype.getItem = () => { throw new DOMException("Blocked", "SecurityError"); };
    Storage.prototype.setItem = () => { throw new DOMException("Blocked", "SecurityError"); };
  });
  const other = await blocked.newPage();
  await other.goto("/");
  await other.getByRole("combobox", { name: "Language", exact: true }).selectOption("ja");
  await expect(other.getByRole("heading", { name: "対局形式を選ぶ", exact: true })).toBeVisible();
  await blocked.close();
});

test("switch language at a live table without replacing it or leaking other hands", async ({ browser }) => {
  const context = await browser.newContext({ locale: "en-US", viewport: { width: 390, height: 844 } });
  const page = await context.newPage();
  let creates = 0;
  page.on("request", (request) => { if (request.url().endsWith("/api/table") && request.method() === "POST") creates++; });
  await page.goto("/");
  await page.getByRole("button", { name: "Watch AI vs AI", exact: true }).click();
  await page.getByRole("button", { name: /Single round/ }).click();
  await page.getByRole("button", { name: "Start watching", exact: true }).click();
  await expect(page.getByRole("button", { name: "Pause playback", exact: true })).toBeEnabled({ timeout: 30000 });
  await page.getByRole("button", { name: "Pause playback", exact: true }).click();
  await expect(page.getByRole("button", { name: "Resume playback", exact: true })).toBeEnabled();
  await page.getByRole("combobox", { name: "Language", exact: true }).selectOption("ja");
  await expect(page.getByRole("button", { name: "再生を再開", exact: true })).toBeEnabled();
  await expect(page.getByRole("combobox", { name: "観戦視点", exact: true })).toHaveValue("0");
  await expect(page.locator('.river-0[aria-label="東席の河"]')).toBeVisible();
  await expect(page.locator(".self-hand button")).toHaveCount(0);
  await page.getByRole("button", { name: "AIの判断の要約", exact: true }).click();
  await expect(page.locator(".decision-panel")).toContainText("確定済みの判断");
  expect(creates).toBe(1);
  expect(await page.locator("body").evaluate((element) => element.scrollWidth)).toBeLessThanOrEqual(390);
  await page.getByRole("button", { name: "卓から退出", exact: true }).click();
  await expect(page.getByRole("dialog")).toContainText("現在の対局は保存されません");
  await context.close();
});

test("Auto reacts to browser-language changes while a manual choice stays selected", async ({ browser }) => {
  const context = await browser.newContext({ locale: "en-US" });
  const page = await context.newPage();
  await page.goto("/");
  await page.getByRole("combobox", { name: "Language", exact: true }).selectOption("ja");
  await page.evaluate(() => {
    Object.defineProperty(navigator, "languages", { configurable: true, value: ["zh-CN"] });
    window.dispatchEvent(new Event("languagechange"));
  });
  await expect(page.locator("html")).toHaveAttribute("lang", "ja");
  await page.getByRole("combobox", { name: "表示言語", exact: true }).selectOption("");
  await expect(page.locator("html")).toHaveAttribute("lang", "zh-CN");
  await page.evaluate(() => {
    Object.defineProperty(navigator, "languages", { configurable: true, value: ["fr-FR", "en-GB"] });
    window.dispatchEvent(new Event("languagechange"));
  });
  await expect(page.locator("html")).toHaveAttribute("lang", "en");
  await context.close();
});

for (const locale of ["en-US", "ja-JP"]) {
  test(`results, tile labels and scoring are localized in ${locale}`, async ({ browser }) => {
    const context = await browser.newContext({ locale });
    const page = await context.newPage();
    const english = locale === "en-US";
    const table = {
      table_id: "localized-result", page_id: "fixture", view_version: 5, state: "finished", mode: "single", busy: false,
      round_id: 0, revision: 2, round_wind: "E", round_number: 1, honba: 1, kyotaku: 2,
      live_remaining: 0, dora_indicators: ["3s"], active_seats: [], response_window: false, response_trigger: null,
      window: null, candidates: [], thinking: null, cross_mode: false,
      players: [0, 1, 2, 3].map((seat) => ({ seat, wind: "ESWN"[seat], score: 25000, dealer: seat === 0,
        riichi: false, hand: seat === 0 ? ["5pr"] : null, drawn: null, hand_count: 13, melds: [], river: [] })),
      result: { kind: "win", winners: [{ actor: 0, target: 0, tsumo: true, hand: ["5pr"], winning_tile: "5p",
        melds: [], ura_indicators: [], han: 3, fu: 40, yakuman_units: 0, yaku: ["Y_RIICHI", "Y_PINFU"], dora: 1, aka: 1, ura: 0 }],
        score_delta: [0, 0, 0, 0], game_over: true },
      final: { scores: [25000, 25000, 25000, 25000], ranks: [1, 2, 3, 4] },
    };
    await page.route("**/api/**", async (route) => {
      const path = new URL(route.request().url()).pathname;
      const json = path === "/api/bootstrap" ? { token: "fixture", generation: 1, device: "cpu" }
        : path === "/api/catalog" ? { runs: [], errors: [] }
        : path === "/api/table" ? route.request().method() === "POST" ? { table_id: table.table_id } : table : {};
      await route.fulfill({ json });
    });
    await page.goto("/");
    await page.locator(".start-button").click();
    await expect(page.getByRole("heading", { name: english ? "Game complete" : "対局終了", exact: true })).toBeVisible();
    await expect(page.locator(".winner-title")).toContainText(english ? "3 han 40 fu" : "3翻 40符");
    await expect(page.locator(".yaku-tags")).toContainText(english ? "Riichi" : "リーチ");
    await expect(page.locator(".yaku-tags")).toContainText(english ? "Red dora" : "赤ドラ");
    await expect(page.locator(".result-footnote")).toContainText(english ? "Unclaimed riichi sticks: 2 (2000 points)" : "未回収の供託：2本（2000点）");
    await expect(page.locator(".winner-hand .tile").first()).toHaveAttribute("title", english ? "Red 5 pin" : "赤五筒");
    await context.close();
  });
}
