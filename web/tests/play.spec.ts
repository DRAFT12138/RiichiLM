import { expect, test } from "@playwright/test";

test("pure AI spectator pauses, steps, changes perspective and shows decision notes", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/");
  await expect(page.locator(".start-button")).toBeEnabled();
  await page.getByRole("button", { name: "纯 AI 观战" }).click();
  await expect(page.locator(".opponent-card")).toHaveCount(4);
  await page.getByRole("button", { name: /单局练习/ }).click();
  const created = page.waitForRequest((r) => r.url().endsWith("/api/table") && r.method() === "POST");
  await page.getByRole("button", { name: "开始观战" }).click();
  const config = (await created).postDataJSON();
  expect(config.spectator).toBe(true);
  expect(config.opponents).toEqual(["random", "random", "random", "random"]);
  await expect(page.getByRole("button", { name: "暂停播放" })).toBeEnabled({ timeout: 30000 });
  await expect(page.locator(".self-hand button")).toHaveCount(0);
  await page.getByRole("button", { name: "暂停播放" }).click();
  await expect(page.getByRole("button", { name: "继续播放" })).toBeEnabled();
  const snapshot = () => page.evaluate(async (pageId) => {
    const b = await fetch("/api/bootstrap").then((r) => r.json());
    return fetch(`/api/table?page_id=${pageId}`, { headers: { "X-Session-Token": b.token } }).then((r) => r.json());
  }, config.page_id);
  const paused = await snapshot();
  await page.waitForTimeout(900);
  expect((await snapshot()).revision).toBe(paused.revision);
  await page.getByRole("combobox", { name: "观战视角" }).selectOption("2");
  await expect(page.locator(".self-info")).toContainText("西席 · 观战视角");
  const viewed = await snapshot();
  expect(viewed.players[2].hand).not.toBeNull();
  expect(viewed.players.filter((p: { seat: number }) => p.seat !== 2).every((p: { hand: unknown; drawn: unknown }) => p.hand === null && p.drawn === null)).toBe(true);
  await expect(page.locator('.river-0[aria-label="西席的牌河"]')).toHaveCount(1);
  await page.getByRole("button", { name: "单步", exact: true }).click();
  await expect.poll(async () => (await snapshot()).revision).toBeGreaterThan(paused.revision);
  await expect(page.getByRole("button", { name: "单步", exact: true })).toBeEnabled();
  for (const seat of [1, 3, 0]) {
    await page.getByRole("combobox", { name: "观战视角" }).selectOption(String(seat));
    await expect.poll(async () => (await snapshot()).viewer).toBe(seat);
    const view = await snapshot();
    expect(view.players[seat].hand).not.toBeNull();
    expect(view.players.filter((p: { seat: number }) => p.seat !== seat).every((p: { hand: unknown; drawn: unknown }) => p.hand === null && p.drawn === null)).toBe(true);
  }
  await page.getByRole("button", { name: "AI决策说明" }).click();
  await expect(page.locator(".decision-note").first()).toContainText("等概率采样");
  await page.setViewportSize({ width: 390, height: 844 });
  const controls = await page.locator(".spectator-controls").boundingBox();
  expect(controls!.x).toBeGreaterThanOrEqual(0);
  expect(controls!.x + controls!.width).toBeLessThanOrEqual(390);
  await expect(page.getByRole("button", { name: "继续播放" })).toBeEnabled();
  expect(errors).toEqual([]);
});

test("LLM opponents can be selected independently for several seats", async ({ page }) => {
  await page.route("**/api/catalog", (route) => route.fulfill({
    json: { runs: [], errors: [], llms: [
      { id: "llm:first", name: "通用模型一", model: "model-a" },
      { id: "llm:second", name: "通用模型二", model: "model-b" },
    ] },
  }));
  await page.goto("/");
  await expect(page.locator(".start-button")).toBeEnabled();
  await page.locator("#opponent-0").selectOption("llm:first");
  await page.locator("#opponent-1").selectOption("llm:second");
  await page.locator("#opponent-2").selectOption("llm:first");
  await expect(page.locator("#opponent-0")).toHaveValue("llm:first");
  await expect(page.locator("#opponent-1")).toHaveValue("llm:second");
  await expect(page.locator("#opponent-2")).toHaveValue("llm:first");
  await expect(page.locator(".opponent-note").first()).toContainText("通用 LLM · 通用模型一");
  await expect(page.locator(".start-button")).toBeEnabled();
});

test("real random single game, action selection and final result", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/");
  await expect(page.locator(".start-button")).toBeEnabled();
  await page.getByRole("button", { name: /单局练习/ }).click();
  await page.locator(".start-button").click();
  await expect(page.locator(".self-hand")).toBeVisible({ timeout: 30000 });
  await expect(page.locator("aside, .sidebar")).toHaveCount(0);
  await expect(page.locator(".self-hand .tile-face").first()).toBeVisible();
  await expect
    .poll(async () =>
      page
        .locator(".self-hand img")
        .evaluateAll((images) =>
          images.every(
            (image) =>
              (image as HTMLImageElement).complete &&
              (image as HTMLImageElement).naturalWidth > 0,
          ),
        ),
    )
    .toBe(true);

  const gameId = await page.evaluate(async () => {
    const b = await fetch("/api/bootstrap").then((r) => r.json());
    return b.generation;
  });
  for (let i = 0; i < 180; i++) {
    if (await page.locator(".result-modal").count()) break;
    const tile = page.locator(".self-hand button:not(:disabled)").first();
    const pass = page.getByRole("button", { name: "跳过", exact: true });
    if (await tile.count()) {
      await tile.click();
    } else if (await pass.count()) await pass.click();
    await expect(page.locator(".error-banner")).toHaveCount(0);
    await page.waitForTimeout(600);
  }
  await expect(page.locator(".result-modal")).toBeVisible();
  await expect(page.getByText("未领取供托：", { exact: false })).toBeVisible();
  await expect(page.locator(".score-results>div")).toHaveCount(4);
  expect(errors).toEqual([]);
  expect(gameId).toBeGreaterThan(0);
  await page.getByRole("button", { name: "返回配置", exact: true }).click();
  await expect(page.locator(".start-button")).toBeEnabled();
});

test("refresh discards old table and exit starts from config", async ({
  page,
}) => {
  await page.goto("/");
  await expect(page.locator(".start-button")).toBeEnabled();
  await page.locator(".start-button").click();
  await expect(page.locator(".self-hand")).toBeVisible({ timeout: 30000 });
  await page.reload();
  await expect(page.locator(".start-button")).toBeEnabled();
  await expect(page.locator(".self-hand")).toHaveCount(0);
  await page.locator(".start-button").click();
  await expect(page.locator(".self-hand")).toBeVisible({ timeout: 30000 });
  await page.getByRole("button", { name: "退出牌桌", exact: true }).click();
  await page
    .getByRole("dialog")
    .getByRole("button", { name: "退出牌桌", exact: true })
    .click();
  await expect(page.locator(".start-button")).toBeEnabled();
  await expect(page.locator(".self-hand")).toHaveCount(0);
});

test("desktop and narrow viewport keep main controls within screen", async ({
  page,
}) => {
  for (const width of [1280, 900, 390]) {
    await page.setViewportSize({ width, height: 1000 });
    await page.goto("/");
    await expect(page.locator(".start-button")).toBeEnabled();
    expect(
      await page.locator("body").evaluate((e) => e.scrollWidth),
    ).toBeLessThanOrEqual(width);
    await page.locator(".start-button").click();
    await expect(page.locator(".self-hand")).toBeVisible({ timeout: 30000 });
    expect(
      await page.locator("body").evaluate((e) => e.scrollWidth),
    ).toBeLessThanOrEqual(width);
    const hand = await page.locator(".self-hand").boundingBox();
    expect(hand!.x).toBeGreaterThanOrEqual(0);
    expect(hand!.x + hand!.width).toBeLessThanOrEqual(width);
  }
});

test("a late polling response cannot reopen a table after exit", async ({
  page,
}) => {
  await page.goto("/");
  await expect(page.locator(".start-button")).toBeEnabled();
  await page.locator(".start-button").click();
  await expect(page.locator(".self-hand")).toBeVisible({ timeout: 30000 });
  let release!: () => void, captured!: () => void;
  const hold = new Promise<void>((resolve) => {
    release = resolve;
  });
  const gotResponse = new Promise<void>((resolve) => {
    captured = resolve;
  });
  let used = false;
  await page.route("**/api/table?*", async (route) => {
    if (used) {
      await route.continue();
      return;
    }
    used = true;
    const response = await route.fetch();
    captured();
    await hold;
    await route.fulfill({ response });
  });
  await gotResponse;
  await page.getByRole("button", { name: "退出牌桌", exact: true }).click();
  await page
    .getByRole("dialog")
    .getByRole("button", { name: "退出牌桌", exact: true })
    .click();
  await expect(page.locator(".start-button")).toBeEnabled();
  release();
  await page.waitForTimeout(1200);
  await expect(page.locator(".start-button")).toBeEnabled();
  await expect(page.locator(".self-hand")).toHaveCount(0);
});
