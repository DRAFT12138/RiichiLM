import { expect, test } from "@playwright/test";

test("custom weights show their filename and can be assigned in each language", async ({ page }) => {
  const cp = { id: "weights:one", checkpoint_id: "my-policy.pt", attempt_id: "", step: 0,
    created_at: null, selectable: true, status: "待加载验证" };
  await page.route("**/api/**", async route => {
    const path = new URL(route.request().url()).pathname;
    await route.fulfill({ json: path === "/api/bootstrap" ? { token: "fixture", generation: 1, device: "cpu" }
      : path === "/api/catalog" ? { runs: [{ run_id: "weights:my-policy.pt", experiment: "my-policy",
        created_at: null, training_type: "Weights", checkpoints: [cp], attempts: [], errors: [] }], errors: [] } : {} });
  });
  await page.goto("/");
  for (const [locale, label] of [["en", "Custom weights"], ["ja", "カスタム重み"], ["zh-CN", "自定义权重"]]) {
    await page.locator(".topbar .language-switcher select").selectOption(locale);
    await expect(page.locator("#opponent-0 option[value='weights:one']")).toHaveText(`${label} · my-policy.pt`);
    await expect(page.locator(".run-heading span")).toHaveText(label);
    await expect(page.locator(".checkpoint code")).toHaveText("my-policy.pt");
    await expect(page.locator(".checkpoint b, .run-heading code")).toHaveCount(0);
    await page.locator(".assign-buttons button").first().click();
    await expect(page.locator("#opponent-0")).toHaveValue(cp.id);
  }
});
