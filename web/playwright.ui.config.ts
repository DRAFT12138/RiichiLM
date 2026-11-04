import { defineConfig } from "@playwright/test";
const port = Number(process.env.WEB_UI_PORT || 5178);
/** Frontend-only regression tests; API fixtures need no Python model runtime. */
export default defineConfig({
  testDir: "./tests",
  testMatch: "interactions.spec.ts",
  workers: 1,
  timeout: 60000,
  expect: { timeout: 15000 },
  use: {
    baseURL: `http://127.0.0.1:${port}`,
    locale: "zh-CN",
    viewport: { width: 1440, height: 1100 },
    launchOptions: process.env.CHROMIUM_EXECUTABLE ? { executablePath: process.env.CHROMIUM_EXECUTABLE } : {},
  },
  webServer: {
    command: `node node_modules/vite/bin/vite.js preview --host 127.0.0.1 --port ${port}`,
    url: `http://127.0.0.1:${port}`,
    timeout: 30000,
  },
});
