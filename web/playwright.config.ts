import { defineConfig } from "@playwright/test";
const port = Number(process.env.WEB_E2E_PORT || 8766);
const python = process.env.WEB_E2E_PYTHON || ".venv/bin/python";
export default defineConfig({
  testDir: "./tests",
  workers: 1,
  timeout: 120000,
  use: {
    locale: "zh-CN",
    baseURL: `http://127.0.0.1:${port}`,
    viewport: { width: 1440, height: 1100 },
    launchOptions: process.env.CHROMIUM_EXECUTABLE
      ? { executablePath: process.env.CHROMIUM_EXECUTABLE }
      : {},
  },
  webServer: {
    command: `cd .. && CUDA_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES="" "${python}" -m mahjong web-play --models-root /tmp/mahjong-web-e2e-no-models --weights-root /tmp/mahjong-web-e2e-no-weights --port ${port} --lease-seconds 120`,
    url: `http://127.0.0.1:${port}`,
    reuseExistingServer: false,
    timeout: 30000,
  },
});
