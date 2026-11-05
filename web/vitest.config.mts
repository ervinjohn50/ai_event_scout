import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [react()],
  resolve: { tsconfigPaths: true },
  test: {
    environment: "jsdom",
    // A zone west of UTC, so date bugs show up on any computer.
    env: { TZ: "America/Chicago" },
    setupFiles: ["./vitest.setup.ts"],
  },
});
