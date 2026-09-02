import { createRequire } from "node:module";
import { fileURLToPath, URL } from "node:url";

import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const require = createRequire(import.meta.url);

// Tailwind v4 has no config file and no PostCSS step: the plugin below is what
// makes `@import "tailwindcss"` in src/index.css compile. Without it the app
// renders unstyled, which is exactly what was happening before this file existed.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
      events: require.resolve("events/"),
    },
  },
  server: {
    host: "0.0.0.0",
    port: Number(process.env.PORT ?? 5173),
  },
  build: {
    outDir: "dist",
    sourcemap: true,
  },
  worker: {
    format: "es",
  },
  optimizeDeps: {
    include: [
      "dicom-parser",
      "@cornerstonejs/codec-charls/decodewasmjs",
      "@cornerstonejs/codec-libjpeg-turbo-8bit/decodewasmjs",
      "@cornerstonejs/codec-openjpeg/decodewasmjs",
      "@cornerstonejs/codec-openjph/wasmjs",
    ],
  },
});
