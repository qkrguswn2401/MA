import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

// Dev server proxies API calls to the FastAPI backend so the browser sees one origin
// (no CORS). Override the target with VITE_API_TARGET in a .env file if the backend
// isn't on :8000.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, ".", "VITE_");
  const target = env.VITE_API_TARGET || "http://localhost:8000";
  return {
    plugins: [react()],
    server: {
      host: "0.0.0.0",   // bind all interfaces (external/VESSL access), not just localhost
      port: 5173,
      strictPort: true,   // fail fast if 5173 is taken rather than silently hopping ports
      proxy: {
        // /ask, /ask/stream (SSE), /health, /info, /datasets all live on the FastAPI backend
        "/ask": { target, changeOrigin: true },
        "/health": { target, changeOrigin: true },
        "/info": { target, changeOrigin: true },
        "/datasets": { target, changeOrigin: true },
      },
    },
    build: { outDir: "dist" },
  };
});
