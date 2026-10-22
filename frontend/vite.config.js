import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const apiTarget = process.env.NOVA_API_PROXY_TARGET || "http://localhost:8001";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: apiTarget,
        changeOrigin: true,
      },
    },
  },
});
