import path from "path";
import { fileURLToPath } from "url";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";
import { viteSingleFile } from "vite-plugin-singlefile";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// https://vite.dev/config/
export default defineConfig({
  cacheDir: ".vite",
  plugins: [react(), tailwindcss(), viteSingleFile()],
  resolve: {
    preserveSymlinks: true,
    alias: {
      "@": path.resolve(__dirname, "src"),
      "/src": path.resolve(__dirname, "src"),
      "/main.entry.tsx": path.resolve(__dirname, "main.entry.tsx"),
      "/src/main.tsx": path.resolve(__dirname, "src/main.tsx"),
    },
  },
});
