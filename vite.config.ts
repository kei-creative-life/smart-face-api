import path from "path";
import { defineConfig } from "vite";

export default defineConfig({
  build: {
    lib: {
      entry: path.resolve(__dirname, "src/packages/main.ts"),
      name: "SmartFaceAPI",
      formats: ["es", "cjs", "umd"],
      fileName: (format) => `smart-face-api.${format}.js`,
    },
  },
});
