import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";
import fs from "node:fs";
import path from "node:path";

// Serves ../data, ../results (repo root) and ./fixtures straight from disk.
// Missing files return a real 404 (no SPA fallback), and mp4s support Range
// requests so <video> looping / seeking / sync works.
const ROOT = path.resolve(__dirname, "..");
const MOUNTS: Record<string, string> = {
  "/data/": path.join(ROOT, "data"),
  "/results/": path.join(ROOT, "results"),
  "/fixtures/": path.join(__dirname, "fixtures"),
};
const TYPES: Record<string, string> = {
  ".json": "application/json",
  ".mp4": "video/mp4",
  ".webm": "video/webm",
  ".jpg": "image/jpeg",
  ".png": "image/png",
};

function staticMounts(): Plugin {
  const handler = (req: any, res: any, next: () => void) => {
    const url = decodeURIComponent((req.url || "").split("?")[0]);
    const prefix = Object.keys(MOUNTS).find((p) => url.startsWith(p));
    if (!prefix) return next();
    const base = MOUNTS[prefix];
    const file = path.normalize(path.join(base, url.slice(prefix.length)));
    if (!file.startsWith(base)) { res.statusCode = 403; return res.end(); }
    let stat: fs.Stats;
    try { stat = fs.statSync(file); if (!stat.isFile()) throw 0; }
    catch {
      // Missing JSON -> 200 `null` so the UI shows an empty state without console 404 noise.
      if (file.endsWith(".json")) { res.setHeader("Content-Type", "application/json"); res.setHeader("Cache-Control", "no-store"); return res.end("null"); }
      res.statusCode = 404; return res.end("not found");
    }
    res.setHeader("Content-Type", TYPES[path.extname(file)] || "application/octet-stream");
    res.setHeader("Cache-Control", "no-store");
    res.setHeader("Accept-Ranges", "bytes");
    const range = req.headers.range as string | undefined;
    if (range) {
      const m = /bytes=(\d*)-(\d*)/.exec(range);
      const start = m && m[1] ? +m[1] : 0;
      const end = m && m[2] ? Math.min(+m[2], stat.size - 1) : stat.size - 1;
      if (start >= stat.size) { res.statusCode = 416; return res.end(); }
      res.statusCode = 206;
      res.setHeader("Content-Range", `bytes ${start}-${end}/${stat.size}`);
      res.setHeader("Content-Length", end - start + 1);
      return fs.createReadStream(file, { start, end }).pipe(res);
    }
    res.setHeader("Content-Length", stat.size);
    fs.createReadStream(file).pipe(res);
  };
  return {
    name: "spore-static-mounts",
    configureServer(s) { s.middlewares.use(handler); },
    configurePreviewServer(s) { s.middlewares.use(handler); },
  };
}

export default defineConfig({
  plugins: [react(), staticMounts()],
  server: { port: 5173, host: true },
});
