import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";
import fs from "node:fs";
import path from "node:path";
import { spawn } from "node:child_process";

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

// Live endpoints for the demo: /api/ask?q= runs a real VSS search (vss/ask.py);
// /api/clip?source=s3://... streams a VSS segment through the backend with the team token.
function liveApi(): Plugin {
  const handler = (req: any, res: any, next: () => void) => {
    const u = new URL(req.url || "", "http://x");
    if (u.pathname.endsWith("/api/ask")) {
      const q = (u.searchParams.get("q") || "").slice(0, 200);
      if (!q.trim()) { res.statusCode = 400; return res.end("{}"); }
      const p = spawn("python3", [path.join(ROOT, "vss", "ask.py"), q], { cwd: ROOT });
      let out = "", err = "";
      p.stdout.on("data", (d) => (out += d));
      p.stderr.on("data", (d) => (err += d));
      p.on("close", (code) => {
        res.setHeader("Content-Type", "application/json");
        if (code !== 0) { res.statusCode = 502; return res.end(JSON.stringify({ error: err.slice(-300) })); }
        res.end(out);
      });
      return;
    }
    if (u.pathname.endsWith("/api/clip")) {
      const src = u.searchParams.get("source") || "";
      if (!src.startsWith("s3://")) { res.statusCode = 400; return res.end(); }
      let tok = "";
      try { tok = fs.readFileSync(path.join(ROOT, ".vss_token"), "utf8").trim(); } catch { /* */ }
      const url = `https://team-16-vss.thecosmoslabs.com/api/v1/videos/stream?source=${encodeURIComponent(src)}&token=${tok}`;
      res.setHeader("Content-Type", "video/mp4");
      res.setHeader("Cache-Control", "max-age=3600");
      const p = spawn("curl", ["-s", "-m", "60", url]);
      p.stdout.pipe(res);
      req.on("close", () => p.kill());
      return;
    }
    next();
  };
  return {
    name: "spore-live-api",
    configureServer(s) { s.middlewares.use(handler); },
    configurePreviewServer(s) { s.middlewares.use(handler); },
  };
}

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
  base: "./",
  plugins: [react(), liveApi(), staticMounts()],
  server: { port: 5173, host: true },
});
