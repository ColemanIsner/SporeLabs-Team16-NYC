import { useCallback, useEffect, useRef, useState, type FormEvent, type ReactNode } from "react";
import { url, useJSON, getJSON, type Coverage, type GapReport, type Eval, type SeverityCurve, type FixRow, type LoopEntry } from "./data";
import "./story.css";

// Step-by-step demo. One idea per screen; → / space to advance, ← to go back.
// The full dashboard is still at ?full=1.

const SEED = "i24_scene1_p1c2_00";
const HERO = `${SEED}__phys_fog_s04`;
const WEAVE = "https://wandb.ai/colemanisner-sporelabs/sporelabs-hackathon/weave";
const LINE = (
  <>
    <span>Other tools find what your video archive is missing.</span>{" "}
    <span className="hl">Spore grows the data to fill it, and shows where your AI goes blind.</span>
  </>
);

const GROW = [
  { src: `data/seeds/${SEED}.mp4`, label: "Real", sub: "clear day", real: true },
  { src: `data/synthetic/${SEED}__phys_fog_s07.mp4`, label: "Fog", sub: "weather layer" },
  { src: `data/synthetic/${SEED}__phys_rain_s07.mp4`, label: "Rain", sub: "weather layer" },
  { src: `data/synthetic/${SEED}__phys_snow_s07.mp4`, label: "Snow", sub: "weather layer" },
  { src: `data/synthetic/${SEED}__clear_night_light__cw0.5.mp4`, label: "Night", sub: "NVIDIA Cosmos Transfer" },
];

// Fallback scene grouping when results/inventory.json isn't there yet.
const SCENES: { name: string; cams: string[] }[] = [
  { name: "Highway cameras", cams: ["i24_cam-1"] },
  { name: "City street cameras", cams: ["nyc_streets_cam-1", "nyc_streets_cam-2", "sf_streets_cam-1", "sf_streets_cam-2", "sf_streets_cam-3", "sf_streets_cam-4", "sf_streets_cam-5"] },
  { name: "Dashcam & bike", cams: ["pie_cam-3", "nyc_bike_gopro-1"] },
  { name: "Residential street", cams: ["neighborhood_cam-1"] },
  { name: "Warehouse & indoor", cams: ["sdg_warehouse_cam-2", "smartspace_cam-1"] },
];

type Inventory = { scene_type: string; label?: string; n_clips: number; cameras?: string[] }[];
type Candidate = {
  id?: string; label?: string; condition?: string; scene_type?: string; why?: string; rank?: number;
  real_clips_highway?: number; real_clips_all?: number; priority?: number; score?: number;
  scene_label?: string; real_clips?: number; scene_clips?: number; mean_recall?: number | null;
};

const rv = (i: number) => ({ className: "rv", style: { animationDelay: `${0.15 + i * 0.22}s` } });

function Video({ src, className }: { src: string; className?: string }) {
  return <video className={className} src={url(src)} autoPlay muted loop playsInline />;
}

export default function Story() {
  const coverage = useJSON<Coverage & { cameras?: Record<string, number> }>("results/coverage.json");
  const inventory = useJSON<Inventory>("results/inventory.json");
  const report = useJSON<GapReport & { candidates?: Candidate[] }>("results/gap_report.json", 5000);
  const curve = useJSON<SeverityCurve>("results/severity_curve.json");
  const headline = useJSON<{ by_condition?: Record<string, { mean: number }> }>("results/headline.json");
  const fix = useJSON<FixRow[]>("results/fix.json", 5000);
  const loop = useJSON<LoopEntry[]>("results/loop_log.json", 5000);
  const [seedEval, setSeedEval] = useState<Eval | null>(null);
  const [heroEval, setHeroEval] = useState<Eval | null>(null);
  useEffect(() => {
    getJSON<Eval>(`results/evals/${SEED}.json`).then(setSeedEval);
    getJSON<Eval>(`results/evals/${HERO}.json`).then(setHeroEval);
  }, []);

  const steps: { key: string; render: () => ReactNode }[] = [
    { key: "intro", render: () => <Intro /> },
    { key: "inventory", render: () => <Inv coverage={coverage} inventory={inventory} /> },
    { key: "missing", render: () => <Missing coverage={coverage} /> },
    { key: "matters", render: () => <Matters report={report} /> },
    { key: "grow", render: () => <Grow /> },
    { key: "test", render: () => <Test seed={seedEval} hero={heroEval} /> },
    { key: "blind", render: () => <Blind curve={curve} headline={headline} /> },
    ...((fix ?? []).some((r) => r.after > r.before) ? [{ key: "fix", render: () => <Fix fix={fix} /> }] : []),
    { key: "again", render: () => <Again report={report} loop={loop} /> },
    { key: "ask", render: () => <Ask /> },
  ];

  const initial = Math.max(0, steps.findIndex((s) => s.key === location.hash.slice(1)));
  const [i, setI] = useState(initial);
  const go = useCallback((d: number) => setI((x) => Math.min(steps.length - 1, Math.max(0, x + d))), [steps.length]);
  useEffect(() => { history.replaceState(null, "", `#${steps[i].key}`); }, [i]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    const onHash = () => { const k = steps.findIndex((s) => s.key === location.hash.slice(1)); if (k >= 0) setI(k); };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.closest?.("input, textarea")) return;
      if (["ArrowRight", " ", "Enter", "PageDown"].includes(e.key)) { e.preventDefault(); go(1); }
      if (["ArrowLeft", "PageUp", "Backspace"].includes(e.key)) { e.preventDefault(); go(-1); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [go]);

  return (
    <div className="story">
      <header className="st-top">
        <button className="st-brand" onClick={() => setI(0)}>
          Spore
        </button>
        {steps[i].key !== "intro" && <LoopMap active={STEP_NODE[steps[i].key] ?? -1} />}
        <nav className="st-dots">
          {steps.map((s, k) => (
            <button key={s.key} aria-label={s.key} className={k === i ? "on" : k < i ? "done" : ""} onClick={() => setI(k)} />
          ))}
        </nav>
      </header>
      <main className="st-stage" key={steps[i].key}>{steps[i].render()}</main>
      <footer className="st-nav">
        <button className="st-back" onClick={() => go(-1)} disabled={i === 0}>←</button>
        {i < steps.length - 1 ? (
          <button className="st-next" onClick={() => go(1)}>{i === 0 ? "Start" : "Next"} →</button>
        ) : (
          <button className="st-next" onClick={() => setI(0)}>Start over</button>
        )}
      </footer>
    </div>
  );
}

// ---- live VSS search (vite middleware -> vss/ask.py) ----
type AskHit = { camera_id?: string; score: number; highway: boolean; caption: string; source?: string; shows_it: boolean; object_counts?: string | null };
type AskRes = { query: string; seconds: number; top_k: number; n_shows_it: number; hits: AskHit[]; error?: string };
const API = import.meta.env.BASE_URL.endsWith("/") ? import.meta.env.BASE_URL : import.meta.env.BASE_URL + "/";
const askCache = new Map<string, Promise<AskRes>>();
function askVSS(q: string): Promise<AskRes> {
  if (!askCache.has(q)) {
    const p = fetch(`${API}api/ask?q=${encodeURIComponent(q)}`).then((r) => r.json());
    p.catch(() => askCache.delete(q));
    askCache.set(q, p);
  }
  return askCache.get(q)!;
}
const clipUrl = (src?: string) => (src ? `${API}api/clip?source=${encodeURIComponent(src)}` : undefined);
function counts(oc?: string | null): string {
  try {
    const o = JSON.parse(oc || "{}") as Record<string, number>;
    return Object.entries(o).sort((a, b) => b[1] - a[1]).slice(0, 3).map(([k, v]) => `${v} ${k}`).join(", ") || "nothing";
  } catch { return "—"; }
}
function useElapsed(on: boolean) {
  const [t, setT] = useState(0);
  useEffect(() => {
    if (!on) return;
    const t0 = Date.now(); setT(0);
    const id = setInterval(() => setT((Date.now() - t0) / 1000), 100);
    return () => clearInterval(id);
  }, [on]);
  return t;
}

type SponsorKey = "vast" | "nvidia" | "wandb" | "cursor";
const SPONSOR: Record<SponsorKey, string> = { vast: "VAST Data", nvidia: "NVIDIA", wandb: "W&B · CoreWeave", cursor: "Cursor" };
function Powered({ by, what, call, delay = 1.6 }: { by: SponsorKey[]; what: ReactNode; call?: ReactNode; delay?: number }) {
  return (
    <div className="st-pw rv" style={{ animationDelay: `${delay}s` }}>
      {by.map((b) => <span key={b} className={`sp sp-${b}`}>{SPONSOR[b]}</span>)}
      <span className="w">{what}</span>
      {call && <code>{call}</code>}
    </div>
  );
}

function Title({ kicker, children }: { kicker: string; children: ReactNode }) {
  return (
    <div className="st-title">
      <div className="st-kicker rv">{kicker}</div>
      <h1 className="rv" style={{ animationDelay: "0.05s" }}>{children}</h1>
    </div>
  );
}

// ---- the loop: one diagram on the intro, a compact copy on every step highlighting its piece ----
const LOOP: { id: string; name: string; by: SponsorKey; what: string }[] = [
  { id: "look", name: "Look", by: "vast", what: "VSS explore" },
  { id: "find", name: "Find gaps", by: "vast", what: "VSS search" },
  { id: "decide", name: "Decide", by: "wandb", what: "Inference + Weave" },
  { id: "fill", name: "Fill", by: "nvidia", what: "Cosmos Transfer" },
  { id: "test", name: "Test", by: "nvidia", what: "YOLO + Reason" },
];
const STEP_NODE: Record<string, number> = {
  inventory: 0, missing: 1, matters: 2, grow: 3, test: 4, blind: 4, fix: 4, again: 5,
};
function LoopMap({ active = -1, big = false }: { active?: number; big?: boolean }) {
  const n = LOOP.length, W = big ? 980 : 520, nodeW = big ? 156 : 84, nodeH = big ? 64 : 28;
  const gap = (W - n * nodeW) / (n - 1), top = big ? 8 : 4, H = big ? 190 : 50;
  const x = (k: number) => k * (nodeW + gap);
  const retY = top + nodeH + (big ? 78 : 14);
  const loopOn = active === 5;
  return (
    <svg className={`st-loop ${big ? "big" : "mini"}`} viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Spore loop">
      <defs>
        <marker id={`ah${big ? "b" : "m"}`} viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto">
          <path d="M0,0 L8,4 L0,8 z" fill="currentColor" />
        </marker>
      </defs>
      {LOOP.slice(0, -1).map((_, k) => (
        <line key={k} className={`edge ${active > k && active <= 4 ? "lit" : ""}`} x1={x(k) + nodeW + 4} x2={x(k + 1) - 6}
          y1={top + nodeH / 2} y2={top + nodeH / 2} markerEnd={`url(#ah${big ? "b" : "m"})`} />
      ))}
      <path className={`edge ret ${loopOn ? "lit" : ""}`} markerEnd={`url(#ah${big ? "b" : "m"})`}
        d={`M ${x(n - 1) + nodeW / 2} ${top + nodeH + 4} Q ${x(n - 1) + nodeW / 2} ${retY} ${W / 2} ${retY} Q ${nodeW / 2} ${retY} ${nodeW / 2} ${top + nodeH + 6}`} />
      {big && <text className={`ret-lbl ${loopOn ? "lit" : ""}`} x={W / 2} y={retY - (big ? 10 : 5)} textAnchor="middle">repeat with the next gap</text>}
      {LOOP.map((node, k) => (
        <g key={node.id} className={`node ${k === active ? "on" : ""} ${active > k && active <= 5 ? "done" : ""}`}>
          <rect x={x(k)} y={top} width={nodeW} height={nodeH} rx={nodeH / 2} />
          <text x={x(k) + nodeW / 2} y={top + nodeH / 2 + (big ? -2 : 4)} textAnchor="middle" className="nm">{node.name}</text>
          {big && <text x={x(k) + nodeW / 2} y={top + nodeH / 2 + 17} textAnchor="middle" className="by">{node.what}</text>}
          {big && (
            <g className={`chip chip-${node.by}`}>
              <rect x={x(k) + nodeW / 2 - 62} y={top + nodeH + 10} width={124} height={24} rx={6} />
              <text x={x(k) + nodeW / 2} y={top + nodeH + 26.5} textAnchor="middle">{SPONSOR[node.by]}</text>
            </g>
          )}
        </g>
      ))}
    </svg>
  );
}

function Intro() {
  return (
    <div className="st-intro">
      <div className="st-logo rv">Spore</div>
      <p className="st-line rv" style={{ animationDelay: "0.25s" }}>
        <span>An agent that finds what your cameras have never seen,</span>{" "}
        <span className="hl">grows that footage, tests your video AI on it, and does it again.</span>
      </p>
      <div className="rv st-loop-wrap" style={{ animationDelay: "0.5s" }}><LoopMap big /></div>
      <p className="st-hint rv" style={{ animationDelay: "0.8s" }}>A video agent on VAST, NVIDIA and W&B. Press → to watch it work.</p>
    </div>
  );
}

function useCountUp(target: number, ms = 1600) {
  const [v, setV] = useState(0);
  useEffect(() => {
    if (!target) return;
    const t0 = performance.now();
    let id = 0;
    const tick = (t: number) => {
      const k = Math.min(1, (t - t0) / ms);
      setV(Math.round(target * (1 - Math.pow(1 - k, 3))));
      if (k < 1) id = requestAnimationFrame(tick);
    };
    id = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(id);
  }, [target, ms]);
  return v;
}

function Inv({ coverage, inventory }: { coverage?: (Coverage & { cameras?: Record<string, number> }) | null; inventory?: Inventory | null }) {
  const rows: { name: string; n: number }[] = inventory?.length
    ? inventory.map((r) => ({ name: (r.label ?? r.scene_type.replace(/_/g, " ")).split(" /")[0], n: r.n_clips }))
    : SCENES.map((s) => ({ name: s.name, n: s.cams.reduce((a, c) => a + (coverage?.cameras?.[c] ?? 0), 0) }));
  const total = coverage?.n_indexed ?? rows.reduce((a, r) => a + r.n, 0);
  const shown = useCountUp(total);
  const max = Math.max(1, ...rows.map((r) => r.n));
  return (
    <>
      <Title kicker="1 · Look">First, it looks through everything you have.</Title>
      <div className="st-count rv" style={{ animationDelay: "0.2s" }}>
        <b>{shown}</b><span>real clips scanned in VAST</span>
      </div>
      <div className="st-bars">
        {rows.map((r, k) => (
          <div key={r.name} {...rv(k + 2)}>
            <div className="st-bar-row">
              <span>{r.name}</span>
              <div className="st-bar"><i style={{ width: `${(100 * r.n) / max}%` }} /></div>
              <b>{r.n}</b>
            </div>
          </div>
        ))}
      </div>
      <Powered by={["vast", "nvidia"]} delay={1.9}
        what="VSS index in VastDB, captions by Cosmos Reason" />
    </>
  );
}

const GAP_ROWS = [
  { id: "clear_day", q: "highway in clear daytime" },
  { id: "night", q: "highway at night" },
  { id: "rain", q: "highway in heavy rain" },
  { id: "fog", q: "highway in dense fog" },
  { id: "snow", q: "highway covered in snow" },
];
function Missing({ coverage }: { coverage?: Coverage | null }) {
  const [done, setDone] = useState<Record<string, AskRes | "err">>({});
  useEffect(() => {
    let alive = true;
    GAP_ROWS.forEach((r, k) => {
      // Stagger the starts so rows visibly land one by one.
      setTimeout(() => {
        askVSS(r.q).then((x) => alive && setDone((m) => ({ ...m, [r.id]: x })))
          .catch(() => alive && setDone((m) => ({ ...m, [r.id]: "err" })));
      }, 400 + k * 500);
    });
    // Never let a slow search stall the demo: reveal the archive count after 14 s regardless.
    const t = setTimeout(() => alive && setDone((m) => Object.fromEntries(GAP_ROWS.map((r) => [r.id, m[r.id] ?? "err"]))), 14000);
    return () => { alive = false; clearTimeout(t); };
  }, []);
  const t = useElapsed(Object.keys(done).length < GAP_ROWS.length);
  const cov = Object.fromEntries((coverage?.conditions ?? []).map((c) => [c.id, c]));
  return (
    <>
      <Title kicker="2 · Find the gaps">It checks what your highway cameras have actually seen.</Title>
      <div className="st-live">
        {GAP_ROWS.map((r, k) => {
          const c = cov[r.id];
          const res = done[r.id];
          const n = c?.n_highway;
          const zero = n === 0;
          return (
            <div key={r.id} {...rv(k + 1)}>
              <div className={`st-live-row ${res ? (zero ? "zero" : "have") : "busy"}`}>
                <span className="lbl">{c?.label ?? r.id}</span>
                <code>vss.search(“{r.q}”)</code>
                <span className="res">
                  {!res ? <span className="pending"><i className="ring" />{t.toFixed(1)}s</span>
                    : <><b>{n ?? "?"}</b>{zero ? <em>no footage</em> : <em>clips</em>}</>}
                </span>
              </div>
            </div>
          );
        })}
      </div>
      <Powered by={["vast"]} delay={1.6}
        what="Live VSS search, running now" />
    </>
  );
}

const SUGGEST = ["construction zone at night", "pedestrian crossing in the rain", "truck stopped on the shoulder"];
function Ask() {
  const [q, setQ] = useState("");
  const [busy, setBusy] = useState(false);
  const [res, setRes] = useState<AskRes | null>(null);
  const [err, setErr] = useState("");
  const inp = useRef<HTMLInputElement>(null);
  const t = useElapsed(busy);
  const run = async (query: string) => {
    if (!query.trim() || busy) return;
    setQ(query); setBusy(true); setErr(""); setRes(null);
    inp.current?.blur(); // so → / ← work again right after searching
    try { const r = await askVSS(query.trim()); if (r.error) throw new Error(r.error); setRes(r); }
    catch { setErr("VSS didn't answer. Try again."); }
    setBusy(false);
  };
  const submit = (e: FormEvent) => { e.preventDefault(); run(q); };
  const shown = res ? [...res.hits.filter((h) => h.shows_it), ...res.hits.filter((h) => !h.shows_it)].slice(0, 3) : [];
  return (
    <>
      <Title kicker="Try it · live">Ask the archive for anything.</Title>
      <form className="st-ask rv" style={{ animationDelay: "0.3s" }} onSubmit={submit}
        onKeyDown={(e) => e.stopPropagation()}>
        <input ref={inp} value={q} onChange={(e) => setQ(e.target.value)} placeholder="e.g. construction zone at night" onKeyDown={(e) => { if (e.key === "Escape") (e.target as HTMLInputElement).blur(); }} />
        <button type="submit" disabled={busy}>{busy ? `${t.toFixed(1)}s` : "Search"}</button>
      </form>
      {!res && !busy && (
        <div className="st-suggest rv" style={{ animationDelay: "0.5s" }}>
          {SUGGEST.map((s) => <button key={s} onClick={() => run(s)}>{s}</button>)}
        </div>
      )}
      {busy && <p className="st-sub">Searching {""}every clip in VSS…</p>}
      {err && <p className="st-sub bad">{err}</p>}
      {res && (
        <>
          <div className="st-big rv">
            {res.n_shows_it
              ? <><b>{res.n_shows_it} of {res.hits.length}</b> top results actually show it.</>
              : <><b className="bad">No footage.</b> None of the top {res.hits.length} actually show it. That's a gap Spore can fill.</>}
          </div>
          <div className="st-hits">
            {shown.map((h, k) => (
              <figure key={(h.source ?? "") + k} {...rv(k + 1)} className={`rv ${h.shows_it ? "yes" : "no"}`}>
                <video src={clipUrl(h.source)} autoPlay muted loop playsInline />
                <figcaption>
                  <div className="meta"><b>{h.shows_it ? "shows it" : "doesn't"}</b> {h.camera_id} · {h.score.toFixed(2)}</div>
                  <div className="cap">{h.caption.slice(0, 150)}…</div>
                  <div className="yolo">YOLO in VSS: {counts(h.object_counts)}</div>
                </figcaption>
              </figure>
            ))}
          </div>
        </>
      )}
      <Powered by={["vast", "nvidia"]} delay={0.7}
        what={res ? `Live VSS search · ${res.seconds}s` : "Live VSS search"} />
    </>
  );
}

function Matters({ report }: { report?: (GapReport & { candidates?: Candidate[] }) | null }) {
  const seen = new Set<string>();
  const cands = (report?.candidates ?? []).filter((c) => { const l = c.label ?? c.condition ?? ""; if (seen.has(l)) return false; seen.add(l); return true; }).slice(0, 4);
  // This run's demo fills highway fog; the ranked list decides what comes next (see Repeat).
  const all = report?.candidates ?? [];
  const top = all.find((c) => c.id === "highway:fog") ?? cands[0];
  const others = all.filter((c) => (c.scene_type ?? c.scene) === "highway" && c.id !== top?.id)
    .map((c) => (c.label ?? c.condition ?? "").replace(/^Highway at /i, "").toLowerCase()).filter(Boolean);
  const why = (c: Candidate) => {
    const cond = (c.label ?? c.condition ?? "this").toLowerCase().replace(/^highway at /, "");
    const scene = (c.scene_label ?? "Highway").split(" /")[0];
    return `${scene} cameras face ${cond}, and nobody has tested the AI in it.`;
  };
  return (
    <>
      <Title kicker="3 · Decide">It decides which gap matters most.</Title>
      {top && (
        <div className="st-pick rv" style={{ animationDelay: "0.35s" }}>
          <div className="k">Filling first</div>
          <div className="n">{top.label ?? top.condition}</div>
          <div className="w">{why(top)}</div>
        </div>
      )}
      {others.length > 0 && (
        <p className="st-sub rv" style={{ animationDelay: "0.7s" }}>
          Also missing on highway cameras: {[...new Set(others)].slice(0, 4).join(", ")}. Each gets its own loop.
        </p>
      )}
      <Powered by={["wandb"]} delay={0.9}
        what={<>LLM on W&B Inference, traced in <a href={WEAVE} target="_blank" rel="noreferrer">Weave</a></>} />
    </>
  );
}

function Grow() {
  const [k, setK] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setK((x) => (x + 1) % GROW.length), 2600);
    return () => clearInterval(id);
  }, []);
  const box = useRef<HTMLDivElement>(null);
  useEffect(() => {
    box.current?.querySelectorAll("video").forEach((v) => { if (v.paused) v.play().catch(() => {}); });
  }, [k]);
  const g = GROW[k];
  return (
    <>
      <Title kicker="4 · Fill">It grows the missing weather onto a real clip.</Title>
      <div ref={box} className="st-morph rv" style={{ animationDelay: "0.3s" }}>
        {GROW.map((v, j) => (
          <video key={v.src} src={url(v.src)} autoPlay muted loop playsInline preload="auto" className={j === k ? "on" : ""} />
        ))}
        <div className={`st-morph-tag ${g.real ? "real" : ""}`}><b>{g.label}</b><span>{g.sub}</span></div>
        <div className="st-morph-dots">{GROW.map((v, j) => <i key={v.src} className={j === k ? "on" : ""} />)}</div>
      </div>
      <Powered by={["nvidia"]} delay={1.3}
        what="Cosmos Transfer 2.5 · same cars, so the answer is known" />
    </>
  );
}

function Test({ seed, hero }: { seed: Eval | null; hero: Eval | null }) {
  const rec = hero?.yolo?.recall_vs_seed;
  const rH = hero?.reason?.answers?.vehicle_count as number | undefined;
  return (
    <>
      <Title kicker="5 · Test">It tests the models inside VSS.</Title>
      <div className="rv" style={{ animationDelay: "0.3s" }}>
        <Video src={`results/overlays/${HERO}.mp4`} className="st-hero" />
      </div>
      <div className="st-verdict rv" style={{ animationDelay: "1.0s" }}>
        <span className="ok">Cosmos Reason: <b>“dense fog, {rH ?? "…"} vehicles”</b></span>
        <span className="vs">vs</span>
        <span className="bad">YOLO11: <b>misses {rec != null ? Math.round((1 - rec) * 10) : "…"} in 10 cars</b></span>
      </div>
      <Powered by={["nvidia"]} delay={1.5}
        what="Hosted YOLO11 + Cosmos Reason, the models VSS runs" />
    </>
  );
}

function Blind({ curve, headline }: { curve?: SeverityCurve | null; headline?: { by_condition?: Record<string, { mean: number }> } | null }) {
  const fog = headline?.by_condition?.["fog@0.4"]?.mean;
  const outOf10 = fog != null ? Math.round(fog * 10) : null;
  const colors: Record<string, string> = { fog: "#e8ecef", rain: "#6fb6ff", snow: "#d4ff3a" };
  const W = 560, H = 220, P = 30;
  const x = (s: number) => P + s * (W - 2 * P), y = (r: number) => H - P - r * (H - 2 * P);
  return (
    <>
      <Title kicker="6 · Measure">It finds where your AI goes blind.</Title>
      <div className="st-big rv" style={{ animationDelay: "0.3s" }}>
        In light fog, YOLO finds <b>{outOf10 ?? "…"} in 10</b> cars a person can still see.
      </div>
      <svg className="st-curve rv" style={{ animationDelay: "0.8s" }} viewBox={`0 0 ${W + 50} ${H}`}>
        <line x1={P} x2={W - P} y1={y(0.5)} y2={y(0.5)} className="half" />
        <text x={W - P} y={y(0.5) - 6} textAnchor="end" className="ax">half the cars</text>
        <text x={P} y={H - 8} className="ax">clear</text>
        <text x={W - P} y={H - 8} textAnchor="end" className="ax">heavy weather →</text>
        {["fog", "snow", "rain"].filter((k) => curve?.curves?.[k]?.length).map((k, j) => {
          const pts = curve!.curves![k];
          const last = pts[pts.length - 1];
          return (
            <g key={k}>
              <polyline fill="none" stroke={colors[k]} strokeWidth={3}
                points={pts.map((p) => `${x(p.severity)},${y(p.recall)}`).join(" ")} />
              <text x={x(last.severity) + 8} y={y(0.12) - 22 + j * 20} fill={colors[k]} className="lbl">{k}</text>
            </g>
          );
        })}
      </svg>
      <Powered by={["wandb"]} delay={1.5}
        what={<>Every eval traced in <a href={WEAVE} target="_blank" rel="noreferrer">W&B Weave</a></>} />
    </>
  );
}

function Fix({ fix }: { fix?: FixRow[] | null }) {
  const rows = (fix ?? []).filter((r) => r.after > r.before);
  const fmt = (v: number) => (v <= 1 ? `${Math.round(v * 100)}%` : v.toFixed(1));
  return (
    <>
      <Title kicker="Fix">Some of it, it can fix right away.</Title>
      {rows.length ? (
        <>
          <p className="st-sub rv" style={{ animationDelay: "0.3s" }}>
            VSS couldn't search for weather, because its captions never mentioned it. Re-ingested real clips with a weather-aware prompt:
          </p>
          <div className="st-fix">
            {rows.slice(0, 4).map((r, k) => (
              <div key={r.metric} {...rv(k + 2)}>
                <div className="st-fix-row">
                  <span className="m">{r.metric}</span>
                  <span className="b">{fmt(r.before)}</span>
                  <span className="arr">→</span>
                  <span className="a">{fmt(r.after)}</span>
                </div>
              </div>
            ))}
          </div>
        </>
      ) : (
        <div className="st-next-list">
          <div {...rv(1)}><div className="st-cand"><span className="rank">→</span><div><div className="name">Search</div><div className="why">Re-ingest with a weather-aware prompt so VSS captions say "fog" and fog becomes searchable.</div></div></div></div>
          <div {...rv(2)}><div className="st-cand"><span className="rank">→</span><div><div className="name">Detector</div><div className="why">Train on Spore's own clips, which come with free labels from the clear-day original.</div></div></div></div>
        </div>
      )}
      <Powered by={["vast", "nvidia"]} delay={1.3}
        what="VSS re-ingest: the same clips go back through the pipeline with a weather-aware prompt for Cosmos Reason"
        call="POST /api/v1/dashboard/reingest {custom_prompt}" />
    </>
  );
}

function Again({ report, loop }: { report?: GapReport | null; loop?: LoopEntry[] | null }) {
  return (
    <>
      <Title kicker="Repeat">Then it does it again.</Title>
      {(() => {
        const last = [...(loop ?? [])].reverse().find((e) => (e as { filled?: number }).filled) as
          (LoopEntry & { gap?: string; filled?: number; recall?: number }) | undefined;
        const cands = (report as { candidates?: Candidate[] } | null | undefined)?.candidates ?? [];
        const lastLbl = (last?.gap ?? "").split(" ·")[0].toLowerCase();
        const next = cands.find((c) => (c.label ?? "").toLowerCase() !== lastLbl && c.id !== "highway:fog");
        return (
          <>
            {last && (
              <div className="st-big rv" style={{ animationDelay: "0.3s" }}>
                Last loop: <b>{last.gap?.split(" ·")[0]}</b> → grew {last.filled} clips → YOLO kept{" "}
                <b>{Math.round(100 * (last.recall ?? 0))}%</b>{(last.recall ?? 0) >= 0.7 ? " (holds up)" : " (blind spot)"}
              </div>
            )}
            <div className="st-big rv" style={{ animationDelay: "0.6s" }}>
              Next up: <b>{next?.label ?? report?.next_label ?? "…"}</b>
            </div>
          </>
        );
      })()}
      <div className="st-stack rv" style={{ animationDelay: "0.9s" }}>
        <span><b>VAST</b> finds the gap</span>
        <span><b>NVIDIA</b> grows and tests it</span>
        <span><b>W&B on CoreWeave</b> reasons and traces</span>
        <span><b>Cursor</b> built it</span>
      </div>
      <p className="st-line rv" style={{ animationDelay: "1.3s" }}>{LINE}</p>
    </>
  );
}
