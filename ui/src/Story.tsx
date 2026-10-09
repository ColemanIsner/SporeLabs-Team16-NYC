import { useCallback, useEffect, useRef, useState, type FormEvent, type ReactNode } from "react";
import { url, useJSON, getJSON, type Coverage, type GapReport, type Eval, type SeverityCurve, type FixRow, type LoopEntry } from "./data";
import "./story.css";

// Step-by-step demo. One idea per screen; → / space to advance, ← to go back.
// The full dashboard is still at ?full=1.

const SEED = "i24_scene1_p1c2_00";
const HERO = `${SEED}__phys_fog_s04`;
const WEAVE = "https://wandb.ai/colemanisner-sporelabs/sporelabs-hackathon/weave";
const LINE = "Other tools find what your video archive is missing. Spore grows the data to fill it, and shows where your AI goes blind.";

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
    { key: "ask", render: () => <Ask /> },
    { key: "matters", render: () => <Matters report={report} /> },
    { key: "grow", render: () => <Grow /> },
    { key: "test", render: () => <Test seed={seedEval} hero={heroEval} /> },
    { key: "blind", render: () => <Blind curve={curve} headline={headline} /> },
    { key: "fix", render: () => <Fix fix={fix} /> },
    { key: "again", render: () => <Again report={report} loop={loop} /> },
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
          <span className="st-dot" /> Spore
        </button>
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
      <span className="lab">How</span>
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

function Intro() {
  return (
    <div className="st-intro">
      <div className="st-logo rv"><span className="st-dot big" /> Spore</div>
      <p className="st-line rv" style={{ animationDelay: "0.25s" }}>{LINE}</p>
      <p className="st-hint rv" style={{ animationDelay: "0.6s" }}>An agent for video archives. Press → to watch it work.</p>
    </div>
  );
}

function Inv({ coverage, inventory }: { coverage?: (Coverage & { cameras?: Record<string, number> }) | null; inventory?: Inventory | null }) {
  const rows: { name: string; n: number }[] = inventory?.length
    ? inventory.map((r) => ({ name: r.label ?? r.scene_type.replace(/_/g, " "), n: r.n_clips }))
    : SCENES.map((s) => ({ name: s.name, n: s.cams.reduce((a, c) => a + (coverage?.cameras?.[c] ?? 0), 0) }));
  const total = coverage?.n_indexed ?? rows.reduce((a, r) => a + r.n, 0);
  const max = Math.max(1, ...rows.map((r) => r.n));
  return (
    <>
      <Title kicker="1 · Look">First, it looks through everything you have.</Title>
      <div className="st-big rv" style={{ animationDelay: "0.3s" }}>
        <b>{total || "…"}</b> real clips in VAST
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
        what="Reads every clip VSS has indexed in VastDB, with the captions NVIDIA Cosmos Reason wrote at ingest"
        call={<>GET /api/v1/videos/explore → {total} clips, {rows.length} scene types</>} />
    </>
  );
}

function Missing({ coverage }: { coverage?: Coverage | null }) {
  const order = ["clear_day", "night", "rain", "fog", "snow", "glare", "night_rain"];
  const rows = (coverage?.conditions ?? []).slice().sort((a, b) => order.indexOf(a.id) - order.indexOf(b.id));
  return (
    <>
      <Title kicker="2 · Find the gaps">Then it asks: what conditions do the highway cameras actually cover?</Title>
      <div className="st-grid-cond">
        {rows.map((r, k) => (
          <div key={r.id} {...rv(k + 1)}>
            <div className={`st-cond ${r.n_highway === 0 ? "zero" : ""}`}>
              <span className="n">{r.n_highway}</span>
              <span className="l">{r.label}</span>
              {r.n_highway === 0 && <span className="tag">no footage</span>}
            </div>
          </div>
        ))}
      </div>
      <LiveLog delay={0.4 + rows.length * 0.22} />
      <Powered by={["vast"]} delay={0.6 + rows.length * 0.22}
        what="VSS hybrid search (Cosmos Embed1 vectors + captions), then a check of each hit's own caption"
        call="POST /api/v1/search · live" />
    </>
  );
}

const LIVE_QUERIES = ["highway in heavy rain", "highway in dense fog", "highway covered in snow"];
function LiveLog({ delay }: { delay: number }) {
  const [res, setRes] = useState<Record<string, AskRes | "err">>({});
  useEffect(() => {
    let alive = true;
    LIVE_QUERIES.forEach((q) => askVSS(q).then((r) => alive && setRes((m) => ({ ...m, [q]: r }))).catch(() => alive && setRes((m) => ({ ...m, [q]: "err" }))));
    return () => { alive = false; };
  }, []);
  const t = useElapsed(Object.keys(res).length < LIVE_QUERIES.length);
  return (
    <div className="st-log rv" style={{ animationDelay: `${delay}s` }}>
      {LIVE_QUERIES.map((q) => {
        const r = res[q];
        const hw = r && r !== "err" ? r.hits.filter((h) => h.highway && h.shows_it).length : 0;
        return (
          <div key={q} className="ln">
            <span className="cmd">vss.search(“{q}”)</span>
            {!r ? <span className="run">searching… {t.toFixed(1)}s</span>
              : r === "err" ? <span className="bad">offline</span>
              : <span className={hw ? "ok" : "bad"}>{r.hits.length} hits · {hw} highway clips actually show it · {r.seconds}s</span>}
          </div>
        );
      })}
    </div>
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
        <input ref={inp} value={q} onChange={(e) => setQ(e.target.value)} placeholder="e.g. construction zone at night" autoFocus />
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
        what="Live VSS hybrid search; we read each hit's Cosmos Reason caption and YOLO counts straight from its VastDB row"
        call={res ? `POST /api/v1/search "${res.query}" → ${res.hits.length} hits in ${res.seconds}s` : "POST /api/v1/search"} />
    </>
  );
}

function Matters({ report }: { report?: (GapReport & { candidates?: Candidate[] }) | null }) {
  const cands = (report?.candidates ?? []).slice(0, 3);
  const why = (c: Candidate) => {
    const cond = (c.label ?? c.condition ?? "this").toLowerCase().replace(/^highway at /, "");
    const scene = (c.scene_label ?? "Highway").split(" /")[0];
    const have = c.real_clips ?? c.real_clips_highway ?? 0;
    const of = c.scene_clips != null ? ` of ${c.scene_clips}` : "";
    const tested = c.mean_recall != null ? `In synthetic tests YOLO finds ${Math.round(c.mean_recall * 100)}% of cars.` : "Never tested.";
    return `${scene} cameras face ${cond}. They have ${have}${of} clips in it. ${tested}`;
  };
  return (
    <>
      <Title kicker="3 · Decide">It picks the gaps that matter for this footage.</Title>
      <div className="st-cands">
        {cands.map((c, k) => (
          <div key={c.id ?? k} {...rv(k + 1)}>
            <div className={`st-cand ${k === 0 ? "first" : ""}`}>
              <span className="rank">{k + 1}</span>
              <div>
                <div className="name">{c.label ?? c.condition}</div>
                <div className="why">{why(c)}</div>
              </div>
              {k === 0 && <span className="pick">filling first</span>}
            </div>
          </div>
        ))}
      </div>
      <Powered by={["wandb"]} delay={1.1}
        what={<>An LLM on W&B Inference reads the search results and evals and ranks the gaps. Every call is traced in <a href={WEAVE} target="_blank" rel="noreferrer">Weave</a></>}
        call={report?.model ? `${report.model}${(report as { llm_seconds?: number }).llm_seconds ? ` · ${(report as { llm_seconds?: number }).llm_seconds}s` : ""}` : undefined} />
    </>
  );
}

function Grow() {
  return (
    <>
      <Title kicker="4 · Fill">So it grows the missing footage from a real clip.</Title>
      <div className="st-grow">
        {GROW.map((g, k) => (
          <figure key={g.src} {...rv(k + 1)} className={`rv ${g.real ? "real" : ""}`}>
            <Video src={g.src} />
            <figcaption><b>{g.label}</b> {g.sub}</figcaption>
          </figure>
        ))}
      </div>
      <p className="st-foot rv" style={{ animationDelay: "1.5s" }}>
        Same cars, same lanes. So we already know the right answer for every clip.
      </p>
      <Powered by={["nvidia"]} delay={1.8}
        what="Cosmos Transfer 2.5 relights the real clip (night); a weather layer adds fog, rain and snow without moving a pixel"
        call="cosmos-transfer2.5-2b · edge control · 93 frames 720p · H100" />
    </>
  );
}

function Test({ seed, hero }: { seed: Eval | null; hero: Eval | null }) {
  const yS = seed?.yolo?.mean_count, yH = hero?.yolo?.mean_count;
  const rH = hero?.reason?.answers?.vehicle_count as number | undefined;
  const wH = hero?.reason?.answers?.weather_lighting as string | undefined;
  return (
    <>
      <Title kicker="5 · Test">Then it tests the models inside VSS on it.</Title>
      <div className="rv" style={{ animationDelay: "0.3s" }}>
        <Video src={`results/overlays/${HERO}.mp4`} className="st-hero" />
      </div>
      <div className="st-vs">
        <div {...rv(3)}>
          <div className="st-say ok">
            <div className="who">Cosmos Reason, the caption model</div>
            <div className="what">“{wH ?? "fog"}”, <b>{rH ?? "…"} vehicles</b></div>
          </div>
        </div>
        <div {...rv(4)}>
          <div className="st-say bad">
            <div className="who">YOLO11, the detector</div>
            <div className="what"><b>{yH != null ? yH.toFixed(1) : "…"}</b> vehicles <span>(clear day: {yS != null ? yS.toFixed(1) : "…"})</span></div>
          </div>
        </div>
      </div>
      <p className="st-foot rv" style={{ animationDelay: "1.4s" }}>
        Light fog. Same clip, same VSS pipeline, and two models disagree.
      </p>
      <Powered by={["nvidia"]} delay={1.7}
        what="The hosted models VSS runs at ingest, on CoreWeave GPUs: YOLO11s detector and Cosmos3-Reason"
        call={`YOLO11s × 93 frames · Cosmos3-Reason ${hero?.reason && (hero.reason as { seconds?: number }).seconds ? `${(hero.reason as { seconds?: number }).seconds}s` : ""}`} />
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
      <Title kicker="6 · Measure">And finds exactly where it goes blind.</Title>
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
      {curve?.breaking_point && (
        <p className="st-foot rv" style={{ animationDelay: "1.2s" }}>
          Below half the cars at {Object.entries(curve.breaking_point).filter(([k, v]) => v != null && k in colors).map(([k, v]) => `${k} ${v}`).join(" · ")}
        </p>
      )}
      <Powered by={["wandb"]} delay={1.5}
        what={<>Every generation and eval is logged and traced in <a href={WEAVE} target="_blank" rel="noreferrer">W&B Weave</a></>}
        call="recall on still-visible seed vehicles · 11 seeds × 3 severities" />
    </>
  );
}

function Fix({ fix }: { fix?: FixRow[] | null }) {
  const rows = (fix ?? []).filter((r) => r.after > r.before);
  const fmt = (v: number) => (v <= 1 ? `${Math.round(v * 100)}%` : v.toFixed(1));
  return (
    <>
      <Title kicker="7 · Fix">Some of it, it can fix right away.</Title>
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
  const n = loop?.length ?? 0;
  return (
    <>
      <Title kicker="8 · Repeat">Then it searches again and picks the next gap.</Title>
      <div className="st-big rv" style={{ animationDelay: "0.3s" }}>
        Next up: <b>{report?.next_label ?? "…"}</b>
      </div>
      <p className="st-sub rv" style={{ animationDelay: "0.6s" }}>
        {n} loop iteration{n === 1 ? "" : "s"} so far. Every step is traced in{" "}
        <a href={WEAVE} target="_blank" rel="noreferrer">W&B Weave</a>.
      </p>
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
