import { useCallback, useEffect, useState, type ReactNode } from "react";
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
      <p className="st-foot rv" style={{ animationDelay: `${0.4 + rows.length * 0.22}s` }}>
        Highway clips per condition, from <code>vss.search()</code> plus VSS's own captions.
      </p>
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
      {report?.model && (
        <p className="st-foot rv" style={{ animationDelay: "1.1s" }}>
          Reasoned by {report.model} on W&B Inference · traced in Weave
        </p>
      )}
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
