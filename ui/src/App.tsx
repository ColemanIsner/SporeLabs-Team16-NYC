import { useEffect, useMemo, useState, type ReactNode } from "react";
import {
  FIXTURES, url, useJSON, useEvals, condLabel, isPixel, pixelName, pct, num, fmtAnswer,
  WEATHERS, TIMES, INTENSITIES, ago,
  type Seed, type Variant, type Eval, type FailureMap, type Arm, type RealCheck, type FixRow, type Condition,
  type Coverage, type GapReport, type SeverityCurve, type SevPoint, type Overlay, type LoopEntry,
} from "./data";
import { SyncGroup, Video } from "./sync";

const POLL = 3000;

export default function App() {
  const seeds = useJSON<Seed[]>("data/seeds/manifest.json", 10000);
  const synthetic = useJSON<Variant[]>("data/synthetic/manifest.json", POLL);
  const rawMap = useJSON<FailureMap>("results/failure_map.json", POLL);
  const real = useJSON<RealCheck[]>("results/real_check.json", POLL);
  const fix = useJSON<FixRow[]>("results/fix.json", POLL);
  const coverage = useJSON<Coverage>("results/coverage.json", POLL);
  const report = useJSON<GapReport>("results/gap_report.json", POLL);
  const overlays = useJSON<Overlay[]>("results/overlays/index.json", POLL);
  const loopLog = useJSON<LoopEntry[]>("results/loop_log.json", POLL);
  const map = useFixtureReplay(rawMap);
  const sevCurve = useJSON<SeverityCurve>("results/severity_curve.json", POLL);

  const spore = useMemo(() => (Array.isArray(loopLog) ? loopLog : []).filter((e) => e && e.kind === "spore"), [loopLog]);
  const lastFill = useMemo(() => [...spore].reverse().find((e) => e.stages?.["3"]?.clips), [spore]);
  const filled = useMemo(() => new Set<string>(((lastFill?.stages?.["3"]?.clips as string[]) ?? [])), [lastFill]);

  const [seedId, setSeedId] = useState<string | null>(null);
  const fillSeed = (lastFill?.stages?.["3"]?.seeds as string[] | undefined)?.[0];
  useEffect(() => {
    if (!seedId && seeds && seeds.length) setSeedId(fillSeed && seeds.some((s) => s.seed_id === fillSeed) ? fillSeed : seeds[0].seed_id);
  }, [seeds, seedId, fillSeed]);
  const seed = seeds?.find((s) => s.seed_id === seedId) ?? null;
  const variants = useMemo(() => (synthetic ?? []).filter((v) => v.seed_id === seedId), [synthetic, seedId]);
  const evals = useEvals(useMemo(() => [seedId ?? "", ...variants.map((v) => v.clip_id)].filter(Boolean), [seedId, variants]));

  const gaps = (coverage?.conditions ?? []).filter((c) => c.gap).length;
  const worst = (map?.arms ?? []).filter((a) => a.n > 0 && a.mean_recall != null).sort((a, b) => (a.mean_recall ?? 1) - (b.mean_recall ?? 1))[0];
  const running = spore.find((e) => e.status === "running");

  return (
    <div className="app">
      <div className="grain" aria-hidden />
      <TopBar running={running} />
      <Hero
        stats={[
          { k: "real clips searched", v: coverage?.n_indexed ?? 0 },
          { k: "conditions with no highway footage", v: gaps, hot: gaps > 0 },
          { k: "synthetic clips grown", v: synthetic?.length ?? 0 },
          { k: "worst detector recall", v: worst ? pct(worst.mean_recall) : "—", hot: !!worst },
          { k: "loop iterations", v: spore.filter((e) => e.mode === "loop").length, ok: spore.length > 0 },
        ]}
      />
      <Section n="1" id="search" kicker="Search to find the gap"
        title={<>We searched the real archive for every condition. <em>Most of the bad-weather rows are empty.</em></>}
        aside={<LiveTag label={coverage?.updated ? `searched ${ago(coverage.updated)}` : undefined} active={running?.running_stage === 1} />}>
        <CoverageBars cov={coverage} />
        {real && real.length > 0 && (
          <details className="realfold">
            <summary>Real archive clips pulled for confirmation ({real.length} conditions)</summary>
            <RealConfirm data={real} />
          </details>
        )}
      </Section>
      <Section n="2" id="report" kicker="Report & explain the gap"
        title={<>An LLM reads the search results and the evals. <em>It says what's missing, why it matters, and what to fill next.</em></>}
        aside={<LiveTag label={report?.updated ? `written ${ago(report.updated)}` : undefined} active={running?.running_stage === 2} />}>
        <GapCard rep={report} />
      </Section>
      <SyncGroup>
        {(ctl) => (
          <Section n="3" id="fill" kicker="Fill the gap"
            title={<>Grow the missing footage from a real seed. <em>Same cars, same lanes, new weather, so the answer is already known.</em></>}
            aside={<div className="aside-row"><LiveTag active={running?.running_stage === 3} /><PlayCtl {...ctl} /></div>}>
            <SeedPicker seeds={seeds} seed={seed} onPick={setSeedId} filled={filled} synthetic={synthetic ?? []} />
            <GrowGrid seed={seed} variants={variants} evals={evals} loading={synthetic === undefined} filled={filled} />
          </Section>
        )}
      </SyncGroup>
      <Section n="4" id="measure" kicker="Measure"
        title={<>Run the stack on the filled gap. <em>Green boxes it still sees, red ones it lost.</em></>}
        aside={<LiveTag label={map?.budget_used != null ? `${map.budget_used} clips scored` : undefined} active={running?.running_stage === 4} />}>
        <Aha ov={overlays} />
        <h3 className="subhead">Breaking point: how much weather before YOLO goes blind</h3>
        <BreakingPoint curve={sevCurve} synthetic={synthetic ?? []} seeds={seeds ?? []} />
        <h3 className="subhead">Failure rate by condition</h3>
        <Heatmap map={map} />
        <h3 className="subhead">Fix: fine-tune on the filled gap, before → after</h3>
        <FixTable rows={fix} />
      </Section>
      <Section n="5" id="loop" kicker="Continue the loop"
        title={<>Then search again and pick the next gap. <em>Every iteration: gap → filled → measured.</em></>}
        aside={<LiveTag active={!!running} label={running ? `iteration ${running.iteration} · stage ${running.running_stage ?? "…"}` : undefined} />}>
        <Timeline entries={Array.isArray(loopLog) ? loopLog : loopLog} />
      </Section>
      <Footer />
    </div>
  );
}

/* ---------- 1 coverage ---------- */

function CoverageBars({ cov }: { cov: Coverage | null | undefined }) {
  if (cov === undefined) return <div className="skeleton tall" />;
  if (!cov || !cov.conditions?.length) return <Empty title="Archive not searched yet" hint="loop/.venv/bin/python loop/spore.py --stage 1" />;
  const max = Math.max(1, ...cov.conditions.map((c) => c.n_highway));
  return (
    <div className="cov">
      <p className="cov-cap mono">
        {cov.n_indexed} real clips indexed in VSS · {cov.n_highway} highway / traffic · bars = highway clips whose VSS caption shows the condition
      </p>
      <div className="cov-rows">
        {cov.conditions.map((c, i) => {
          const empty = c.n_highway === 0;
          return (
            <div key={c.id} className={`cov-row ${empty ? "zero" : ""} ${c.gap ? "gap" : ""}`} style={{ animationDelay: `${i * 50}ms` }}>
              <div className="cov-label">
                <b>{c.label}</b>
                <code className="query"><span>vss.search</span>(“{c.vss_query}”)</code>
              </div>
              <div className="cov-bar">
                <i style={{ width: `${(c.n_highway / max) * 100}%` }} />
                {empty && <span className="cov-zero">no footage</span>}
              </div>
              <div className="cov-num">
                <b>{c.n_highway}</b><small>highway</small>
              </div>
              <div className="cov-num dim">
                <b>{c.n_clips}</b><small>any camera</small>
              </div>
              <div className="cov-search" title={(c.top_hits ?? []).map((h) => `${num(h.score, 3)}  ${h.camera_id}  ${h.caption_matches ? "✓" : "✗"}  ${h.caption ?? ""}`).join("\n")}>
                <b>{c.search_relevant ?? 0}/{c.top_k ?? 10}</b>
                <small>top hits really show it · best {num(c.search_top_score ?? undefined, 2)}</small>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/* ---------- 2 gap report ---------- */

function GapCard({ rep }: { rep: GapReport | null | undefined }) {
  if (rep === undefined) return <div className="skeleton tall" />;
  if (!rep || !rep.gap) return <Empty title="No gap report yet" hint="loop/.venv/bin/python loop/spore.py --stage 2" />;
  const ev = Array.isArray(rep.evidence) ? rep.evidence : rep.evidence ? [rep.evidence] : [];
  return (
    <div className="gapcard">
      <div className="gap-main">
        <p className="label">The gap</p>
        <p className="gap-text">{rep.gap}</p>
        <p className="label">What that leads to</p>
        <p className="risk-text">{rep.risk}</p>
      </div>
      <aside className="gap-side">
        <p className="label">Evidence</p>
        <ul className="evidence">{ev.map((e, i) => <li key={i}>{e}</li>)}</ul>
        <div className="next">
          <p className="label">Fill next</p>
          <b>{rep.next_label ?? rep.next_condition}</b>
          <p>{rep.recommendation}</p>
        </div>
        <p className="src mono">{rep.source === "llm" ? `W&B Inference · ${rep.model ?? ""} · Weave-traced` : "deterministic fallback"}{rep.updated ? ` · ${ago(rep.updated)}` : ""}</p>
      </aside>
    </div>
  );
}

/* ---------- 3 seed picker ---------- */

function SeedPicker({ seeds, seed, onPick, filled, synthetic }: { seeds: Seed[] | null | undefined; seed: Seed | null; onPick: (id: string) => void; filled: Set<string>; synthetic: Variant[] }) {
  if (!seeds || seeds.length < 2) return null;
  const newBy = new Set(synthetic.filter((v) => filled.has(v.clip_id)).map((v) => v.seed_id));
  return (
    <div className="picker-row seedbar">
      {seeds.map((s) => (
        <button key={s.seed_id} className={s.seed_id === seed?.seed_id ? "on" : ""} onClick={() => onPick(s.seed_id)}>
          {s.seed_id}{newBy.has(s.seed_id) && <span className="newdot" title="filled this iteration" />}
        </button>
      ))}
    </div>
  );
}

/* ---------- 4 aha ---------- */

function Aha({ ov }: { ov: Overlay[] | null | undefined }) {
  const [i, setI] = useState(0);
  if (ov === undefined) return <div className="skeleton tall" />;
  if (!ov || !ov.length) return <Empty title="No overlay rendered yet" hint="eval/.venv/bin/python eval/render_all.py" />;
  const o = ov[Math.min(i, ov.length - 1)];
  return (
    <div className="aha">
      <div className="aha-video frame wide">
        <video key={o.mp4} src={url(o.mp4)} poster={url(o.jpg)} autoPlay muted loop playsInline className="vid contain" />
      </div>
      <aside className="aha-side">
        <span className="chip fail">{o.label ?? condLabel(o.condition)}</span>
        <div className="bignum">
          <b className="bad">{pct(o.recall_vs_seed)}</b>
          <span>of the seed's vehicles<br />still detected</span>
        </div>
        <dl className="answers">
          <div><dt>vehicles / frame, seed</dt><dd>{num(o.seed_mean_count, 1)}</dd></div>
          <div><dt>vehicles / frame, variant</dt><dd>{num(o.mean_detected, 1)}</dd></div>
          <div><dt>scene kept (edge SSIM)</dt><dd>{num(o.edge_ssim, 2)}</dd></div>
        </dl>
        {o.reason_seed && <p className="note"><span className="label">Cosmos Reason · seed</span><br />{o.reason_seed}</p>}
        {o.reason_variant && <p className="note"><span className="label">Cosmos Reason · variant</span><br />{o.reason_variant}</p>}
        {ov.length > 1 && (
          <div className="picker-row">
            {ov.slice(0, 8).map((x, j) => (
              <button key={x.clip_id} className={j === i ? "on" : ""} onClick={() => setI(j)}>{x.label ?? x.clip_id}</button>
            ))}
          </div>
        )}
      </aside>
    </div>
  );
}

/* ---------- 4b breaking point ---------- */

const KIND_COLOR: Record<string, string> = { fog: "#d4ff3a", rain: "#4f9bff", snow: "#ff9f6a", night: "#a58bff", glare: "#ffd166" };
const sevOf = (v: Variant) => (typeof v.condition?.severity === "number" ? (v.condition.severity as number) : undefined);
const kindOf = (v: Variant) => String(v.condition?.physics ?? v.condition?.weather ?? "");

function crossing(pts: SevPoint[], thr = 0.5): number | null {
  for (let i = 1; i < pts.length; i++) {
    const a = pts[i - 1], b = pts[i];
    if (a.recall >= thr && b.recall < thr) return a.severity + ((a.recall - thr) / (a.recall - b.recall || 1)) * (b.severity - a.severity);
  }
  return pts.length && pts[0].recall < thr ? pts[0].severity : null;
}

function BreakingPoint({ curve, synthetic, seeds }: { curve: SeverityCurve | null | undefined; synthetic: Variant[]; seeds: Seed[] }) {
  const phys = useMemo(() => synthetic.filter((v) => v.condition?.kind === "physics" && sevOf(v) != null && ["fog", "rain", "snow"].includes(kindOf(v))), [synthetic]);
  const needFallback = !curve || !curve.curves || !Object.keys(curve.curves).length;
  const evals = useEvals(useMemo(() => (needFallback ? phys.map((v) => v.clip_id) : []), [needFallback, phys]), 5000);

  const curves: Record<string, SevPoint[]> = useMemo(() => {
    let raw: Record<string, SevPoint[]> = {};
    if (!needFallback) raw = curve!.curves!;
    else {
      const acc: Record<string, Record<string, number[]>> = {};
      for (const v of phys) {
        const r = evals[v.clip_id]?.yolo?.recall_vs_seed;
        if (r == null) continue;
        const k = kindOf(v), s = sevOf(v)!.toFixed(2);
        ((acc[k] ??= {})[s] ??= []).push(r);
      }
      for (const [k, by] of Object.entries(acc))
        raw[k] = Object.entries(by).map(([s, rs]) => ({ severity: +s, recall: rs.reduce((x, y) => x + y, 0) / rs.length, n: rs.length }));
    }
    const out: Record<string, SevPoint[]> = {};
    for (const [k, pts] of Object.entries(raw)) {
      const p = [...(pts ?? [])].filter((x) => x && x.recall != null).sort((a, b) => a.severity - b.severity);
      if (!p.length) continue;
      if (p[0].severity > 0.05) p.unshift({ severity: 0, recall: 1, n: 0 });
      out[k] = p;
    }
    return out;
  }, [needFallback, curve, phys, evals]);

  const bp: Record<string, number | null> = {};
  for (const k of Object.keys(curves)) bp[k] = curve?.breaking_point?.[k] ?? crossing(curves[k]);
  const headK = bp.fog != null ? "fog" : Object.keys(bp).filter((k) => bp[k] != null).sort((a, b) => bp[a]! - bp[b]!)[0];

  const W = 620, H = 340, L = 52, B = 40, T = 16, Rr = 16;
  const x = (s: number) => L + s * (W - L - Rr);
  const y = (r: number) => T + (1 - r) * (H - T - B);

  return (
    <div className="bp">
      <div className="bp-chart">
        {headK ? (
          <div className="bp-callout">
            YOLO goes blind at <b style={{ color: KIND_COLOR[headK] }}>{headK}</b> severity <b className="bp-num">{bp[headK]!.toFixed(2)}</b>
            <small>recall on intact vehicles drops below 50%</small>
          </div>
        ) : (
          <div className="bp-callout dim">{Object.keys(curves).length ? "No condition crossed 50% recall yet" : "Severity sweep running…"}<small>results/severity_curve.json</small></div>
        )}
        <svg viewBox={`0 0 ${W} ${H}`} className="bp-svg" role="img" aria-label="YOLO recall vs weather severity">
          {[0, 0.25, 0.5, 0.75, 1].map((r) => (
            <g key={r}>
              <line x1={L} x2={W - Rr} y1={y(r)} y2={y(r)} stroke="rgba(255,255,255,.07)" />
              <text x={L - 10} y={y(r) + 4} textAnchor="end" className="bp-tick">{Math.round(r * 100)}%</text>
            </g>
          ))}
          {[0, 0.2, 0.4, 0.6, 0.8, 1].map((s) => (
            <text key={s} x={x(s)} y={H - B + 20} textAnchor="middle" className="bp-tick">{s.toFixed(1)}</text>
          ))}
          <text x={(L + W - Rr) / 2} y={H - 4} textAnchor="middle" className="bp-axis">weather severity →</text>
          <line x1={L} x2={W - Rr} y1={y(0.5)} y2={y(0.5)} stroke="#ff5233" strokeDasharray="6 5" strokeWidth={1.5} />
          <text x={W - Rr - 4} y={y(0.5) - 6} textAnchor="end" className="bp-fifty">50% recall</text>
          {Object.entries(curves).map(([k, pts]) => (
            <g key={k}>
              <polyline fill="none" stroke={KIND_COLOR[k] ?? "#ccc"} strokeWidth={3} strokeLinejoin="round" points={pts.map((p) => `${x(p.severity)},${y(p.recall)}`).join(" ")} />
              {pts.map((p) => <circle key={p.severity} cx={x(p.severity)} cy={y(p.recall)} r={4} fill={KIND_COLOR[k] ?? "#ccc"}><title>{`${k} ${p.severity.toFixed(1)}: ${Math.round(p.recall * 100)}% (n=${p.n ?? "?"})`}</title></circle>)}
              {bp[k] != null && <line x1={x(bp[k]!)} x2={x(bp[k]!)} y1={y(0.5) - 8} y2={y(0.5) + 8} stroke={KIND_COLOR[k]} strokeWidth={2} />}
            </g>
          ))}
        </svg>
        <div className="bp-legend">
          {Object.keys(curves).map((k) => (
            <span key={k}><i style={{ background: KIND_COLOR[k] }} />{k}{bp[k] != null ? ` · breaks at ${bp[k]!.toFixed(2)}` : " · holds"}</span>
          ))}
        </div>
      </div>
      <SeverityStrip phys={phys} seeds={seeds} />
    </div>
  );
}

function SeverityStrip({ phys, seeds }: { phys: Variant[]; seeds: Seed[] }) {
  const [kind, setKind] = useState("fog");
  const seedIds = useMemo(() => [...new Set(phys.map((v) => v.seed_id))], [phys]);
  const [sid, setSid] = useState<string | null>(null);
  const seedId = sid && seedIds.includes(sid) ? sid : seedIds[0];
  const seed = seeds.find((s) => s.seed_id === seedId);
  if (!seedId || !seed) return <Empty title="Severity strip appears when physics variants exist" hint="gen/weather.py" />;
  const pick = (s: number) => phys.find((v) => v.seed_id === seedId && kindOf(v) === kind && Math.abs((sevOf(v) ?? -1) - s) < 0.05);
  const frames: { s: number; src?: string }[] = [{ s: 0, src: seed.src }, ...[0.4, 0.7, 1.0].map((s) => ({ s, src: pick(s)?.src }))];
  return (
    <div className="strip">
      <div className="picker-row">
        {["fog", "rain", "snow"].map((k) => (
          <button key={k} className={k === kind ? "on" : ""} onClick={() => setKind(k)}>{k}</button>
        ))}
        <select value={seedId} onChange={(e) => setSid(e.target.value)} className="strip-sel">
          {seedIds.map((s) => <option key={s} value={s}>{s}</option>)}
        </select>
      </div>
      <div className="strip-frames">
        {frames.map((f) => (
          <div key={f.s} className="strip-frame">
            <div className="frame">
              {f.src ? <video key={f.src} src={url(f.src)} autoPlay muted loop playsInline className="vid" /> : <div className="video-missing"><span>generating…</span></div>}
            </div>
            <span className="mono">severity {f.s.toFixed(1)}{f.s === 0 ? " · seed" : ""}</span>
          </div>
        ))}
      </div>
    </div>
  );
}


/* ---------- 5 timeline ---------- */

const STAGE_NAMES: Record<string, string> = { "1": "search", "2": "report", "3": "fill", "4": "measure", "5": "re-search" };

function Timeline({ entries }: { entries: LoopEntry[] | null | undefined }) {
  if (entries === undefined) return <div className="skeleton" />;
  const list = (entries ?? []).filter((e) => e && (e.kind === "spore" || e.round != null));
  if (!list.length) return <Empty title="Loop hasn't run yet" hint="loop/.venv/bin/python loop/spore.py --iterations 2" />;
  return (
    <ol className="timeline">
      {[...list].reverse().slice(0, 12).map((e, idx) => {
        if (e.kind !== "spore") {
          return (
            <li key={`r${idx}`} className="tl done">
              <div className="tl-head"><b>round {e.round}</b><span className="mono dim">{ago(e.t)}</span></div>
              <p className="tl-sum">bandit arms {(e.arms ?? []).join(", ")} → {e.made ?? 0} clips</p>
            </li>
          );
        }
        const st = e.stages ?? {};
        const s2 = st["2"], s3 = st["3"], s4 = st["4"];
        const recall = s4?.mean_recall as number | undefined;
        return (
          <li key={`s${e.iteration}-${idx}`} className={`tl ${e.status === "running" ? "running" : "done"}`}>
            <div className="tl-head">
              <b>iteration {e.iteration}</b>
              {e.mode && e.mode !== "loop" && <span className="chip">{e.mode.replace("stage", "stage ")}</span>}
              <span className="mono dim">{ago(e.started)}</span>
            </div>
            <div className="tl-flow">
              <span className="tl-gap">gap: <b>{String(s2?.next_label ?? e.gap_id ?? s3?.label ?? "—")}</b></span>
              <span className="arr">→</span>
              <span>filled <b>{(s3?.clips as string[] | undefined)?.length ?? "—"}</b> clips</span>
              <span className="arr">→</span>
              <span>measured recall <b className={recall != null && recall < 0.7 ? "bad" : ""}>{pct(recall)}</b></span>
            </div>
            <div className="tl-stages">
              {["1", "2", "3", "4", "5"].map((k) => {
                const s = st[k];
                return (
                  <span key={k} className={`tl-st ${s ? s.status : "skip"}`}>
                    <i>{k}</i>{STAGE_NAMES[k]}{s?.seconds != null && <small>{Math.round(s.seconds)}s</small>}
                    {s?.status === "running" && <span className="spin" />}
                  </span>
                );
              })}
            </div>
            {e.summary && <p className="tl-sum">{e.summary}</p>}
          </li>
        );
      })}
    </ol>
  );
}

/* ---------- chrome ---------- */

function SporeMark() {
  return (
    <svg className="mark" viewBox="0 0 32 32" aria-hidden>
      <circle cx="16" cy="16" r="5" fill="var(--spore)" />
      {[0, 60, 120, 180, 240, 300].map((a, i) => {
        const r = (a * Math.PI) / 180;
        return <circle key={a} cx={16 + Math.cos(r) * 11} cy={16 + Math.sin(r) * 11} r={i % 2 ? 1.6 : 2.3} fill="var(--spore)" opacity={i % 2 ? 0.5 : 0.85} />;
      })}
    </svg>
  );
}

function TopBar({ running }: { running?: LoopEntry }) {
  const links = [["1", "Search", "search"], ["2", "Report", "report"], ["3", "Fill", "fill"], ["4", "Measure", "measure"], ["5", "Loop", "loop"]];
  return (
    <header className="topbar">
      <a className="brand" href="#top">
        <SporeMark />
        <span className="wordmark">Spore</span>
        <span className="by">by SporeLabs</span>
      </a>
      <nav>
        {links.map(([n, l, id]) => (
          <a key={id} href={`#${id}`} className={running?.running_stage === Number(n) ? "live-nav" : ""}><span>{n}</span>{l}</a>
        ))}
      </nav>
      {FIXTURES && <span className="fixture-tag">fixtures</span>}
    </header>
  );
}

function Hero({ stats }: { stats: { k: string; v: number | string; hot?: boolean; ok?: boolean }[] }) {
  return (
    <section className="hero" id="top">
      <p className="eyebrow">Find the gap · fill it · measure · repeat</p>
      <h1>
        Your video archive has <br />no rain at night.
        <span className="sub">Spore searches it, explains what's missing, grows the missing footage, and measures what breaks.</span>
      </h1>
      <div className="stats">
        {stats.map((s) => (
          <div key={s.k} className={`stat ${s.hot ? "hot" : ""} ${s.ok ? "ok" : ""}`}>
            <b>{s.v}</b>
            <span>{s.k}</span>
          </div>
        ))}
      </div>
    </section>
  );
}

function Section({ n, id, kicker, title, aside, children }: { n: string; id: string; kicker: string; title: ReactNode; aside?: ReactNode; children: ReactNode }) {
  return (
    <section className="section" id={id}>
      <header className="section-head">
        <div>
          <p className="kicker"><span>{n}</span>{kicker}</p>
          <h2>{title}</h2>
        </div>
        {aside}
      </header>
      {children}
    </section>
  );
}

function Empty({ title, hint }: { title: string; hint: string }) {
  return (
    <div className="empty">
      <div className="empty-dots" aria-hidden><i /><i /><i /></div>
      <p>{title}</p>
      <code>{hint}</code>
    </div>
  );
}

function Footer() {
  return (
    <footer className="footer">
      <div className="brand"><SporeMark /><span className="wordmark">Spore</span></div>
      <p>VAST Data · NVIDIA Cosmos Transfer &amp; Reason · YOLO · W&amp;B Weave · CoreWeave</p>
    </footer>
  );
}

/* ---------- 02 grow ---------- */

function PlayCtl({ playing, toggle, restart }: { playing: boolean; toggle: () => void; restart: () => void }) {
  return (
    <div className="playctl">
      <span className="sync-dot" /> synced
      <button onClick={restart} title="Restart all">↺</button>
      <button onClick={toggle} title={playing ? "Pause all" : "Play all"}>{playing ? "❚❚" : "▶"}</button>
    </div>
  );
}

function CondChips({ c }: { c?: Condition }) {
  if (!c) return null;
  if (isPixel(c)) return (<><span className="chip pix">pixel</span><span className="chip">{pixelName(c).replace(/_/g, " ")}</span></>);
  return (
    <>
      {[c.weather, c.time, c.intensity].filter(Boolean).map((x) => (
        <span key={String(x)} className={`chip c-${x}`}>{String(x)}</span>
      ))}
    </>
  );
}

function Metric({ k, v, bad, d }: { k: string; v?: number; bad?: boolean; d?: number }) {
  const w = v == null ? 0 : Math.max(0, Math.min(1, v));
  return (
    <div className={`metric ${bad ? "bad" : ""}`}>
      <span>{k}</span>
      <b>{d != null ? num(v, d) : pct(v)}</b>
      <i><em style={{ width: `${w * 100}%` }} /></i>
    </div>
  );
}

const isPhys = (v: Variant) => v.condition?.kind === "physics" || v.condition?.kind === "hybrid" || /^weather\.py/.test(v.generator ?? "");
const isCosmos = (v: Variant) => v.condition?.kind === "generative" || /cosmos/i.test(v.generator ?? "");
function genLabel(v: Variant) {
  if (isCosmos(v)) return "Cosmos Transfer";
  if (isPhys(v)) {
    const sev = v.condition?.severity as number | undefined;
    return `physics layer${sev != null ? ` · sev ${sev.toFixed(1)}` : ""}`;
  }
  if (isPixel(v.condition)) return "ffmpeg pixel";
  return v.generator ?? "generator";
}
function preferred(v: Variant) {
  const c = v.condition ?? {};
  if (isPhys(v) && ["fog", "rain", "snow"].includes(String(c.physics ?? c.weather))) {
    const sev = c.severity as number | undefined;
    return sev == null || Math.abs(sev - 0.7) < 0.05 ? 2 : 0;
  }
  if (isCosmos(v) && c.time === "night") return 2;
  return isCosmos(v) ? 1 : 0;
}

function GrowGrid({ seed, variants, evals, loading, filled }: { seed: Seed | null; variants: Variant[]; evals: Record<string, Eval | null>; loading: boolean; filled?: Set<string> }) {
  const [all, setAll] = useState(false);
  if (loading) return <div className="grid">{[0, 1, 2].map((i) => <div key={i} className="skeleton tile-sk" />)}</div>;
  if (!seed) return <Empty title="Pick a seed first" hint="data/seeds/manifest.json" />;
  const sorted = [...variants].sort((a, b) => {
    const ea = evals[a.clip_id], eb = evals[b.clip_id];
    const fa = filled?.has(a.clip_id) ? 1 : 0, fb = filled?.has(b.clip_id) ? 1 : 0;
    return fb - fa || preferred(b) - preferred(a) || Number(!!eb?.failure) - Number(!!ea?.failure) || (ea?.yolo?.recall_vs_seed ?? 2) - (eb?.yolo?.recall_vs_seed ?? 2);
  });
  return (
    <>
      <div className="grid">
        <div className="tile seedtile">
          <div className="frame">
            <Video src={url(seed.src)} className="vid" />
            <div className="vid-overlay"><span className="chip solid">seed · ground truth</span></div>
          </div>
          <div className="tile-foot">
            <Metric k="fidelity" v={1} d={2} />
            <Metric k="recall" v={1} />
            <Metric k="agree" v={1} />
          </div>
        </div>
        {(all ? sorted : sorted.slice(0, 8)).map((v) => {
          const e = evals[v.clip_id];
          const fidFail = e?.fidelity?.pass === false;
          const cls = e?.failure ? "fail" : fidFail ? "gated" : e ? "pass" : "pending";
          return (
            <div key={v.clip_id} className={`tile ${cls}`}>
              <div className="frame">
                <Video src={url(v.src)} className="vid" />
                <div className="vid-overlay"><span className={`chip gen ${isCosmos(v) ? "cosmos" : "phys"}`}>{genLabel(v)}</span><CondChips c={v.condition} />{filled?.has(v.clip_id) && <span className="chip solid">just filled</span>}</div>
                {e?.failure && <div className="flag">blind spot</div>}
                {fidFail && <div className="flag gate">fidelity gate · excluded</div>}
              </div>
              <div className="tile-foot">
                {e ? (
                  <>
                    <Metric k="fidelity" v={e.fidelity?.edge_ssim} d={2} bad={fidFail} />
                    <Metric k="recall" v={e.yolo?.recall_vs_seed} bad={(e.yolo?.recall_vs_seed ?? 1) < 0.7} />
                    <Metric k="agree" v={e.reason?.agree_vs_seed} bad={(e.reason?.agree_vs_seed ?? 1) < 0.67} />
                  </>
                ) : (
                  <p className="pending-note"><span className="spin" /> evaluating…</p>
                )}
              </div>
            </div>
          );
        })}
      </div>
      {sorted.length > 8 && <button className="showall" onClick={() => setAll(!all)}>{all ? "show fewer" : `show all ${sorted.length} variants`}</button>}
      {!variants.length && <Empty title="No variants grown for this seed yet" hint="data/synthetic/manifest.json" />}
    </>
  );
}

/* ---------- 03 heatmap ---------- */

const RAMP = ["#16191d", "#3b2318", "#7a2e17", "#c7401c", "#ff6a3d", "#ffc49a"];
function heat(r: number) {
  const x = Math.max(0, Math.min(1, r)) * (RAMP.length - 1);
  const i = Math.min(RAMP.length - 2, Math.floor(x));
  const t = x - i;
  const a = RAMP[i], b = RAMP[i + 1];
  const ch = (h: string, k: number) => parseInt(h.slice(1 + k * 2, 3 + k * 2), 16);
  const c = [0, 1, 2].map((k) => Math.round(ch(a, k) + (ch(b, k) - ch(a, k)) * t));
  return `rgb(${c.join(",")})`;
}

const sameCond = (a: Condition, w: string, t: string, i: string) =>
  !isPixel(a) && a.weather === w && a.time === t && a.intensity === i;

function LiveTag({ label, active }: { label?: string; active?: boolean }) {
  return (
    <div className={`live ${active ? "active" : ""}`}>
      <span className="pulse" /> {active ? "running now" : "live · polling 3s"}
      {label && <b className="mono">{label}</b>}
    </div>
  );
}

function Heatmap({ map }: { map: FailureMap | null | undefined }) {
  if (map === undefined) return <div className="skeleton tall" />;
  if (!map || !Array.isArray(map.arms)) return <Empty title="The hunt hasn't started" hint="results/failure_map.json" />;
  const arms = map.arms;
  const weathers = [...WEATHERS, ...new Set(arms.filter((a) => !isPixel(a.condition) && a.condition.weather && !WEATHERS.includes(a.condition.weather)).map((a) => a.condition.weather!))];
  const pixel = arms.filter((a) => isPixel(a.condition));
  const top = arms.filter((a) => a.n > 0).sort((a, b) => b.failure_rate - a.failure_rate || b.n - a.n).slice(0, 4);
  return (
    <div className="hunt">
      <div className="facets">
        {INTENSITIES.map((inten) => (
          <div key={inten} className="facet">
            <p className="label">intensity · <b>{inten}</b></p>
            <div className="hm" style={{ gridTemplateColumns: `64px repeat(${TIMES.length}, 1fr)` }}>
              <span />
              {TIMES.map((t) => <span key={t} className="hm-col">{t}</span>)}
              {weathers.map((w) => (
                <Row key={w} w={w} inten={inten} arms={arms} />
              ))}
            </div>
          </div>
        ))}
      </div>
      <aside className="hunt-side">
        <p className="label">Worst conditions</p>
        {top.length ? (
          <ol className="rank">
            {top.map((a, i) => (
              <li key={condLabel(a.condition)} style={{ animationDelay: `${i * 60}ms` }}>
                <span className="rank-n">{i + 1}</span>
                <div>
                  <p>{condLabel(a.condition)}</p>
                  <small>recall {pct(a.mean_recall)} · agree {pct(a.mean_agree)} · n={a.n}</small>
                </div>
                <b style={{ color: heat(Math.max(a.failure_rate, 0.55)) }}>{pct(a.failure_rate)}</b>
              </li>
            ))}
          </ol>
        ) : <p className="muted">No arms explored yet.</p>}
        <div className="legend">
          <span>0%</span>
          <i style={{ background: `linear-gradient(90deg, ${RAMP.join(",")})` }} />
          <span>100%</span>
        </div>
        <p className="legend-cap">failure rate · number = clips tested · hatched = unexplored</p>
        {pixel.length > 0 && (
          <div className="pixarms">
            <p className="label">Pixel baselines (ffmpeg)</p>
            <div className="pixrow">
              {pixel.map((a) => (
                <div key={pixelName(a.condition)} className="pixcell" style={{ background: a.n ? heat(a.failure_rate) : undefined }}>
                  <span>{pixelName(a.condition).replace(/_/g, " ")}</span>
                  <b>{a.n ? pct(a.failure_rate) : "—"}</b>
                </div>
              ))}
            </div>
          </div>
        )}
      </aside>
    </div>
  );
}

function Row({ w, inten, arms }: { w: string; inten: string; arms: Arm[] }) {
  return (
    <>
      <span className="hm-row">{w}</span>
      {TIMES.map((t) => {
        const a = arms.find((x) => sameCond(x.condition, w, t, inten));
        const control = w === "clear" && t === "day" && inten === "light";
        const n = a?.n ?? 0;
        const r = a?.failure_rate ?? 0;
        return (
          <div
            key={`${t}-${n}`}
            className={`cell ${n ? "on" : "off"} ${r >= 0.5 && n ? "hotcell" : ""}`}
            style={{ background: n ? heat(r) : undefined, color: r > 0.75 ? "#1a0d07" : undefined }}
            title={a ? `${condLabel(a.condition)}\nfailure ${pct(r)} · n=${n}\nrecall ${pct(a.mean_recall)} · agree ${pct(a.mean_agree)}` : "unexplored"}
          >
            {n ? (
              <>
                <b>{n}</b>
                <small>{pct(r)}</small>
              </>
            ) : <small>—</small>}
            {control && <em className="ctrl">control</em>}
          </div>
        );
      })}
    </>
  );
}

/** In fixtures mode, replay the static failure map arm-by-arm so the heatmap visibly fills in. */
function useFixtureReplay(map: FailureMap | null | undefined): FailureMap | null | undefined {
  const [tick, setTick] = useState(0);
  useEffect(() => {
    if (!FIXTURES) return;
    const id = setInterval(() => setTick((t) => t + 1), 1200);
    return () => clearInterval(id);
  }, []);
  return useMemo(() => {
    if (!FIXTURES || !map?.arms) return map;
    const order = map.arms.map((_, i) => i).sort((a, b) => ((a * 7919) % 13) - ((b * 7919) % 13));
    const total = order.length + 6; // pause on the full map
    const k = tick % total;
    const shown = new Set(order.slice(0, Math.min(k + 1, order.length)));
    const arms = map.arms.map((a, i) => (shown.has(i) ? a : { ...a, n: 0, failure_rate: 0 }));
    return { arms, budget_used: arms.reduce((s, a) => s + a.n, 0) };
  }, [map, tick]);
}

/* ---------- 04 real confirmation ---------- */

function RealConfirm({ data }: { data: RealCheck[] | null | undefined }) {
  if (data === undefined) return <div className="skeleton tall" />;
  if (!data || !data.length) return <Empty title="No real-archive checks yet" hint="results/real_check.json" />;
  return (
    <div className="real">
      {data.map((rc, i) => (
        <article key={i} className={`realcard ${rc.confirmed ? "confirmed" : ""}`}>
          <header>
            <div>
              <div className="chips"><CondChips c={rc.condition} /></div>
              {rc.vss_query && <p className="query mono"><span>vss.search</span>(“{rc.vss_query}”)</p>}
            </div>
            <span className={`badge ${rc.confirmed ? "yes" : "no"}`}>{rc.confirmed ? "✓ confirmed on real" : "not confirmed"}</span>
          </header>
          <div className="realclips">
            {(rc.clips ?? []).map((c, j) => (
              <div key={j} className="realclip">
                <div className="frame">
                  <Video src={url(c.src)} className="vid" />
                  <div className="vid-overlay"><span className="chip solid">real archive</span></div>
                  {c.vss_id && <div className="vid-foot mono">{c.vss_id}</div>}
                </div>
                <div className="realstats">
                  <div className="bignum sm"><b>{num(c.yolo_mean_count, 1)}</b><span>vehicles / frame</span></div>
                  <dl className="answers">
                    {Object.entries(c.reason_answers ?? {}).map(([k, v]) => (
                      <div key={k}><dt>{k.replace(/_/g, " ")}</dt><dd>{fmtAnswer(v)}</dd></div>
                    ))}
                  </dl>
                  {c.human_note && <p className="note">“{c.human_note}”</p>}
                </div>
              </div>
            ))}
          </div>
        </article>
      ))}
    </div>
  );
}

/* ---------- 05 fix ---------- */

function FixTable({ rows }: { rows: FixRow[] | null | undefined }) {
  if (rows === undefined) return <div className="skeleton" />;
  if (!rows || !rows.length) return <Empty title="Fix not measured yet" hint="results/fix.json" />;
  const isFrac = (x: number) => Math.abs(x) <= 1.0001;
  const fmt = (x: number) => (isFrac(x) ? pct(x) : num(x, 2));
  const head = rows[0];
  return (
    <div className="fix">
      <div className="fix-hero">
        <p className="label">{head.metric}</p>
        <div className="fix-big">
          <span className="before">{fmt(head.before)}</span>
          <span className="arrow">→</span>
          <span className="after">{fmt(head.after)}</span>
        </div>
        <p className="muted">{head.set ?? "real held-out"}</p>
      </div>
      <table className="fixtable">
        <thead><tr><th>metric</th><th>before</th><th /><th>after</th><th>Δ</th><th>set</th></tr></thead>
        <tbody>
          {rows.map((r, i) => {
            const d = r.after - r.before;
            const frac = isFrac(r.before) && isFrac(r.after);
            return (
              <tr key={i}>
                <td>{r.metric}</td>
                <td className="mono dim">{fmt(r.before)}</td>
                <td className="barcell">
                  {frac && (
                    <div className="dbar">
                      <i className="b" style={{ width: `${r.before * 100}%` }} />
                      <i className="a" style={{ width: `${r.after * 100}%` }} />
                    </div>
                  )}
                </td>
                <td className="mono strong">{fmt(r.after)}</td>
                <td className={`mono ${d > 0.005 ? "up" : d < -0.005 ? "down" : "dim"}`}>
                  {d > 0 ? "+" : ""}{frac ? `${Math.round(d * 100)} pts` : num(d, 2)}
                </td>
                <td className="dim">{r.set ?? ""}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
