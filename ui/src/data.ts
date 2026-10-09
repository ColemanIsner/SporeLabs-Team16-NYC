import { useEffect, useRef, useState } from "react";

export type Condition = {
  weather?: string; time?: string; intensity?: string; kind?: string;
  effect?: string; pixel?: string; name?: string; [k: string]: unknown;
};
export type Seed = { seed_id: string; dataset?: string; vss_id?: string; src: string; notes?: string; fps?: number };
export type Variant = { clip_id: string; seed_id: string; src: string; condition: Condition; prompt?: string; generator?: string; gen_seconds?: number; gpu?: string };
export type Eval = {
  clip_id: string; seed_id?: string; condition?: Condition;
  fidelity?: { edge_ssim?: number; pass?: boolean };
  yolo?: { model?: string; mean_count?: number; recall_vs_seed?: number };
  reason?: { summary?: string; answers?: Record<string, unknown>; agree_vs_seed?: number };
  failure?: boolean; failure_reasons?: string[];
};
export type Arm = { condition: Condition; n: number; failure_rate: number; mean_recall?: number; mean_agree?: number };
export type FailureMap = { arms: Arm[]; budget_used?: number };
export type RealCheck = {
  condition: Condition; vss_query?: string; confirmed?: boolean;
  clips: { vss_id?: string; src?: string; yolo_mean_count?: number; reason_answers?: Record<string, unknown>; human_note?: string }[];
};
export type FixRow = { metric: string; before: number; after: number; set?: string };

export const FIXTURES = new URLSearchParams(location.search).has("fixtures");
const BASE = FIXTURES ? "/fixtures/" : "/";

/** Resolve a repo-relative path (e.g. "data/seeds/x.mp4") to a served URL. */
export function url(p?: string): string | undefined {
  if (!p) return undefined;
  if (/^https?:\/\//.test(p)) return p;
  let s = p.replace(/\\/g, "/");
  const i = s.search(/(^|\/)(data|results)\//);
  if (i > 0) s = s.slice(i + 1);
  return BASE + s.replace(/^\.?\//, "");
}

export async function getJSON<T>(path: string): Promise<T | null> {
  try {
    const r = await fetch(url(path)! + `?t=${Date.now()}`, { cache: "no-store" });
    if (!r.ok) return null;
    const ct = r.headers.get("content-type") || "";
    if (!ct.includes("json")) return null;
    return (await r.json()) as T;
  } catch {
    return null;
  }
}

/** Fetch JSON once, optionally re-polling every `pollMs`. undefined = loading, null = missing. */
export function useJSON<T>(path: string | null, pollMs?: number): T | null | undefined {
  const [val, setVal] = useState<T | null | undefined>(undefined);
  const last = useRef<string>("");
  useEffect(() => {
    if (!path) { setVal(null); return; }
    let alive = true;
    last.current = "";
    setVal(undefined);
    const load = async () => {
      const v = await getJSON<T>(path);
      if (!alive) return;
      const key = JSON.stringify(v);
      if (key !== last.current) { last.current = key; setVal(v); }
    };
    load();
    const id = pollMs ? setInterval(load, pollMs) : undefined;
    return () => { alive = false; if (id) clearInterval(id); };
  }, [path, pollMs]);
  return val;
}

/** Fetch results/evals/<id>.json for many clip ids, retrying missing ones on each poll. */
export function useEvals(ids: string[], pollMs = 3000): Record<string, Eval | null> {
  const [evals, setEvals] = useState<Record<string, Eval | null>>({});
  const key = ids.join("|");
  useEffect(() => {
    let alive = true;
    const have: Record<string, Eval | null> = {};
    setEvals({});
    const load = async () => {
      const todo = ids.filter((id) => !have[id]);
      if (!todo.length) return;
      const got = await Promise.all(todo.map((id) => getJSON<Eval>(`results/evals/${id}.json`)));
      if (!alive) return;
      let changed = false;
      todo.forEach((id, i) => { if (got[i] || !(id in have)) { have[id] = got[i]; changed = true; } });
      if (changed) setEvals({ ...have });
    };
    load();
    const t = setInterval(load, pollMs);
    return () => { alive = false; clearInterval(t); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, pollMs]);
  return evals;
}

export const WEATHERS = ["clear", "fog", "rain", "snow"];
export const TIMES = ["day", "dusk", "night"];
export const INTENSITIES = ["light", "heavy"];

export function isPixel(c?: Condition) {
  return !!c && (c.kind === "pixel" || !!c.effect || !!c.pixel);
}
export function pixelName(c: Condition): string {
  return String(c.effect ?? c.pixel ?? c.name ?? c.weather ?? "pixel");
}
export function condLabel(c?: Condition): string {
  if (!c) return "unknown";
  if (isPixel(c)) return pixelName(c).replace(/_/g, " ");
  return [c.weather, c.time, c.intensity].filter(Boolean).join(" · ");
}
export const pct = (x?: number) => (x == null || Number.isNaN(x) ? "—" : `${Math.round(x * 100)}%`);
export const num = (x?: number, d = 2) => (x == null || Number.isNaN(x) ? "—" : x.toFixed(d));
export function fmtAnswer(v: unknown): string {
  if (v === true) return "yes";
  if (v === false) return "no";
  if (v == null) return "—";
  return String(v);
}

export type CovHit = { filename?: string; camera_id?: string; score?: number; caption_matches?: boolean; highway?: boolean; caption?: string };
export type CovRow = {
  id: string; label: string; vss_query: string; endpoint?: string; top_k?: number;
  n_clips: number; n_highway: number; frac_highway?: number; cameras?: string[];
  search_top_score?: number | null; search_relevant?: number; search_relevant_highway?: number;
  top_hits?: CovHit[]; gap?: boolean; error?: string | null;
};
export type Coverage = { n_indexed: number; n_highway: number; conditions: CovRow[]; updated?: number; method?: string; seconds?: number };
export type GapReport = {
  gap?: string; evidence?: string[] | string; risk?: string; recommendation?: string;
  next_condition?: string | null; next_label?: string | null; source?: string; model?: string; updated?: number;
};
export type Overlay = {
  clip_id: string; seed_id?: string; label?: string; mp4?: string; jpg?: string; condition?: Condition;
  recall_vs_seed?: number; seed_mean_count?: number; mean_detected?: number; edge_ssim?: number;
  reason_seed?: string; reason_variant?: string; poster_missed?: number; poster_seed_count?: number;
};
export type LoopStage = { name?: string; status?: string; t0?: number; t1?: number; seconds?: number; headline?: string; [k: string]: unknown };
export type LoopEntry = {
  kind?: string; iteration?: number; round?: number; mode?: string; started?: number; t?: number; status?: string;
  running_stage?: number | null; stages?: Record<string, LoopStage>; summary?: string; gap_id?: string;
  arms?: string[]; made?: number;
};
export const ago = (t?: number) => {
  if (!t) return "";
  const s = Math.max(0, Date.now() / 1000 - t);
  return s < 60 ? `${Math.round(s)}s ago` : s < 3600 ? `${Math.round(s / 60)}m ago` : `${Math.round(s / 3600)}h ago`;
};
export type SevPoint = { severity: number; recall: number; n?: number };
export type SeverityCurve = { curves?: Record<string, SevPoint[]>; breaking_point?: Record<string, number | null>; by_scene?: unknown; updated?: number };
