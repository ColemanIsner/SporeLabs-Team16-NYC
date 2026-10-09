import { createContext, useContext, useEffect, useRef, useState, type ReactNode } from "react";

/** Keeps every registered <video> on the same clock as the leader (first registered, or one flagged `leader`). */
type Ctx = { register: (v: HTMLVideoElement, leader?: boolean) => () => void; playing: boolean; toggle: () => void; restart: () => void };
const SyncCtx = createContext<Ctx | null>(null);

export function SyncGroup({ children }: { children: (ctl: { playing: boolean; toggle: () => void; restart: () => void }) => ReactNode }) {
  const vids = useRef<{ v: HTMLVideoElement; leader: boolean }[]>([]);
  const [playing, setPlaying] = useState(true);
  const playingRef = useRef(true);

  const leader = () => (vids.current.find((x) => x.leader) ?? vids.current[0])?.v;

  useEffect(() => {
    const id = setInterval(() => {
      const L = leader();
      if (!L || L.readyState < 2) return;
      const t = L.currentTime;
      for (const { v } of vids.current) {
        if (v === L || v.readyState < 2) continue;
        const d = v.duration || L.duration || 0;
        let diff = Math.abs(v.currentTime - t);
        if (d) diff = Math.min(diff, d - diff);
        if (diff > 0.12 && (!d || t < d - 0.05)) { try { v.currentTime = d ? Math.min(t, d - 0.01) : t; } catch { /* ignore */ } }
        if (playingRef.current && v.paused) v.play().catch(() => {});
      }
      if (playingRef.current && L.paused) L.play().catch(() => {});
    }, 250);
    return () => clearInterval(id);
  }, []);

  const ctx: Ctx = {
    register(v, isLeader = false) {
      vids.current.push({ v, leader: isLeader });
      const L = leader();
      if (L && L !== v && L.readyState >= 2) { try { v.currentTime = L.currentTime; } catch { /* */ } }
      if (playingRef.current) v.play().catch(() => {}); else v.pause();
      return () => { vids.current = vids.current.filter((x) => x.v !== v); };
    },
    playing,
    toggle() {
      playingRef.current = !playingRef.current;
      setPlaying(playingRef.current);
      for (const { v } of vids.current) playingRef.current ? v.play().catch(() => {}) : v.pause();
    },
    restart() { for (const { v } of vids.current) { try { v.currentTime = 0; } catch { /* */ } } },
  };
  return <SyncCtx.Provider value={ctx}>{children({ playing, toggle: ctx.toggle, restart: ctx.restart })}</SyncCtx.Provider>;
}

export function Video({ src, leader, className }: { src?: string; leader?: boolean; className?: string }) {
  const ref = useRef<HTMLVideoElement>(null);
  const ctx = useContext(SyncCtx);
  const [err, setErr] = useState(false);
  useEffect(() => { setErr(false); }, [src]);
  useEffect(() => {
    const v = ref.current;
    if (!v || !ctx || err) return;
    return ctx.register(v, leader);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [src, leader, err]);
  if (!src || err) {
    return (
      <div className={`video-missing ${className ?? ""}`}>
        <span>{src ? "clip unavailable" : "no clip"}</span>
      </div>
    );
  }
  return (
    <video
      ref={ref}
      className={className}
      src={src}
      muted
      loop
      autoPlay
      playsInline
      preload="auto"
      onError={() => setErr(true)}
    />
  );
}
