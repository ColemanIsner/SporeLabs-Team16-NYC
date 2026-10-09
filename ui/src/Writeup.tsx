// Long-form write-up under the Story on the public page (snapshot mode only). Every number here comes
// from the results JSON in this repo; each phase links its code and outputs on GitHub and shows a real excerpt.
import { useState, type ReactNode } from "react";
import { SNAP } from "./snapshot";

const REPO = "https://github.com/ColemanIsner/SporeLabs-Team16-NYC";
const gh = (p: string) => `${REPO}/blob/main/${p}`;

type Hit = { camera_id?: string; highway: boolean; shows_it: boolean };
type Ask = { query: string; seconds: number; top_k: number; n_shows_it: number; hits: Hit[]; n_synthetic_dropped?: number };
type Cov = { n_indexed: number; n_highway: number; seconds: number; cameras: Record<string, number>; conditions: { id: string; label: string; n_highway: number }[] };

const GAP_QUERIES: [string, string][] = [
  ["clear_day", "highway in clear daytime"], ["night", "highway at night"], ["rain", "highway in heavy rain"],
  ["fog", "highway in dense fog"], ["snow", "highway covered in snow"],
];

type Work = { run: string; trace?: unknown; traceNote?: ReactNode; files: [string, string][] };

function HowItWorks({ id, work }: { id: string; work: Work }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="wu-how">
      <button className="wu-btn" aria-expanded={open} aria-controls={`how-${id}`} onClick={() => setOpen(!open)}>
        {open ? "Hide the work" : "See how it works"} <span aria-hidden="true">{open ? "−" : "+"}</span>
      </button>
      {open && (
        <div className="wu-how-body" id={`how-${id}`}>
          <h5>Run it yourself</h5>
          <pre className="wu-code">{work.run}</pre>
          {work.trace !== undefined && <>
            <h5>From the real run</h5>
            {work.traceNote && <p className="wu-fine">{work.traceNote}</p>}
            <pre className="wu-code wu-trace">{typeof work.trace === "string" ? work.trace : JSON.stringify(work.trace, null, 2)}</pre>
          </>}
          <h5>Code and full outputs on GitHub</h5>
          <ul className="wu-files">
            {work.files.map(([label, path]) => <li key={label + path}><a href={gh(path)} target="_blank" rel="noopener">{label}</a> <code>{path}</code></li>)}
          </ul>
        </div>
      )}
    </div>
  );
}

function Phase({ id, n, name, agent, work, children }: { id: string; n: string; name: string; agent: ReactNode; work: Work; children: ReactNode }) {
  return (
    <section className="wu-phase" id={`phase-${id}`}>
      <h3><span>{n}</span> {name}</h3>
      <div className="wu-cols">
        <div><h4>What the agent does</h4>{agent}</div>
        <div><h4>What it found</h4>{children}</div>
      </div>
      <HowItWorks id={id} work={work} />
    </section>
  );
}

const Src = ({ f }: { f: string }) => <p className="wu-src">Source: <a href={gh(f)} target="_blank" rel="noopener"><code>{f}</code></a></p>;

export default function Writeup() {
  if (!SNAP) return null;
  const cov = SNAP.json["results/coverage.json"] as Cov;
  const ask = SNAP.ask as Record<string, Ask>;
  const tr = ((SNAP as unknown as { traces?: Record<string, unknown> }).traces) ?? {};
  const byId = Object.fromEntries(cov.conditions.map((c) => [c.id, c]));
  return (
    <article className="wu" id="writeup">
      <header className="wu-head">
        <p className="wu-kicker">How it works</p>
        <h2>Each phase of the loop, what the agent did, and what came out</h2>
        <div className="wu-real">
          <p><b>The agent runs and the analysis are real.</b> Every number, trace and clip on this page comes from
            Spore&rsquo;s actual runs on October 9, 2026. A full loop takes hours: generating footage on a GPU and
            putting every clip through two hosted models. So this page replays the recorded outputs instead of running
            live.</p>
          <p>All the code and results are public. <a href={REPO} target="_blank" rel="noopener">Clone the repo</a> and
            run any step yourself; each phase below shows the exact command. I&rsquo;m happy to talk about how I would
            turn this into a product from here: <a href="mailto:coleman@sporelabs.dev">coleman@sporelabs.dev</a>.</p>
          <p className="wu-fine">Spore is a hackathon project. Team 16 built it from scratch in one day at the VAST
            Builders Challenge NYC on the event&rsquo;s VAST, NVIDIA and CoreWeave/W&amp;B stack. It is not a SporeLabs
            product; we host the results here so people can see them.</p>
          <p className="wu-links">
            <a className="wu-btn" href={REPO} target="_blank" rel="noopener">Code and results on GitHub &rarr;</a>
            <a className="wu-btn ghost" href={`${REPO}#reproduce`} target="_blank" rel="noopener">How to reproduce &rarr;</a>
          </p>
        </div>
      </header>

      <section className="wu-run">
        <h3>The real run: what the archive had never seen</h3>
        <p>
          The event&rsquo;s VSS index held <b>{cov.n_indexed}</b> real clips from {Object.keys(cov.cameras).length} cameras.
          Only <b>{cov.n_highway}</b> came from a highway camera (I-24), and every one of them was clear daylight; we
          checked all 30 by eye. Spore asked VSS for each condition, then read the Cosmos Reason caption of every clip it
          returned to check whether the clip really shows it.
        </p>
        <table className="wu-table">
          <thead><tr><th>Condition</th><th>VSS search</th><th>Real highway clips</th><th>VSS top 10: real highway clips that show it</th></tr></thead>
          <tbody>
            {GAP_QUERIES.map(([id, q]) => {
              const r = ask[q]; const c = byId[id];
              const hw = r ? r.hits.filter((h) => h.highway && h.shows_it).length : 0;
              return (
                <tr key={id} className={c?.n_highway === 0 ? "zero" : ""}>
                  <td>{c?.label ?? id}</td>
                  <td><code>{q}</code></td>
                  <td className="num">{c?.n_highway ?? "?"}</td>
                  <td className="num">{!r ? "?" : r.top_k ? `${hw} of ${r.top_k} real` : "0 (all 10 were our synthetic clips)"}</td>
                </tr>
              );
            })}
            {(["glare", "night_rain"] as const).map((id) => byId[id] && (
              <tr key={id} className="zero"><td>{byId[id].label}</td><td>caption scan</td><td className="num">{byId[id].n_highway}</td><td className="num">&ndash;</td></tr>
            ))}
          </tbody>
        </table>
        <p className="wu-fine">
          Searches recorded {SNAP.recorded}. Late that day the team indexed its own synthetic clips in VSS (camera
          <code>spore_synthetic</code>); they are left out here so the table shows the real archive only. Highway
          counts come from the full caption scan ({cov.seconds}s over every clip). The 30 highway clips, checked by eye:{" "}
          <a href={gh("results/verify/i24_all_chunks.jpg")} target="_blank" rel="noopener">contact sheet</a>.
        </p>
        <Src f="results/coverage.json" />
      </section>

      <Phase id="look" n="1" name="Look" agent={<>
        <p><code>vss/coverage.py</code> lists every clip indexed in VSS (<code>GET /api/v1/videos/explore</code>) and
          reads the caption the Cosmos Reason model wrote for it at ingest.</p>
        <p>It groups the clips by camera into scene types (highway, dashcam, city street, residential, warehouse) and
          counts, per scene, how many captions mention each condition. That gives the agent a map of what the archive is
          made of before it looks for gaps.</p></>}
        work={{
          run: "git clone https://github.com/ColemanIsner/SporeLabs-Team16-NYC && cd SporeLabs-Team16-NYC\n# VSS credentials in .env (see README → Environment)\npython3 vss/coverage.py          # -> results/coverage.json, results/inventory.json",
          trace: tr.look, traceNote: "The first three scene types in results/inventory.json:",
          files: [["Code", "vss/coverage.py"], ["VSS client", "vss/client.py"], ["Archive inventory", "results/inventory.json"]],
        }}>
        <p><b>{cov.n_indexed}</b> clips from {Object.keys(cov.cameras).length} cameras: a bike dashcam, city streets in
          New York and San Francisco, a residential street, a warehouse, an indoor space and one highway camera with
          {" "}{cov.n_highway} clips.</p>
        <p>The highway camera is the one a traffic-monitoring customer would care about most, and it is also the
          thinnest.</p>
        <Src f="results/inventory.json" />
      </Phase>

      <Phase id="find" n="2" name="Find weak spots" agent={<>
        <p>For each condition the agent writes one plain-English query and sends it to VSS
          (<code>POST /api/v1/search</code>). Search ranking alone can return a clip that merely looks similar, so it
          also scans every caption for the condition&rsquo;s keywords and counts only clips whose caption actually says
          it.</p>
        <p>The answer is a count per condition and per scene: what the cameras have seen, and what they never
          have.</p></>}
        work={{
          run: "python3 vss/coverage.py                       # all conditions, every caption\npython3 vss/ask.py \"highway in dense fog\"      # one live VSS search, judged by caption",
          trace: tr.find, traceNote: "One recorded VSS search: none of the real hits is a highway clip.",
          files: [["Coverage search", "vss/coverage.py"], ["Ask the archive", "vss/ask.py"], ["Coverage result", "results/coverage.json"]],
        }}>
        <p>On the highway camera: 30 clear-day clips and <b>0</b> for night, heavy rain, dense fog, snow, low-sun glare
          and night with rain. Snow appears on no camera at all: none of the captions across 13 cameras mentions it.</p>
        <Src f="results/coverage.json" />
      </Phase>

      <Phase id="decide" n="3" name="Decide" agent={<>
        <p><code>loop/report.py</code> hands the coverage counts and the latest test results to an LLM on W&amp;B
          Inference (<code>deepseek-ai/DeepSeek-V4-Pro-0813</code>). It writes a gap report, explains the risk, ranks
          the candidate gaps and picks the next condition to fill.</p>
        <p>Every call is traced in W&amp;B Weave, so each decision can be audited later.</p></>}
        work={{
          run: "loop/.venv/bin/python loop/report.py          # -> results/gap_report.json (Weave-traced)\nloop/.venv/bin/python loop/spore.py --stage 2  # same step inside the loop",
          trace: tr.decide, traceNote: <>The LLM&rsquo;s recorded report. The Weave traces are in the{" "}
            <a href="https://wandb.ai/colemanisner-sporelabs/sporelabs-hackathon/weave" target="_blank" rel="noopener">W&amp;B project</a> (sign-in may be required).</>,
          files: [["Code", "loop/report.py"], ["LLM client", "loop/llm.py"], ["Weave tracing", "loop/tracing.py"], ["Gap report", "results/gap_report.json"]],
        }}>
        <p>Latest pick: <b>low-sun glare on the highway</b> (priority 0.85). Highway cameras have 0 real glare clips and
          no synthetic tests yet, and glare can hide vehicles from both search and detection. An earlier iteration
          picked heavy rain. The call took 15.3 s.</p>
        <Src f="results/gap_report.json" />
      </Phase>

      <Phase id="grow" n="4" name="Grow data" agent={<>
        <p>The agent pulls 11 real seed clips from VSS (9 from the I-24 highway camera, 2 from NYC) and grows the missing
          conditions onto them with two generators:</p>
        <ul>
          <li><b>Weather layer</b> (<code>gen/weather.py</code>): fog, rain and snow drawn over the frame at severity
            0.4, 0.7 and 1.0. It never moves a pixel of the scene, so every car stays where it was and the seed&rsquo;s
            labels remain exact. About 5 s per clip on a laptop.</li>
          <li><b>NVIDIA Cosmos Transfer 2.5</b> (2B edge-distilled, on a Modal H100): night, night with rain, fog and
            snow re-rendered from the seed&rsquo;s edges, plus a clear-day control that should change nothing. About
            two minutes per clip.</li>
        </ul></>}
        work={{
          run: "# weather layer, one clip\npython3 gen/weather.py --in data/seeds/i24_scene1_p1c2_00.mp4 --kind fog --severity 0.4 \\\n  --out data/synthetic/i24_scene1_p1c2_00__phys_fog_s04.mp4 \\\n  --manifest data/synthetic/manifest.json --seed-id i24_scene1_p1c2_00\n\n# Cosmos Transfer 2.5 on Modal\nmodal deploy gen/modal_transfer.py\nSPORE_CONDS=fog_day_heavy,rain_night_heavy,snow_day_heavy,clear_night_light,clear_day_light python3 gen/first_batch.py",
          trace: tr.grow, traceNote: "A seed clip, then one weather-layer variant and one Cosmos Transfer variant with its real prompt, from the manifests:",
          files: [["Weather layer", "gen/weather.py"], ["Cosmos Transfer on Modal", "gen/modal_transfer.py"], ["Seed clips", "data/seeds/manifest.json"], ["All 160 variants", "data/synthetic/manifest.json"]],
        }}>
        <table className="wu-table">
          <thead><tr><th>Generator</th><th>Clips</th></tr></thead>
          <tbody>
            <tr><td>Weather layer, 3 conditions &times; 3 severities &times; 11 seeds</td><td className="num">99</td></tr>
            <tr><td>Weather layer, loop iterations (rain 0.8, glare 0.8)</td><td className="num">6</td></tr>
            <tr><td>Cosmos Transfer, 4 conditions + clear-day control</td><td className="num">55</td></tr>
            <tr className="tot"><td>Synthetic clips</td><td className="num">160</td></tr>
          </tbody>
        </table>
        <Src f="data/synthetic/manifest.json" />
      </Phase>

      <Phase id="test" n="5" name="Test" agent={<>
        <p>Every clip goes through the two models inside the VSS pipeline: the hosted YOLO11s detector and the hosted
          Cosmos3 Nano Reasoner, which answers fixed questions (vehicle count, lane changes, stopped vehicles,
          weather).</p>
        <p>Two gates keep the test honest:</p>
        <ul>
          <li><b>Fidelity:</b> the variant&rsquo;s edges must match the seed (edge-SSIM &ge; 0.5).</li>
          <li><b>Vehicle integrity:</b> at least 70% of the seed&rsquo;s cars must still be visibly there, so a miss
            counts only on a car that exists.</li>
        </ul>
        <p>Recall is the share of cars the detector found on the clear seed that it still finds under the condition.
          Then the agent searches the real archive for each condition and runs the same two models on what it finds.</p></>}
        work={{
          run: "eval/.venv/bin/python eval/run_all.py --reason   # YOLO11s + Cosmos3 Reasoner + fidelity -> results/evals/\neval/.venv/bin/python eval/integrity.py          # vehicle-integrity gate\neval/.venv/bin/python eval/severity_curve.py     # -> results/severity_curve.json\neval/.venv/bin/python eval/real_confirm.py       # same models on real archive clips",
          trace: tr.test, traceNote: "The full record for one clip (seed i24 p1c2, weather-layer fog 0.4), as written to results/evals/:",
          files: [["Run all evals", "eval/run_all.py"], ["Integrity gate", "eval/integrity.py"], ["Severity curve code", "eval/severity_curve.py"],
            ["Per-clip results", "results/evals/i24_scene1_p1c2_00__phys_fog_s04.json"], ["Severity curve", "results/severity_curve.json"],
            ["Cosmos arms", "results/failure_map.json"], ["Real-footage check", "results/real_check.json"]],
        }}>
        <table className="wu-table">
          <thead><tr><th>Weather layer</th><th>0.4</th><th>0.7</th><th>1.0</th><th>Below 50% at</th></tr></thead>
          <tbody>
            <tr><td>Snow</td><td className="num">0.41</td><td className="num">0.28</td><td className="num">0.21</td><td className="num"><b>0.34</b></td></tr>
            <tr><td>Fog</td><td className="num">0.43</td><td className="num">0.30</td><td className="num">0.06</td><td className="num"><b>0.35</b></td></tr>
            <tr><td>Rain</td><td className="num">0.71</td><td className="num">0.39</td><td className="num">0.14</td><td className="num"><b>0.60</b></td></tr>
          </tbody>
        </table>
        <p className="wu-fine">11 clips per cell. The highway camera is far more fragile than the NYC street camera:
          at fog 0.4, recall is 0.32 on I-24 and 0.96 on NYC.</p>
        <Src f="results/severity_curve.json" />
        <p>Cosmos Transfer costs recall on its own, so its arms are compared with the clear-day control (recall 0.81):
          night with rain keeps 60% of the control&rsquo;s recall, clear night 86%, fog 92% and snow 95% (6 clips each).</p>
        <Src f="results/failure_map.json" />
        <p>Real footage shows the same signature. On real glare, dusk and rain clips pulled from the archive, the
          detector fell well short of the Reasoner, as it does on grown footage; glare is the clearest case, with YOLO
          finding a quarter of what Reason counts. There is no real snow footage anywhere, so grown footage is the only
          way to test it.</p>
        <Src f="results/real_check.json" />
      </Phase>

      <Phase id="fix" n="6" name="Fix: the training data" agent={<>
        <p><code>fix/build_dataset.py</code> turns the grown clips into a detector training set. It takes every third
          frame and copies the clear seed&rsquo;s vehicle boxes onto the same frame of each variant. The weather layer
          keeps cars exactly in place, and Cosmos clips must pass the fidelity gate. Cameras are split so the test
          camera (I-24 p1c3) and NYC are never trained on.</p>
        <p><code>fix/train_modal.py</code> fine-tuned YOLO11s on it for 10 epochs on a Modal L40S (136 s), and
          {" "}<code>fix/evaluate.py</code> compared it with the stock model and a no-training contrast fix.</p></>}
        work={{
          run: "eval/.venv/bin/python fix/build_dataset.py --stride 3 --out fix/dataset\neval/.venv/bin/modal run fix/train_modal.py --epochs 10 --imgsz 960\neval/.venv/bin/python fix/evaluate.py --imgsz 960   # -> results/fix/fix_detail.json",
          trace: tr.fix, traceNote: "Frames per condition and split (fix/dataset/meta.json, built locally; rebuild it with the first command), the training run and its log:",
          files: [["Build the dataset", "fix/build_dataset.py"], ["Train on Modal", "fix/train_modal.py"], ["Evaluate", "fix/evaluate.py"],
            ["Seed labels", "fix/seed_labels/i24_scene1_p1c1_00.json"], ["Result, per clip", "results/fix/fix_detail.json"], ["Training log", "results/fix/train_log.json"]],
        }}>
        <table className="wu-table">
          <thead><tr><th>Training set</th><th>Train</th><th>Val</th></tr></thead>
          <tbody>
            <tr><td>Clear seeds</td><td className="num">155</td><td className="num">31</td></tr>
            <tr><td>Cosmos Transfer (4 conditions)</td><td className="num">620</td><td className="num">124</td></tr>
            <tr><td>Weather layer (fog, rain, snow &times; 3 severities)</td><td className="num">784</td><td className="num">144</td></tr>
            <tr className="tot"><td>Labelled frames</td><td className="num">1,559</td><td className="num">299</td></tr>
          </tbody>
        </table>
      </Phase>

      <figure className="wu-fig">
        <img src="/media/spore-training.jpg" alt="Eight training frames from one I-24 seed: clear, weather-layer fog, rain and snow, and Cosmos Transfer night, night rain, fog and snow, each with the same vehicle boxes" loading="lazy" />
        <figcaption>One seed frame in the training set: the clear original, three weather-layer variants and four Cosmos
          Transfer variants. The boxes are the seed&rsquo;s labels, copied unchanged: exact on the weather layer, which never moves a car, but only approximate on Cosmos Transfer, which re-renders the scene.</figcaption>
      </figure>

      <section className="wu-phase">
        <h3><span>6</span> Fix: what the fine-tune showed</h3>
        <table className="wu-table">
          <thead><tr><th>Held-out set</th><th>Stock</th><th>Fine-tuned</th><th>Contrast fix, no training</th></tr></thead>
          <tbody>
            <tr><td>60 synthetic variants, recall</td><td className="num">0.318</td><td className="num">0.310</td><td className="num"><b>0.363</b></td></tr>
            <tr><td>Held-out camera I-24 p1c3 (42 clips), recall</td><td className="num">0.161</td><td className="num">0.205</td><td className="num">0.202</td></tr>
            <tr><td>Clear seeds (5), recall</td><td className="num">1.000</td><td className="num">0.763</td><td className="num">0.956</td></tr>
          </tbody>
        </table>
        <p>The ship gate was: beat stock on held-out variants without losing more than 5 points on clear footage. A
          free contrast step (CLAHE) gave the best recall on held-out variants. At this scale (6 training seeds, highly
          correlated frames) the grown footage is most valuable as a stress test, and every frame already comes labeled
          for larger training runs.</p>
        <Src f="results/fix/fix_detail.json" />
      </section>

      <Phase id="repeat" n="↻" name="Repeat" agent={<>
        <p><code>loop/spore.py</code> chains the phases: Look &rarr; Find &rarr; Decide &rarr; Grow &rarr; Test, then
          picks the next weak spot and goes again. It logs every stage with its timing and outputs.</p></>}
        work={{
          run: "loop/.venv/bin/python loop/spore.py --iterations 2 --k 3   # full loop\nloop/.venv/bin/python loop/spore.py --stage 1               # one stage",
          trace: tr.repeat, traceNote: "The last logged loop iteration, every stage with its timing:",
          files: [["The loop", "loop/spore.py"], ["Loop log", "results/loop_log.json"]],
        }}>
        <p>Four iterations ran on the day. Re-searching the archive found no new real footage for the open gaps, so the
          loop kept filling them with grown footage.</p>
        <Src f="results/loop_log.json" />
      </Phase>

      <section className="wu-next">
        <h3>From hackathon to product</h3>
        <p>What ran here is the loop by hand, in a day: one archive, one detector, a few conditions. Turning it into a
          product means running it continuously against a customer&rsquo;s own archive and models, and proving each fix
          before it ships. I&rsquo;m happy to talk about how I would build that:{" "}
          <a href="mailto:coleman@sporelabs.dev">coleman@sporelabs.dev</a>.</p>
      </section>

      <footer className="wu-foot">
        <p>Built with VAST Data (VSS search and captions), NVIDIA (Cosmos Transfer 2.5, Cosmos3 Nano Reasoner, YOLO11s
          hosting), CoreWeave and Weights &amp; Biases (W&amp;B Inference and Weave), and Cursor.</p>
        <p><a href={REPO} target="_blank" rel="noopener">github.com/ColemanIsner/SporeLabs-Team16-NYC</a> &middot; <a href="/research">&larr; SporeLabs Research</a></p>
      </footer>
    </article>
  );
}
