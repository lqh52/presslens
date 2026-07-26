"use client";

import {
  ArrowLeft, ChevronDown, CircleHelp, Clock3, Film,
  Filter, Info, Menu, Play, Search, Sparkles, Target, X,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import type { Clip, DemoManifest, Situation } from "@/lib/types";
import {
  hybridSearch,
  type BrowserSearchIndex,
} from "@/lib/browser-retrieval";

const situations: Array<{ value: Situation | "all"; label: string }> = [
  { value: "all", label: "All situations" },
  { value: "high_press", label: "High press" },
  { value: "central_screen", label: "Central screen" },
  { value: "trap_left", label: "Left trap" },
  { value: "trap_right", label: "Right trap" },
  { value: "unstructured", label: "No local pressure" },
];

const suggestedQueries = [
  "high press around the build-up",
  "central passing lanes are screened",
  "no local pressure around the ball",
];

const SEARCH_STATE_KEY = "presslens-search-state";
const SITE_BASE_PATH = (process.env.NEXT_PUBLIC_BASE_PATH ?? "").replace(/\/$/, "");
const MEDIA_ASSET_BASE = (
  process.env.NEXT_PUBLIC_MEDIA_ASSET_BASE_URL ?? ""
).replace(/\/$/, "");

const staticPath = (path: string) => `${SITE_BASE_PATH}${path}`;
const mediaPath = (path: string) => MEDIA_ASSET_BASE && path.startsWith("/demo/")
  ? `${MEDIA_ASSET_BASE}/${path.slice("/demo/".length)}`
  : staticPath(path);

const resolveManifestAssets = (manifest: DemoManifest): DemoManifest => ({
  ...manifest,
  videos: manifest.videos.map((video) => ({
    ...video,
    path: mediaPath(video.path),
  })),
  clips: manifest.clips.map((clip) => ({
    ...clip,
    video: mediaPath(clip.video),
    canonicalImage: mediaPath(clip.canonicalImage),
    canonicalVideo: mediaPath(clip.canonicalVideo),
    thumbnail: mediaPath(clip.thumbnail),
  })),
});
type RetrievalResult = {
  id: string;
  score: number;
  cosine?: number;
  bm25?: number;
};

export function PressLens() {
  const [manifest, setManifest] = useState<DemoManifest | null>(null);
  const [searchIndex, setSearchIndex] = useState<BrowserSearchIndex | null>(null);
  const [loadError, setLoadError] = useState("");
  const [query, setQuery] = useState("");
  const [submittedQuery, setSubmittedQuery] = useState("");
  const [selected, setSelected] = useState<Clip | null>(null);
  const [situation, setSituation] = useState<Situation | "all">("all");
  const [reliablePossession, setReliablePossession] = useState(false);
  const [activeTab, setActiveTab] = useState<"retrieval" | "evidence">("retrieval");
  const [vectorResults, setVectorResults] = useState<RetrievalResult[] | null>(null);
  const [searching, setSearching] = useState(false);
  const [searchError, setSearchError] = useState("");
  const [notesOpen, setNotesOpen] = useState(false);
  const broadcastRef = useRef<HTMLVideoElement>(null);
  const canonicalRef = useRef<HTMLVideoElement>(null);
  const searchRequestRef = useRef<AbortController | null>(null);
  const searchSequenceRef = useRef(0);

  useEffect(() => {
    Promise.all([
      fetch(staticPath("/demo/manifest.json")),
      fetch(staticPath("/demo/search-index.json")),
    ])
      .then(async ([manifestResponse, indexResponse]) => {
        if (!manifestResponse.ok) {
          throw new Error(`Dataset returned ${manifestResponse.status}`);
        }
        if (!indexResponse.ok) {
          throw new Error(`Search index returned ${indexResponse.status}`);
        }
        return [
          resolveManifestAssets(await manifestResponse.json() as DemoManifest),
          await indexResponse.json() as BrowserSearchIndex,
        ] as const;
      })
      .then(([data, index]) => {
        setManifest(data);
        setSearchIndex(index);
        try {
          const saved = JSON.parse(sessionStorage.getItem(SEARCH_STATE_KEY) ?? "null") as {
            query?: string;
            submittedQuery?: string;
            situation?: Situation | "all";
            reliablePossession?: boolean;
            vectorResults?: RetrievalResult[];
            selectedId?: string;
          } | null;
          const validIds = new Set(data.clips.map((clip) => clip.id));
          const savedResults = saved?.vectorResults?.filter((row) => validIds.has(row.id)) ?? null;
          setQuery(saved?.query ?? "");
          setSubmittedQuery(saved?.submittedQuery ?? "");
          setSituation(saved?.situation ?? "all");
          setReliablePossession(saved?.reliablePossession ?? false);
          setVectorResults(savedResults?.length ? savedResults : null);
          setSelected(
            data.clips.find((clip) => clip.id === saved?.selectedId)
              ?? data.clips[0]
              ?? null,
          );
        } catch {
          sessionStorage.removeItem(SEARCH_STATE_KEY);
          setSelected(data.clips[0] ?? null);
        }
      })
      .catch((error: Error) => setLoadError(error.message));
  }, []);

  useEffect(() => {
    if (!manifest) return;
    sessionStorage.setItem(SEARCH_STATE_KEY, JSON.stringify({
      query,
      submittedQuery,
      situation,
      reliablePossession,
      vectorResults,
      selectedId: selected?.id,
    }));
  }, [
    manifest,
    query,
    submittedQuery,
    situation,
    reliablePossession,
    vectorResults,
    selected,
  ]);

  useEffect(() => {
    if (!notesOpen) return;
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setNotesOpen(false);
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [notesOpen]);

  const results = useMemo(() => {
    if (!manifest) return [];
    if (searching) return [];
    const scores = new Map(vectorResults?.map((row) => [row.id, row.score]) ?? []);
    const order = new Map(vectorResults?.map((row, index) => [row.id, index]) ?? []);
    return manifest.clips
      .filter((clip) => !vectorResults || scores.has(clip.id))
      .map((clip) => ({
        ...clip,
        score: vectorResults ? Math.round(scores.get(clip.id)! * 100) : clip.confidence,
      }))
      .filter((clip) => (situation === "all" || clip.situation === situation)
        && (!reliablePossession || clip.possessionConfident))
      .sort((a, b) => vectorResults
        ? (order.get(a.id) ?? 999) - (order.get(b.id) ?? 999)
        : b.confidence - a.confidence);
  }, [manifest, vectorResults, situation, reliablePossession, searching]);

  const runSearch = async (value = query) => {
    const sequence = ++searchSequenceRef.current;
    searchRequestRef.current?.abort();
    const controller = new AbortController();
    searchRequestRef.current = controller;
    setQuery(value);
    const clean = value.trim();
    setSubmittedQuery(clean);
    setSearchError("");
    setVectorResults(null);
    setSituation("all");
    setReliablePossession(false);
    setActiveTab("retrieval");
    broadcastRef.current?.pause();
    canonicalRef.current?.pause();
    if (!manifest || !searchIndex || !clean) {
      setVectorResults(null);
      if (manifest) setSelected(manifest.clips[0] ?? null);
      setSearching(false);
      searchRequestRef.current = null;
      return;
    }
    setSearching(true);
    try {
      const results = await hybridSearch(clean, searchIndex);
      if (controller.signal.aborted) return;
      const payload = { results };
      if (sequence !== searchSequenceRef.current) return;
      setVectorResults(payload.results);
      const first = payload.results
        .map((row) => manifest.clips.find((clip) => clip.id === row.id))
        .find((clip) => clip !== undefined);
      if (first) setSelected(first);
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") return;
      if (sequence !== searchSequenceRef.current) return;
      setSearchError(error instanceof Error ? error.message : "Vector search failed");
    } finally {
      if (sequence === searchSequenceRef.current) {
        setSearching(false);
        searchRequestRef.current = null;
      }
    }
  };

  const selectClip = (clip: Clip) => {
    setSelected(clip);
    setActiveTab("evidence");
  };
  const selectedRetrievalScore = vectorResults?.find((row) => row.id === selected?.id)?.score;
  const syncCanonical = () => {
    const broadcast = broadcastRef.current;
    const canonical = canonicalRef.current;
    if (!broadcast || !canonical) return;
    if (Math.abs(canonical.currentTime - broadcast.currentTime) > 0.04) {
      canonical.currentTime = broadcast.currentTime;
    }
  };

  if (loadError) {
    return <main className="load-state"><strong>Dataset unavailable</strong><span>{loadError}</span><code>python scripts/build_reviewed_web_demo.py</code></main>;
  }
  if (!manifest || !selected) {
    return <main className="load-state"><span className="loading-dot" /> Loading reconstructed game states…</main>;
  }

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark"><Target size={21} strokeWidth={2.4} /></div>
          <div><strong>PressLens</strong><span>Football intelligence</span></div>
        </div>
        <nav>
          <button className={`nav-item ${activeTab === "retrieval" ? "nav-item--active" : ""}`} onClick={() => setActiveTab("retrieval")}><Search size={18} /> Tactical retrieval</button>
        </nav>
        <div className="sidebar-project">
          <span className="eyebrow">Runtime</span>
          <div className="project-row"><span className="status-dot" /> Models on local node</div>
          <span>GSR + graph classifier · {manifest.videoCount} videos</span>
        </div>
        <div className="sidebar-bottom">
          <button onClick={() => setNotesOpen(true)}><CircleHelp size={17} /> README</button>
          <div className="profile"><span>AI</span><div><strong>Football intelligence</strong><small>Real broadcast reconstruction</small></div></div>
        </div>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <button className="mobile-menu" aria-label="Open menu"><Menu size={20} /></button>
          <div className="crumbs"><strong>Tactical retrieval.</strong></div>
          <div className="top-actions">
            <span className="model-pill"><Sparkles size={14} /> MiniLM + BM25 · {manifest.videoCount} videos</span>
          </div>
        </header>

        <div className="content">
          {activeTab === "retrieval" && <>
          <section className="intro" id="search">
            <div>
              <span className="eyebrow">Real-video tactical retrieval</span>
              <h1>Search the press.<br />Inspect the evidence.</h1>
            </div>
            <div className="dataset-summary">
              <span><Film size={16} /> {manifest.count} states from {manifest.videoCount} videos</span>
              <span><Clock3 size={16} /> {manifest.matchCount ?? 1} matches</span>
            </div>
          </section>

          <section className="search-panel">
            <form onSubmit={(event) => { event.preventDefault(); runSearch(); }}>
              <Search size={21} />
              <input value={query} onChange={(event) => setQuery(event.target.value)}
                placeholder="e.g. Find dense high pressure around the build-up"
                aria-label="Search tactical situations" />
              {query && <button type="button" className="clear-query" onClick={() => { setQuery(""); setSubmittedQuery(""); setVectorResults(null); }}><X size={16} /></button>}
              <button type="submit" className="search-button">{searching ? "Update search…" : "Retrieve situations"}</button>
            </form>
            {searchError && <div className="search-error">{searchError}</div>}
            <div className="suggestions"><span>Try</span>{suggestedQueries.map((item) =>
              <button key={item} onClick={() => runSearch(item)}>{item}</button>)}</div>
          </section>

          <section className="analysis-grid analysis-grid--retrieval">
            <div className="results-column">
              <div className="section-heading">
                <div><h2>{submittedQuery ? "Ranked matches" : "Highest-confidence states"}</h2>
                  <p>{results.length} results · semantic cosine + lexical BM25</p></div>
                <div className="filters">
                  <label className="visibility-toggle"><input type="checkbox" checked={reliablePossession}
                    onChange={(event) => setReliablePossession(event.target.checked)} /> Reliable possession</label>
                  <label className="select-wrap"><Filter size={15} />
                    <select value={situation} onChange={(event) => setSituation(event.target.value as Situation | "all")}>
                      {situations.filter((item) => item.value === "all"
                        || manifest.clips.some((clip) => clip.situation === item.value))
                        .map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}
                    </select><ChevronDown size={14} />
                  </label>
                </div>
              </div>

              <div className="result-list" id="library">
                {results.map((clip, index) => (
                  <button className={`result-card ${selected.id === clip.id ? "result-card--selected" : ""}`}
                    key={clip.id} onClick={() => selectClip(clip)}>
                    <div className="result-rank">{String(index + 1).padStart(2, "0")}</div>
                    <div className="result-pitch"><img className="frame-thumb" src={clip.thumbnail} alt="" />
                      <span className="play-chip"><Play size={10} fill="currentColor" /> 4 sec video</span></div>
                    <div className="result-copy">
                      <div className="result-meta"><span>{clip.id}</span><span>{clip.minute}</span>
                        <span className={`outcome outcome--${clip.situation}`}>{clip.title}</span></div>
                      <strong>{clip.description}</strong>
                      <p>{clip.evidence.slice(0, 2).join(" · ")}</p>
                      <div className="evidence-tags"><span>{clip.visibleNodes} visible nodes</span>
                        <span>{clip.possessionClub} in possession</span>
                        <span>{clip.attackDirection.replaceAll("_", " → ")}</span>
                        <span>{clip.possessionConfident ? "possession reliable" : "possession uncertain"}</span></div>
                    </div>
                    <div className="similarity"><strong>{clip.score}%</strong><span>{vectorResults ? "hybrid match" : "class conf."}</span></div>
                  </button>
                ))}
                {searching
                  ? <div className="empty-state"><span className="loading-dot" /><strong>Updating retrieval results…</strong><span>The previous ranking has been cleared.</span></div>
                  : !results.length && <div className="empty-state"><Search size={22} /><strong>No states match these filters</strong><span>Clear a filter or try broader tactical language.</span></div>}
              </div>
            </div>
          </section>
          </>}
          {activeTab === "evidence" &&
          <section className="analysis-grid analysis-grid--evidence">
            <aside className="detail-panel" id="patterns">
              <div className="evidence-nav">
                <button onClick={() => setActiveTab("retrieval")}><ArrowLeft size={15} /> Back to retrieval</button>
                <span>{selected.id} · {selected.minute}</span>
              </div>
              <div className="detail-topline"><span className="eyebrow">Selected model evidence</span></div>
              <div className="confidence-explainer">
                <div><span>Retrieval match</span><strong>{selectedRetrievalScore === undefined ? "—" : `${Math.round(selectedRetrievalScore * 100)}%`}</strong><small>65% normalized semantic cosine plus 35% normalized BM25 lexical relevance. Not a probability.</small></div>
                <div><span>Classification confidence</span><strong>{selected.confidence}%</strong><small>Mean classifier probability for the accepted class across the selected excerpt.</small></div>
                <div><span>Majority support</span><strong>{selected.majorityFrames}/{selected.validFrames}</strong><small>Valid frames assigned to this video’s final majority label.</small></div>
              </div>
              <div className="evidence-media-grid">
                <div><span className="media-label">Annotated broadcast · {selected.videoId}</span><div className="viewer">
                  <>
                    <video
                      key={selected.id}
                      ref={broadcastRef}
                      controls
                      preload="metadata"
                      poster={selected.thumbnail}
                      src={selected.video}
                      onLoadedMetadata={(event) => {
                        event.currentTarget.currentTime = selected.timeSeconds;
                        if (canonicalRef.current) canonicalRef.current.currentTime = selected.timeSeconds;
                      }}
                      onPlay={() => {
                        syncCanonical();
                        canonicalRef.current?.play();
                      }}
                      onPause={() => canonicalRef.current?.pause()}
                      onSeeking={syncCanonical}
                      onTimeUpdate={syncCanonical}
                      onRateChange={(event) => {
                        if (canonicalRef.current) canonicalRef.current.playbackRate = event.currentTarget.playbackRate;
                      }}
                    />
                    <div className="viewer-overlay">
                      <span><i className="legend-dot legend-dot--team-a" /> Team A</span>
                      <span><i className="legend-dot legend-dot--team-b" /> Team B</span>
                      <span><i className="legend-line" /> Pressure ≤12m</span>
                    </div>
                  </>
                </div></div>
                <div><span className="media-label">Synchronized canonical video · {selected.videoId} <b className="media-format">MP4 · 25 FPS</b></span><div className="viewer canonical-viewer">
                  <video
                    key={`canonical-${selected.id}`}
                    ref={canonicalRef}
                    controls
                    muted
                    playsInline
                    preload="auto"
                    poster={selected.canonicalImage}
                    src={selected.canonicalVideo}
                    onLoadedMetadata={(event) => {
                      event.currentTarget.currentTime = broadcastRef.current?.currentTime ?? selected.timeSeconds;
                    }}
                    onPlay={() => {
                      const broadcast = broadcastRef.current;
                      const canonical = canonicalRef.current;
                      if (!broadcast || !canonical) return;
                      if (Math.abs(broadcast.currentTime - canonical.currentTime) > 0.04) {
                        broadcast.currentTime = canonical.currentTime;
                      }
                      broadcast.play();
                    }}
                    onPause={() => broadcastRef.current?.pause()}
                    onSeeking={(event) => {
                      if (broadcastRef.current) broadcastRef.current.currentTime = event.currentTarget.currentTime;
                    }}
                    onRateChange={(event) => {
                      if (broadcastRef.current) broadcastRef.current.playbackRate = event.currentTarget.playbackRate;
                    }}
                  />
                  <div className="viewer-overlay"><span><i className="legend-dot legend-dot--press" /> Pressing</span><span><i className="legend-dot legend-dot--build" /> Possession</span></div>
                </div></div>
              </div>
              <div className="canonical-note"><Info size={14} /><span>Either player controls both synchronized four-second videos. The canonical view uses calibrated pitch coordinates from each corresponding broadcast frame.</span></div>

              <div className="detail-title"><div><h3>{selected.title}</h3><p>{selected.match} · {selected.videoId} · frame {selected.frame}</p></div>
                <span className="confidence"><strong>{selected.confidence}%</strong><small>Classification confidence</small></span></div>
              <p className="detail-description">{selected.description}</p>
              <div className="metrics">
                <div><span>Visible nodes</span><strong>{selected.visibleNodes} / 23</strong></div>
                <div><span>Ball detector</span><strong>{selected.ballConfidence}%</strong></div>
                <div><span>Possession</span><strong>{selected.possessionConfident ? "Reliable" : "Uncertain"}</strong></div>
                <div><span>Possession team</span><strong>{selected.possessionClub}</strong></div>
                <div><span>Pressing team</span><strong>{selected.pressingClub}</strong></div>
                <div><span>Attack direction</span><strong>{selected.attackDirection === "left_to_right" ? "Left → right" : selected.attackDirection === "right_to_left" ? "Right → left" : "Undetermined"}</strong></div>
                <div><span>Ball-holder distance</span><strong>{selected.ballHolderDistanceM == null ? "—" : `${selected.ballHolderDistanceM.toFixed(1)} m`}</strong></div>
              </div>
              {selected.orientationValidated && <div className="orientation-valid"><span className="status-dot" /> Possession side and attacking direction reviewed</div>}
              <div className="evidence-block"><h4>Geometric evidence</h4>
                {selected.evidence.map((item, index) => <div className="evidence-row" key={item}><span>{index + 1}</span>
                  <div><strong>{item}</strong><small>{index < 3 ? "Measured from reconstructed pitch state" : "Independent weak-label rule"}</small></div></div>)}
              </div>
              <div className="probability-block">
                <h4>Class probabilities</h4>
                {situations.slice(1).map((item) => {
                  const value = selected.probabilities[item.value as Situation] * 100;
                  return <div className="probability-row" key={item.value}><span>{item.label}</span><i><b style={{ width: `${value}%` }} /></i><strong>{value.toFixed(1)}%</strong></div>;
                })}
              </div>
            </aside>
          </section>
          }
        </div>
      </section>
      {notesOpen &&
        <div className="modal-backdrop" onMouseDown={(event) => {
          if (event.target === event.currentTarget) setNotesOpen(false);
        }}>
          <section className="research-modal" role="dialog" aria-modal="true" aria-labelledby="research-notes-title">
            <button className="modal-close" onClick={() => setNotesOpen(false)} aria-label="Close research notes"><X size={19} /></button>
            <span className="eyebrow">Football intelligence</span>
            <h2 id="research-notes-title">Research notes</h2>
            <p>PressLens retrieves short tactical situations from reconstructed broadcast video. Search results combine semantic cosine similarity with BM25 lexical matching.</p>
            <h3>Supported classes</h3>
            <dl className="class-guide">
              <div><dt>Central screen</dt><dd>The defending shape occupies central progression lanes ahead of the ball, discouraging or blocking a direct pass through the middle.</dd></div>
              <div><dt>High press</dt><dd>Several defenders compress the ball area while the possession team builds in its defensive third.</dd></div>
              <div><dt>No local pressure</dt><dd>The reconstructed state does not show coordinated pressure close to the ball; the nearest pressure is distant or structurally unclear.</dd></div>
            </dl>
            <h3>How to use the app</h3>
            <ol>
              <li>Describe a tactical situation in the search field, or choose a suggested query.</li>
              <li>Use the situation and possession filters to narrow the ranked results.</li>
              <li>Select a result to open its synchronized broadcast and canonical pitch views.</li>
              <li>Compare retrieval match with classification confidence: the hybrid score ranks text relevance, while classification confidence is the graph model’s class probability.</li>
              <li>Inspect frame agreement, reconstructed geometry, direction, and class probabilities before interpreting a result.</li>
            </ol>
            <p className="research-caveat"><Info size={15} /> Player locations, possession, team identity, and ball position are reconstructed model outputs and can contain errors.</p>
          </section>
        </div>}
    </main>
  );
}
