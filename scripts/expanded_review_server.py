#!/usr/bin/env python3
"""Local review UI for the multi-match, graph-balanced tactical pool."""

from __future__ import annotations

import argparse
import json
import mimetypes
import threading
import tempfile
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PressLens — expanded tactical review</title>
<style>
:root{color-scheme:dark;font-family:Inter,ui-sans-serif,system-ui,-apple-system,sans-serif;background:#09100c;color:#eef5ef}
*{box-sizing:border-box}body{margin:0;min-width:0}.app{width:min(1240px,100%);margin:auto;padding:22px}
header{display:flex;align-items:flex-end;justify-content:space-between;gap:18px;flex-wrap:wrap;margin-bottom:16px}
h1{font-size:23px;margin:0 0 4px}.muted{color:#91a098;font-size:13px;line-height:1.45}.progress{font-variant-numeric:tabular-nums;overflow-wrap:anywhere}
.class-counts{display:flex;flex-wrap:wrap;gap:6px;margin-top:9px}.count{padding:5px 8px;border:1px solid #2d3d34;border-radius:999px;color:#b9c6be;font-size:11px}
.grid{display:grid;grid-template-columns:minmax(0,1.55fr) minmax(300px,.8fr);gap:18px;min-width:0}
.panel{min-width:0;background:#121a15;border:1px solid #28372e;border-radius:14px;padding:15px;overflow:hidden}
video{display:block;width:100%;max-width:100%;aspect-ratio:16/9;background:#000;border-radius:9px}
.match{font-size:17px;font-weight:700;margin:13px 0 3px;overflow-wrap:anywhere}.meta{display:flex;gap:7px;flex-wrap:wrap}
.chip{max-width:100%;background:#1a251f;border:1px solid #26362d;border-radius:999px;padding:6px 9px;font-size:12px;overflow-wrap:anywhere}
.prediction{display:flex;justify-content:space-between;align-items:flex-start;gap:12px;margin-bottom:13px;min-width:0}.prediction>div{min-width:0}
.eyebrow{text-transform:uppercase;letter-spacing:.09em;color:#84948a;font-size:10px;margin-bottom:5px}.label{font-size:21px;font-weight:800;overflow-wrap:anywhere}
.badge{flex:0 0 auto;color:#13200e;background:#b7f460;border-radius:999px;padding:6px 9px;font-size:11px;font-weight:800}
.metrics{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:8px;margin-bottom:13px;min-width:0}
.metric{min-width:0;background:#0c130f;border:1px solid #29382f;border-radius:10px;padding:10px}
.metric-name{color:#99a79f;font-size:11px;line-height:1.3}.metric-value{font-size:19px;font-weight:800;margin-top:4px;font-variant-numeric:tabular-nums;min-width:0;max-width:100%;overflow-wrap:anywhere;word-break:break-word}
.metric-note{color:#75857c;font-size:10px;line-height:1.35;margin-top:4px;overflow-wrap:anywhere}
.retrieval{border-left:2px solid #d5ad50;padding:2px 0 2px 10px;margin:12px 0 16px;min-width:0}.retrieval p{margin:3px 0;overflow-wrap:anywhere}
.direction{border-left:2px solid #5aa6d6;padding:2px 0 2px 10px;margin:12px 0 16px;min-width:0}.direction p{margin:3px 0;overflow-wrap:anywhere}
label{display:block;color:#a8b6ae;font-size:11px;margin:11px 0 5px}select,textarea{display:block;width:100%;max-width:100%;min-width:0;background:#0b120e;color:#eef5ef;border:1px solid #34483b;border-radius:8px;padding:9px;font:inherit}
textarea{height:72px;resize:vertical}.decision{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:8px;margin-top:13px}
button{border:0;border-radius:9px;padding:11px 9px;font-weight:800;cursor:pointer;min-width:0;overflow-wrap:anywhere}.accept{background:#b7f460;color:#111c0c}.remove{background:#54272b;color:#ffd9da;border:1px solid #804148}
.navigation{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:8px;margin-top:8px}.nav{background:#28362e;color:#edf3ef}.nav:disabled{opacity:.36;cursor:not-allowed}
.saved{min-height:18px;margin-top:9px;color:#b7f460;font-size:12px;overflow-wrap:anywhere}.saved.error{color:#ff9b9b}
@media(max-width:820px){.grid{grid-template-columns:minmax(0,1fr)}.app{padding:14px}.metrics{grid-template-columns:minmax(0,1fr)}}
@media(max-width:420px){.decision,.navigation{grid-template-columns:minmax(0,1fr)}}
</style>
</head>
<body>
<main class="app">
 <header>
  <div><h1>Expanded tactical review</h1><div class="muted">Local-only · one graph-majority label per four-second excerpt</div><div class="class-counts" id="class-counts"></div></div>
  <div class="progress" id="progress">Loading…</div>
 </header>
 <div class="grid" id="content">
  <section class="panel">
   <video id="video" controls playsinline preload="metadata"></video>
   <div class="match" id="match"></div>
   <div class="meta"><span class="chip" id="clock"></span><span class="chip" id="clip-id"></span><span class="chip" id="teams"></span><span class="chip" id="direction"></span></div>
  </section>
  <form class="panel" id="review-form">
   <div class="prediction">
    <div class="min-width"><div class="eyebrow">Graph model majority</div><div class="label" id="model-label"></div></div>
    <span class="badge" id="decision-badge">Pending</span>
   </div>
   <div class="metrics">
    <div class="metric"><div class="metric-name">Classification confidence</div><div class="metric-value" id="classification-confidence"></div><div class="metric-note">Mean model probability for this majority class in the selected excerpt.</div></div>
    <div class="metric"><div class="metric-name">Temporal agreement</div><div class="metric-value" id="temporal-agreement"></div><div class="metric-note" id="temporal-note">Frames voting for the majority class.</div></div>
   </div>
   <div class="retrieval">
    <div class="eyebrow">Candidate retrieval metadata</div>
    <p><strong id="retrieval-score"></strong> video–text cosine</p>
    <p class="muted" id="retrieval-query"></p>
    <p class="metric-note">This proposed the source clip. It is not classification confidence and did not choose the final class.</p>
   </div>
   <div class="direction">
    <div class="eyebrow">Direction evidence</div>
    <p><strong id="direction-confidence"></strong></p>
    <p id="direction-source"></p>
    <p class="metric-note" id="direction-detail"></p>
   </div>
   <label for="label-override">Label override</label><select id="label-override"></select>
   <label for="notes">Review notes</label><textarea id="notes" placeholder="Direction, possession, camera, or reconstruction issue…"></textarea>
   <div class="decision"><button type="button" class="accept" id="accept">Accept &amp; next</button><button type="button" class="remove" id="remove">Remove &amp; next</button></div>
   <div class="navigation"><button type="button" class="nav" id="previous">← Back</button><button type="button" class="nav" id="next">Next →</button></div>
   <div class="saved" id="status" aria-live="polite"></div>
  </form>
 </div>
</main>
<script>
"use strict";
const $=id=>document.getElementById(id);
let state=null,index=0,saving=false;
const pct=value=>`${Math.round(Number(value)*100)}%`;
function title(value){return state.label_titles[value]||value.replaceAll("_"," ")}
function setStatus(message,error=false){$("status").textContent=message;$("status").classList.toggle("error",error)}
function current(){return state.items[index]}
function existing(){const item=current();return item?(state.annotations[item.id]||{}):{}}
function controls(){
 $("accept").disabled=saving;$("remove").disabled=saving;
 $("previous").disabled=saving||index===0;$("next").disabled=saving;
 $("next").textContent=index===state.items.length-1?"First ↻":"Next →";
}
function refreshProgress(){
 const reviewed=state.items.filter(item=>state.annotations[item.id]?.decision).length;
 $("progress").textContent=`${reviewed} / ${state.items.length} reviewed · ${index+1} of ${state.items.length}`;
}
function show(nextIndex){
 if(!state||!state.items.length)return;
 index=Math.max(0,Math.min(Number(nextIndex),state.items.length-1));
 const item=current(),annotation=existing();
 $("video").pause();$("video").src=`/videos/${encodeURIComponent(item.video.split("/").pop())}`;$("video").load();
 $("match").textContent=item.match;$("clock").textContent=`Half ${item.half} · ${item.excerpt_match_clock_start}–${item.excerpt_match_clock_end}`;
 $("clip-id").textContent=item.id;$("teams").textContent="Neutral team clusters";
 $("direction").textContent=item.direction_required
  ?(item.direction_usable?"Direction verified":"Direction unavailable")
  :(item.direction_usable?"Direction verified (not required)":"Direction not required");
 $("model-label").textContent=title(item.model_label);
 $("classification-confidence").textContent=pct(item.classification_confidence);
 $("temporal-agreement").textContent=pct(item.temporal_agreement);
 $("temporal-note").textContent=`${item.majority_frames} of ${item.valid_graph_frames} reliable graph frames vote for this class.`;
 $("retrieval-score").textContent=item.retrieval_score==null?"n/a":Number(item.retrieval_score).toFixed(3);
 $("retrieval-query").textContent=item.retrieval_query||"No video–text query recorded";
 const directionEvidence=item.direction_evidence||{},sources=item.direction_sources||[],statuses=item.direction_statuses||[];
 $("direction-confidence").textContent=item.direction_confidence==null
  ?"Direction confidence: unavailable"
  :`Direction confidence: ${pct(item.direction_confidence)}`;
 $("direction-source").textContent=sources.length?`Source: ${sources.join(", ")}`:"Source not recorded";
 const directionRequirement=item.direction_required
  ?"Required by this canonical tactical class"
  :"Not required by this distance-based class";
 $("direction-detail").textContent=`${directionRequirement} · ${statuses.length?`Status: ${statuses.join(", ")} · `:""}${directionEvidence.usable_rows??0} / ${directionEvidence.eligible_rows??0} reliable rows usable · raw ${directionEvidence.raw_directions?.join(", ")||"n/a"}`;
 $("label-override").value=annotation.label_override||"";
 $("notes").value=annotation.notes||"";
 const decision=annotation.decision||"pending";$("decision-badge").textContent=decision[0].toUpperCase()+decision.slice(1);
 controls();
 setStatus(annotation.updated_at?`Saved ${new Date(annotation.updated_at).toLocaleString()}`:"");
 refreshProgress();
}
async function save(decision){
 if(saving)return;saving=true;controls();
 const item=current(),body={id:item.id,decision,label_override:$("label-override").value||null,notes:$("notes").value};
 try{
  const response=await fetch("/api/annotation",{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify(body)});
  if(!response.ok)throw new Error(await response.text());
  state.annotations[item.id]=await response.json();setStatus("Saved");
  if(index<state.items.length-1)show(index+1);else{show(index);setStatus("Saved · review pool complete")}
 }catch(error){show(index);setStatus(`Save failed: ${error.message}`,true)}
 finally{saving=false;controls()}
}
$("accept").onclick=()=>save("accept");$("remove").onclick=()=>save("remove");
$("previous").onclick=()=>show(index-1);$("next").onclick=()=>show(index===state.items.length-1?0:index+1);
$("review-form").onsubmit=event=>event.preventDefault();
document.addEventListener("keydown",event=>{
 if(event.target.matches("textarea,select,input"))return;
 if(event.key==="ArrowLeft")show(index-1);if(event.key==="ArrowRight")show(index===state.items.length-1?0:index+1);
 if(event.key.toLowerCase()==="a")save("accept");if(event.key.toLowerCase()==="r")save("remove");
});
fetch("/api/state").then(response=>{if(!response.ok)throw new Error("Could not load review state");return response.json()}).then(payload=>{
 state=payload;
 $("label-override").add(new Option("Use model majority",""));
 for(const label of payload.labels)$("label-override").add(new Option(title(label),label));
 for(const label of payload.labels){const count=payload.actual_counts[label]||0;const chip=document.createElement("span");chip.className="count";chip.textContent=`${title(label)} ${count}`;$("class-counts").append(chip)}
 if(!payload.items.length){$("content").innerHTML='<section class="panel">No review videos are available.</section>';$("progress").textContent="0 / 0 reviewed";return}
 const unreviewed=payload.items.findIndex(item=>!payload.annotations[item.id]?.decision);show(unreviewed<0?payload.items.length-1:unreviewed);
}).catch(error=>{$("content").innerHTML=`<section class="panel">${error.message}</section>`;$("progress").textContent="Load failed"});
</script>
</body></html>"""


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    temporary.replace(path)


def parse_range_header(value: str | None, size: int) -> tuple[int, int] | None:
    """Parse one RFC 7233 byte range, returning an inclusive interval."""

    if not value:
        return None
    if not value.startswith("bytes=") or "," in value or size <= 0:
        raise ValueError("Unsupported byte range")
    raw_start, separator, raw_end = value[6:].partition("-")
    if separator != "-":
        raise ValueError("Invalid byte range")
    try:
        if not raw_start:
            suffix = int(raw_end)
            if suffix <= 0:
                raise ValueError
            start, end = max(0, size - suffix), size - 1
        else:
            start = int(raw_start)
            end = int(raw_end) if raw_end else size - 1
            if start < 0 or start >= size or end < start:
                raise ValueError
            end = min(end, size - 1)
    except (TypeError, ValueError) as error:
        raise ValueError("Invalid byte range") from error
    return start, end


def validate_annotation(
    row: Any,
    *,
    valid_ids: set[str],
    labels: set[str],
) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise ValueError("Annotation must be an object")
    clip_id = row.get("id")
    if clip_id not in valid_ids:
        raise ValueError("Unknown review video")
    decision = row.get("decision")
    if decision not in {"accept", "remove"}:
        raise ValueError("Decision must be accept or remove")
    override = row.get("label_override")
    if override in {"", None}:
        override = None
    elif override not in labels:
        raise ValueError("Invalid tactical label override")
    notes = row.get("notes", "")
    if not isinstance(notes, str):
        raise ValueError("Notes must be text")
    if len(notes) > 4000:
        raise ValueError("Notes cannot exceed 4000 characters")
    return {
        "id": clip_id,
        "decision": decision,
        "label_override": override,
        "notes": notes,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


class ReviewServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], directory: Path) -> None:
        self.directory = directory.resolve()
        self.manifest_path = self.directory / "manifest.json"
        self.annotation_path = self.directory / "annotations.json"
        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"Build the expanded review pool first: {self.manifest_path}"
            )
        self.manifest = json.loads(self.manifest_path.read_text())
        items = self.manifest.get("items")
        if not isinstance(items, list):
            raise ValueError("Review manifest has no items array")
        self.item_by_id = {str(item["id"]): item for item in items}
        self.valid_ids = {str(item["id"]) for item in items}
        self.labels = set(map(str, self.manifest.get("labels", [])))
        self.video_by_name = {
            Path(str(item["video"])).name: (self.directory / item["video"]).resolve()
            for item in items
        }
        for name, path in self.video_by_name.items():
            if path.parent != (self.directory / "videos").resolve():
                raise ValueError(f"Unsafe video path for {name}")
        self.write_lock = threading.Lock()
        super().__init__(address, ReviewHandler)

    def annotations(self) -> dict[str, dict[str, Any]]:
        if not self.annotation_path.exists():
            return {}
        payload = json.loads(self.annotation_path.read_text())
        annotations = payload.get("annotations", {})
        return annotations if isinstance(annotations, dict) else {}

    def save(self, row: Any) -> dict[str, Any]:
        annotation = validate_annotation(
            row,
            valid_ids=self.valid_ids,
            labels=self.labels,
        )
        model_label = self.item_by_id[annotation["id"]]["model_label"]
        annotation["model_label"] = model_label
        annotation["effective_label"] = (
            annotation["label_override"] or model_label
        )
        with self.write_lock:
            annotations = self.annotations()
            annotations[annotation["id"]] = annotation
            atomic_write_json(
                self.annotation_path,
                {"schema_version": 1, "annotations": annotations},
            )
        return annotation


class ReviewHandler(BaseHTTPRequestHandler):
    server: ReviewServer

    def send_bytes(
        self,
        body: bytes,
        content_type: str,
        status: int = 200,
        *,
        cache: str = "no-store",
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        route = urlparse(self.path).path
        if route == "/":
            self.send_bytes(HTML.encode(), "text/html; charset=utf-8")
            return
        if route == "/api/state":
            annotations = self.server.annotations()
            payload = {
                "items": self.server.manifest["items"],
                "labels": self.server.manifest["labels"],
                "label_titles": self.server.manifest["label_titles"],
                "actual_counts": self.server.manifest.get("selection", {}).get(
                    "actual_counts", {}
                ),
                "annotations": annotations,
                "reviewed": sum(
                    bool(annotations.get(item["id"], {}).get("decision"))
                    for item in self.server.manifest["items"]
                ),
            }
            self.send_bytes(
                json.dumps(payload).encode(),
                "application/json; charset=utf-8",
            )
            return
        if route.startswith("/videos/"):
            requested_name = unquote(route[8:])
            if Path(requested_name).name != requested_name:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            name = requested_name
            path = self.server.video_by_name.get(name)
            if (
                path is None
                or path.parent != (self.server.directory / "videos").resolve()
                or not path.is_file()
            ):
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self.send_video(path)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_HEAD(self) -> None:
        route = urlparse(self.path).path
        if not route.startswith("/videos/"):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        requested_name = unquote(route[8:])
        if Path(requested_name).name != requested_name:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        path = self.server.video_by_name.get(requested_name)
        if path is None or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self.send_video(path, head_only=True)

    def send_video(self, path: Path, *, head_only: bool = False) -> None:
        size = path.stat().st_size
        try:
            interval = parse_range_header(self.headers.get("Range"), size)
        except ValueError:
            self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return
        start, end = interval if interval is not None else (0, size - 1)
        status = HTTPStatus.PARTIAL_CONTENT if interval is not None else HTTPStatus.OK
        self.send_response(status)
        self.send_header(
            "Content-Type", mimetypes.guess_type(path.name)[0] or "video/mp4"
        )
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Cache-Control", "private, max-age=3600")
        if interval is not None:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if head_only:
            return
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = end - start + 1
            while remaining:
                block = handle.read(min(1024 * 1024, remaining))
                if not block:
                    break
                self.wfile.write(block)
                remaining -= len(block)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/annotation":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 16_384:
                raise ValueError("Invalid request size")
            row = json.loads(self.rfile.read(length))
            saved = self.server.save(row)
            self.send_bytes(
                json.dumps(saved).encode(),
                "application/json; charset=utf-8",
            )
        except (ValueError, json.JSONDecodeError) as error:
            self.send_bytes(
                str(error).encode(),
                "text/plain; charset=utf-8",
                HTTPStatus.BAD_REQUEST,
            )

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.address_string()} - {format % args}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path("data/review/expanded_tactical"),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    server = ReviewServer((args.host, args.port), args.directory)
    print(f"Expanded tactical review: http://{args.host}:{args.port}")
    print("NDA video and annotations remain local; press Ctrl-C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
